import torch
import torch.nn as nn
from einops import rearrange

class MultiFNN(nn.Module):
    """ Block with multiple FNN (Ensemble). """
    def __init__(self, FNN_blocks, in_dims, hidden_dims):
        super().__init__()
        multi_fnn_network = []
        for input_dim in in_dims:
            fc_layers = [FNN_blocks(in_dim=input_dim, out_dim=hidden_dims[0])]
            for i, _ in enumerate(hidden_dims[1:]):
                fc_layers.append(FNN_blocks(in_dim=hidden_dims[i], out_dim=hidden_dims[i+1]))
        
            multi_fnn_network.append(nn.Sequential(*fc_layers))
        
        self.net = nn.ModuleList(multi_fnn_network)

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.net):
            outputs.append(module(x[i]).float())  
        return torch.stack(outputs, dim=1)
    

class MLP_Block(nn.Module):
    """
    MLP Block with ReLU and dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SNN_Block(nn.Module):
    """
    SNN (Self Normalizing Neural Network) Block (Klambauer et al., 2017)
    """
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.ELU(), 
            nn.AlphaDropout(p=dropout, inplace=False))

    def forward(self, x):
        return self.net(x)

class AttentionLayer(nn.Module):
    """
    Single attention layer in the attention module
    """

    def __init__(
            self,
            dim=256,
            dim_head=64,
            heads=1,
            eps=1e-8,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.inner_dim = heads * dim_head
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

    
    def forward(self, x, return_attention=False):
        x_norm = self.norm(x)

        # derive query, keys, values 
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # regular transformer scaling
        q = q * self.scale

        einops_eq = '... i d, ... j d -> ... i j'
        pre_soft_attn_matrix = torch.einsum(einops_eq, q, k)

        attn_matrix = pre_soft_attn_matrix.softmax(dim=-1)

        out  = attn_matrix @ v

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)

        if return_attention:
            return out, attn_matrix.squeeze().detach().cpu()
    
        return out


class PrototypeAggregator(nn.Module):
    """ Aggregates embeddings using learnable prototype weights. Based on ABMIL (Ilse et al., 2018)."""
    def __init__(self, embedding_dim, num_prototypes):
        super(PrototypeAggregator, self).__init__()
        self.embedding_dim = embedding_dim 
        self.num_prototypes = num_prototypes  

        # Learnable linear layer to produce scalar weights for prototypes
        self.weight_generator = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings, dim):
        """
        Main forward function.
        """
        # Generate weights for each prototype (shape: [B, num_prototypes, 1])
        weights = self.weight_generator(embeddings)  # [B, num_prototypes, 1]

        # Normalize weights across the num_prototypes using softmax
        normalized_weights = nn.functional.softmax(weights, dim=dim)  # [B, num_prototypes, 1]

        # Perform weighted sum across the prototype dimension
        weighted_sum = (embeddings * normalized_weights).sum(dim=dim)  # [B, 144]

        return weighted_sum
    