import torch
import torch.nn as nn
from model.layers import SNN_Block, MultiFNN, AttentionLayer, PrototypeAggregator

class Gene_DIMAF(nn.Module):
    """ DIMAF model for Single Modality (SD) using gene expression data. """
    def __init__(self, rna_dims, loss_fn, num_classes=1, aggregation_type='wm'):
        super().__init__()
        self.input_dims = rna_dims
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.aggregation_type = aggregation_type
    
        self.init_architecture()

    def init_architecture(self):
        """ Initialize the architecture of the FNN. """

        # Constructing Transcriptomics Pathway FNN
        self.pw_fc_layers = MultiFNN(SNN_Block, self.input_dims, [256, 256])
        self.single_out_dim, self.rna_pt_embedding = self.append_pt_embed()
        multi_out_dim = self.single_out_dim // 2

        # Self attention block
        self.rna_attention = AttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)
        
        self.layer_norm = nn.LayerNorm(multi_out_dim)

        # Aggregation layer for the pathway embeddings 
        if self.aggregation_type == 'mean':
            self.aggregator = torch.mean
        elif self.aggregation_type == 'wm':
            self.aggregator = PrototypeAggregator(multi_out_dim, 50)
        else:
            raise ValueError(f"Aggregation type {self.aggregation_type} not supported in DIMAF.")
        
        # Risk perdiction 
        self.classifier = nn.Linear(multi_out_dim, self.num_classes, bias=False)
        

    def append_pt_embed(self):
        """ Obtain a learnable pathway embedding to the representation. """
        append_dim = 32
        path_proj_dim_new = 256 + append_dim

        gene_embedding = torch.nn.Parameter(torch.randn(1, 50, append_dim), requires_grad=True)

        return path_proj_dim_new, gene_embedding
    
    def forward_no_loss(self, x):
        """ Forward pass without computing the loss. """
        # Pathway embeddings
        h_emb = self.pw_fc_layers(x)
        # Append prototype embedding
        pt_exp = self.rna_pt_embedding.expand(h_emb.shape[0], -1, -1)
        h_emb_pt = torch.cat([h_emb, pt_exp], dim=-1)
        # Attention
        h_emb_post_att = self.rna_attention(h_emb_pt)
        h_emb_norm = self.layer_norm(h_emb_post_att)
        # Aggregate
        h_emb_aggr = self.aggregator(h_emb_norm, dim=1)
        # Risk prediction
        logits  = self.classifier(h_emb_aggr)
        return logits

    def forward(self, x, label, censorship):
        """ Main forward function"""
        # Forward pass
        output = self.forward_no_loss(x)

        # Compute the total loss
        output_results, output_log = self.compute_loss(output, label, censorship)

        return output_results, output_log
    
    def compute_loss(self, logits, label, censorship):
        """Compute the loss given the output of the model. Supports CoxPH loss."""
        results_dict = {'logits': logits}
        
        total_loss, log_dict = self.loss_fn(logits=logits, times=label, censorships=censorship)
        risk = torch.exp(logits)
        results_dict['risk'] = risk
        results_dict['loss'] = total_loss

        return results_dict, log_dict

    def from_pretrained(self, cp_path, device):
        # Load weights from pretrained model
        state_dict = torch.load(cp_path, map_location=device)

        # Load the weights into the model
        self.load_state_dict(state_dict)

    

