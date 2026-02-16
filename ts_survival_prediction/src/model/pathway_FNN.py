import torch
import torch.nn as nn
from model.layers import MLP_Block, SNN_Block, MultiFNN


class PathwayFFN(nn.Module):
    """ Pathway-level Feedforward Neural Network (FNN) for Survival Prediction. """
    def __init__(self, rna_dims, block_type, hidden_layers, loss_fn, num_classes=1, aggregation_type='concat'):
        super().__init__()
        # input dim(s)
        self.input_dims = rna_dims
        # output dim
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        # Aggregation type (mean, sum, concat)
        self.aggregation_type = aggregation_type

        # FNN type (MLP or SNN)
        self.block_type = block_type
        if block_type == 'mlp':
            self.block_type = MLP_Block
        elif block_type == 'snn':
            self.block_type = SNN_Block
        else:
            raise ValueError(f"Block type {block_type} not recognized.")

        self.init_architecture(hidden_layers)

    def init_architecture(self, hidden_layers):
        """ Initialize the architecture of the FNN. """

        ### Constructing Transcriptomics Pathway FNN
        self.fc_layers = MultiFNN(self.block_type, self.input_dims, hidden_layers)
        if self.aggregation_type == 'concat':
            self.classifier = nn.Linear(hidden_layers[-1] * len(self.input_dims), self.num_classes, bias=False)
        else:
            self.classifier = nn.Linear(hidden_layers[-1], self.num_classes, bias=False)

    def aggregate(self, emb):
        """ Aggregate pathway embeddings. """
        if self.aggregation_type == 'mean':
            out = torch.mean(emb, dim=1)
        elif self.aggregation_type == 'sum':
            out = torch.sum(emb, dim=1)
        elif self.aggregation_type == 'concat':
            out = emb.view(emb.size(0), -1)
        else:
            raise ValueError(f"Aggregation type {self.aggregation_type} not recognized.")
        return out
    
    def forward_no_loss(self, x):
        """ Forward pass without computing the loss. """
        # Obtain pathway embeddings
        h_emb = self.fc_layers(x)
        # Aggregate pathway embeddings
        h_emb_agg = self.aggregate(h_emb)
        # Predict risk score
        logits  = self.classifier(h_emb_agg)
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
        # Load weights fro pretrained model
        state_dict = torch.load(cp_path, map_location=device)

        # Load the weights into the model
        self.load_state_dict(state_dict)

    

