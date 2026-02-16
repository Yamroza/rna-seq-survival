import torch
import torch.nn as nn

from model.layers import MLP_Block, SNN_Block

class FNN(nn.Module):
    """ Gene-level Feedforward Neural Network (FNN) for Survival Prediction. """
    def __init__(self, rna_dims, block_type, hidden_layers, loss_fn, num_classes=1):
        super().__init__()
        self.input_dim = rna_dims
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        if block_type == 'mlp':
            self.block_type = MLP_Block
        elif block_type == 'snn':
            self.block_type = SNN_Block
        else:
            raise ValueError(f"Block type {block_type} not recognized.")

        self.init_architecture(hidden_layers)

    def init_architecture(self, hidden_layers):
        """ Initialize the architecture of the FNN. """

        ### Constructing FNN
        fc_layers = [self.block_type(in_dim=self.input_dim, out_dim=hidden_layers[0])]
        for i, _ in enumerate(hidden_layers[1:]):
            fc_layers.append(self.block_type(in_dim=hidden_layers[i], out_dim=hidden_layers[i+1]))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.classifier = nn.Linear(hidden_layers[-1], self.num_classes, bias=False)

    def forward_no_loss(self, x):
        """ Forward pass without computing the loss. """
        # Obtain representation
        h_emb = self.fc_layers(x)
        # Predict risk score
        logits  = self.classifier(h_emb)
        return logits

    def forward(self, x, label, censorship):
        """ Main forward function. """
        # Forward pass
        output = self.forward_no_loss(x)

        # Compute the loss
        output_results, output_log = self.compute_loss(output, label, censorship)

        return output_results, output_log
    
    def compute_loss(self, logits, label, censorship):
        """Compute the loss given the output of the model. Supports CoxPH loss."""
        results_dict = {'logits': logits}
        
        # Compute the coxPH loss
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
