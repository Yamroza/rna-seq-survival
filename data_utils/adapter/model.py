"""
MLP-A according to Wang et al. 2024 (Path-GPTOmic).
3-warstwowy perceptron z hidden_dim=128 (zgodnie z papierem).
"""

import torch
import torch.nn as nn


class MLP_A(nn.Module):
    """
    Input:  scGPT embedding (default 512 dim)
    Output: n_cell_types   (regresja na mixed soft labels)

    After training use get_embedding() to obtain a representation for
    bulk RNA-seq (output from before the classifier's last layer).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, n_classes: int = 17):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Obtain a representation for bulk RNA-seq."""
        with torch.no_grad():
            return self.feature_extractor(x)
