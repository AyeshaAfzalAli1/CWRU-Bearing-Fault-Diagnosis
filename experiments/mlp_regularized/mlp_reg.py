import torch
import torch.nn as nn


class RegularizedMLP(nn.Module):
    """Small fully connected classifier for raw 1,024-sample windows."""

    def __init__(self, input_dim=1024, num_classes=4, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)
