import torch.nn as nn


class MLP(nn.Module):
    """Fully connected baseline for raw 1,024-sample vibration windows."""

    def __init__(self, input_dim=1024, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            # Flatten keeps the same 1,024 raw values presented to the CNN.
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Logits correspond to Normal, Ball, Inner Race, and Outer Race.
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
