import torch.nn as nn


class ConfigurableMLP(nn.Module):
    """Fully connected classifier for raw 1,024-sample vibration windows."""

    def __init__(self, input_dim=1024, hidden_dims=(256, 128), num_classes=4, dropout=0.3):
        super().__init__()
        layers = [nn.Flatten()]
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
