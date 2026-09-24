import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Configuration
# =========================
SEED = 42
BATCH_SIZE = 64
NUM_CLASSES = 4

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# =========================
# Reproducibility
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Dataset
# =========================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert a 1024-point signal to shape [1, 1024]
        return self.X[idx].unsqueeze(0), self.y[idx]


# =========================
# 1D-CNN Model
# =========================
class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# Test Model and Data
# =========================
def main():

    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    # Load common training split
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    train_dataset = BearingDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CNN1D(NUM_CLASSES).to(DEVICE)

    # Take one batch to verify the complete pipeline
    X_batch, y_batch = next(iter(train_loader))

    X_batch = X_batch.to(DEVICE)

    with torch.no_grad():
        output = model(X_batch)

    print(f"Input batch shape : {X_batch.shape}")
    print(f"Output shape      : {output.shape}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Total parameters  : {total_params:,}")
    print("Baseline 1D-CNN forward pass successful.")


if __name__ == "__main__":
    main()
