import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# ==========================================
# Configuration (Exact match with train_cnn.py)
# ==========================================
SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

RESULT_DIR = os.path.join("results", "cnn")
os.makedirs(RESULT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
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
        return self.classifier(x)

def evaluate_metrics(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels, predictions)
    return avg_loss, acc

def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    # Load splits
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")

    train_dataset = BearingDataset(X_train, y_train)
    val_dataset = BearingDataset(X_val, y_val)

    # Note: to guarantee identical shuffle order across runs, generator with seed is used
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CNN1D(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("\nStarting retraining to record epoch history (preserving existing models and results)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_correct += (pred == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate_metrics(model, val_loader, criterion)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"\nRetraining completed in {elapsed:.2f} seconds.")

    # Save history to CSV (non-destructive)
    df_history = pd.DataFrame(history)
    csv_path = os.path.join(RESULT_DIR, "training_history.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"Saved training history to: {csv_path}")

    # ==========================================
    # 1. Training vs Validation Loss Curve
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    epochs_range = history["epoch"]
    ax.plot(epochs_range, history["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)

    ax.set_title("1D-CNN Training vs Validation Loss over 30 Epochs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Loss (Cross-Entropy)", fontsize=11, fontweight="semibold")
    ax.set_xticks(range(1, EPOCHS + 1, 2) if EPOCHS > 15 else range(1, EPOCHS + 1))
    ax.set_xlim(0.5, EPOCHS + 0.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, framealpha=0.95, fontsize=10, loc="upper right")

    loss_fig_path = os.path.join(RESULT_DIR, "training_validation_loss.png")
    fig.tight_layout()
    fig.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to: {loss_fig_path}")

    # ==========================================
    # 2. Training vs Validation Accuracy Curve
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(epochs_range, history["train_acc"], label="Training Accuracy", color="#2ca02c", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs_range, history["val_acc"], label="Validation Accuracy", color="#d62728", linewidth=2.2, marker="s", markersize=4)

    ax.set_title("1D-CNN Training vs Validation Accuracy over 30 Epochs", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="semibold")
    ax.set_xticks(range(1, EPOCHS + 1, 2) if EPOCHS > 15 else range(1, EPOCHS + 1))
    ax.set_xlim(0.5, EPOCHS + 0.5)
    ax.set_ylim(min(min(history["train_acc"]), min(history["val_acc"])) - 0.05, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, framealpha=0.95, fontsize=10, loc="lower right")

    acc_fig_path = os.path.join(RESULT_DIR, "training_validation_accuracy.png")
    fig.tight_layout()
    fig.savefig(acc_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy curve to: {acc_fig_path}")

if __name__ == "__main__":
    main()
