"""
CWRU Bearing Fault Diagnosis — LSTM Classifier
================================================
Architecture : 2-layer Bidirectional LSTM → Dense → 4-class Softmax
Dataset      : CWRU 12K Drive-End (Normal, Ball, Inner Race, Outer Race)

Key fixes vs v1:
  1. Input RESHAPED: (N, 1024) → (N, 64, 16)
       64 timesteps × 16 features/step → 16× shorter sequences
       This prevents vanishing gradients over 1024 raw steps.
  2. Per-sample Z-score normalization (mean=0, std=1)
  3. Bidirectional LSTM (captures both directions)
  4. LayerNorm before classifier
  5. Label smoothing (0.05) + weight decay
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# =========================================================
# 0. Configuration
# =========================================================
SPLITS_DIR  = "data/splits"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN     = 64    # LSTM timesteps  (1024 / 16 = 64)
INPUT_SIZE  = 16    # features per timestep
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race", "Outer Race"]

BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
LR_PATIENCE = 7
ES_PATIENCE = 15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")


# =========================================================
# 1. Load & Preprocess Data
# =========================================================
def normalize(X):
    """Per-sample Z-score: zero mean, unit variance per window."""
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1,  keepdims=True) + 1e-8
    return (X - mean) / std


def reshape_for_lstm(X):
    """
    (N, 1024) → (N, 64, 16)
    Group every 16 consecutive samples into one LSTM timestep.
    Reduces effective sequence length from 1024 → 64.
    """
    return X.reshape(X.shape[0], SEQ_LEN, INPUT_SIZE)


def load_and_prepare(splits_dir):
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(splits_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(splits_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(splits_dir, "y_test.npy"))

    for arr, name in [(X_train, "X_train"), (X_val, "X_val"), (X_test, "X_test")]:
        print(f"  {name} raw shape: {arr.shape}")

    X_train = reshape_for_lstm(normalize(X_train)).astype(np.float32)
    X_val   = reshape_for_lstm(normalize(X_val)).astype(np.float32)
    X_test  = reshape_for_lstm(normalize(X_test)).astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


print("[INFO] Loading and preprocessing dataset splits...")
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare(SPLITS_DIR)

print(f"\n  Train : {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
print(f"  Input reshaped to (N, seq={SEQ_LEN}, features={INPUT_SIZE})")
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt} train samples")

train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)


# =========================================================
# 2. Bidirectional LSTM Model
# =========================================================
class BearingLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM for bearing fault classification.
    Input  : (batch, 64, 16)
    Output : (batch, 4)
    """

    def __init__(
        self,
        input_size    = INPUT_SIZE,
        hidden_size   = 128,
        num_layers    = 2,
        num_classes   = NUM_CLASSES,
        dropout       = 0.3,
        bidirectional = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.classifier(last_hidden)


model = BearingLSTM().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[INFO] Model parameters: {total_params:,}")
print(model)


# =========================================================
# 3. Loss, Optimizer, Scheduler
# =========================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=LR_PATIENCE,
)


# =========================================================
# 4. Training Loop
# =========================================================
def run_epoch(loader, model, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total


print("\n" + "="*62)
print("  Training Bidirectional LSTM — CWRU Bearing Fault Diagnosis")
print(f"  Input: (N, {SEQ_LEN} timesteps × {INPUT_SIZE} features), Z-normalized")
print("="*62)

best_val_loss = float("inf")
best_epoch    = 0
es_counter    = 0
history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer)
    val_loss,   val_acc   = run_epoch(val_loader,   model, criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:>3}/{EPOCHS}  "
        f"| Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%"
        f"  | Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%"
        f"  | LR: {lr_now:.2e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_lstm_model.pt"))
        es_counter = 0
        print(f"         ↳ ✅ New best saved (val_loss={best_val_loss:.4f})")
    else:
        es_counter += 1
        if es_counter >= ES_PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

print(f"\n[INFO] Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")


# =========================================================
# 5. Training Curves
# =========================================================
def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "b-o", ms=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=3, label="Val Loss")
    axes[0].set_title("Loss per Epoch", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], "b-o", ms=3, label="Train Acc")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   "r-o", ms=3, label="Val Acc")
    axes[1].set_title("Accuracy per Epoch", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


plot_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))


# =========================================================
# 6. Test Evaluation
# =========================================================
print("\n" + "="*62)
print("  Test Set Evaluation (loading best model weights)")
print("="*62)

model.load_state_dict(
    torch.load(os.path.join(RESULTS_DIR, "best_lstm_model.pt"), map_location=DEVICE)
)
test_loss, test_acc = run_epoch(test_loader, model, criterion)
print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc*100:.2f}%")

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(DEVICE))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)


# =========================================================
# 7. Classification Report
# =========================================================
report = classification_report(
    all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
)
print("\nClassification Report:\n")
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc*100:.4f}%\n\n")
    f.write(report)
print(f"[INFO] Report saved → {RESULTS_DIR}/classification_report.txt")


# =========================================================
# 8. Confusion Matrix
# =========================================================
def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES,
                      os.path.join(RESULTS_DIR, "confusion_matrix.png"))

print("\n" + "="*62)
print(f"  ✅  Final Test Accuracy: {test_acc*100:.2f}%")
print("="*62)
print(f"\nAll results saved in: {os.path.abspath(RESULTS_DIR)}/")
