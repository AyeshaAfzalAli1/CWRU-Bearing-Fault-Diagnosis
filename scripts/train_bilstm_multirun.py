"""
CWRU Bearing Fault Diagnosis — Bidirectional LSTM (Multi-Run)
Matches exact output format of train_rnn_lstm.py for fair comparison:
  - NUM_RUNS=5, fixed seeds
  - Saves test_accs.npy, train_times.npy, infer_times.npy, y_true.npy, y_pred.npy, meta.npy, summary.txt
"""

import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# Parameters
# =========================
NUM_RUNS    = 5
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
NUM_CLASSES = 4
SEQ_LEN     = 64    # (1024 / 16 = 64)
FEAT_DIM    = 16

# MPS > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

RESULT_DIR = os.path.join("results", "bilstm")
MODEL_DIR  = os.path.join("models",  "bilstm")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# =========================
# Seed
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# Dataset
# =========================
class BearingDatasetBiLSTM(Dataset):
    def __init__(self, X, y):
        # Z-score normalize per sample
        mean = X.mean(axis=1, keepdims=True)
        std  = X.std(axis=1,  keepdims=True) + 1e-8
        X    = (X - mean) / std
        self.X = torch.tensor(X.reshape(-1, SEQ_LEN, FEAT_DIM), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# Model
# =========================
class BiLSTM(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, hidden_dim=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        lstm_out = hidden_dim * 2   # bidirectional
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # concat last fwd + bwd hidden states
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(last)

# =========================
# Single run
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):
    print(f"\n===== Run {run_id + 1}/{NUM_RUNS} =====")

    model     = BiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    best_val_acc   = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"best_model_run{run_id+1}.pth")

    train_start = time.time()

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- validate ---
        model.eval()
        val_preds, val_labels, val_loss_total = [], [], 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss_total += criterion(logits, y).item()
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss_total / len(val_loader))
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - train_start

    # --- test with best model ---
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_preds, test_labels, infer_times = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0    = time.time()
            preds = model(x).argmax(dim=1)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.time() - t0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc       = accuracy_score(test_labels, test_preds)
    avg_infer_time = float(np.mean(infer_times))

    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")

    return (
        test_acc,
        train_time,
        avg_infer_time,
        np.array(test_labels),
        np.array(test_preds),
        best_model_path,
        best_val_acc,
    )

# =========================
# Main
# =========================
def main():
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val   = np.load("data/splits/X_val.npy")
    y_val   = np.load("data/splits/y_val.npy")
    X_test  = np.load("data/splits/X_test.npy")
    y_test  = np.load("data/splits/y_test.npy")

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Input: ({SEQ_LEN} timesteps × {FEAT_DIM} features), Z-normalized")

    train_loader = DataLoader(BearingDatasetBiLSTM(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(BearingDatasetBiLSTM(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(BearingDatasetBiLSTM(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_accs, train_times, infer_times_list = [], [], []
    seeds = []

    best_run_idx        = -1
    best_run_test_acc   = -1.0
    best_run_labels     = None
    best_run_preds      = None
    best_run_model_path = None
    best_run_val_acc    = None

    for run in range(NUM_RUNS):
        seed = 3000 + run
        seeds.append(seed)
        set_seed(seed)

        acc, t_time, i_time, labels, preds, best_path, best_val = train_and_test(
            run, train_loader, val_loader, test_loader
        )

        test_accs.append(acc)
        train_times.append(t_time)
        infer_times_list.append(i_time)

        if acc > best_run_test_acc:
            best_run_test_acc   = acc
            best_run_idx        = run
            best_run_labels     = labels
            best_run_preds      = preds
            best_run_model_path = best_path
            best_run_val_acc    = best_val

    test_accs    = np.array(test_accs,        dtype=np.float64)
    train_times  = np.array(train_times,      dtype=np.float64)
    infer_times  = np.array(infer_times_list, dtype=np.float64)

    best_overall_path = os.path.join(MODEL_DIR, "best_model_overall.pth")
    if best_run_model_path:
        shutil.copyfile(best_run_model_path, best_overall_path)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.6f}s/step (per batch)")

    # Save (same format as train_rnn_lstm.py)
    np.save(os.path.join(RESULT_DIR, "test_accs.npy"),   test_accs)
    np.save(os.path.join(RESULT_DIR, "train_times.npy"), train_times)
    np.save(os.path.join(RESULT_DIR, "infer_times.npy"), infer_times)
    np.save(os.path.join(RESULT_DIR, "y_true.npy"),      best_run_labels)
    np.save(os.path.join(RESULT_DIR, "y_pred.npy"),      best_run_preds)

    meta = {
        "model_name": "BiLSTM",
        "num_runs": NUM_RUNS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_classes": NUM_CLASSES,
        "device": DEVICE,
        "seeds": seeds,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "best_run_idx_by_test_acc": int(best_run_idx + 1),
        "best_run_test_acc": float(best_run_test_acc),
        "best_run_val_acc": float(best_run_val_acc) if best_run_val_acc else None,
        "best_model_overall_path": best_overall_path.replace("\\", "/"),
    }
    np.save(os.path.join(RESULT_DIR, "meta.npy"), meta, allow_pickle=True)

    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("BiLSTM Results\n")
        f.write(f"Test Accs: {test_accs.tolist()}\n")
        f.write(f"Mean Acc: {test_accs.mean():.6f}\n")
        f.write(f"Std Acc : {test_accs.std():.6f}\n")
        f.write(f"Avg Train Time (s): {train_times.mean():.6f}\n")
        f.write(f"Avg Inference Time (s/step, per batch): {infer_times.mean():.8f}\n")
        f.write(f"Best Run (by Test Acc): Run {best_run_idx+1}\n")
        f.write(f"Best Run Test Acc: {best_run_test_acc:.6f}\n")
        f.write(f"Best Model Overall: {best_overall_path}\n")

if __name__ == "__main__":
    main()
