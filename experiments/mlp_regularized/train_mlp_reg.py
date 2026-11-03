import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mlp_regularized.mlp_reg import RegularizedMLP


SEED = 42
INPUT_DIM = 1024
NUM_CLASSES = 4
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
MAX_SHIFT = 100
NOISE_STD = 0.05
GRADIENT_CLIP_NORM = 1.0
CLASS_NAMES = ["Normal", "Ball", "Inner Race", "Outer Race"]

RESULT_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_mlp_reg_model.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_split_dir():
    candidates = [
        PROJECT_ROOT / "data" / "splits",
        PROJECT_ROOT.parent / "data" / "data" / "splits",
        Path.cwd() / "data" / "splits",
        Path.cwd() / "data" / "data" / "splits",
    ]
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required):
            return candidate.resolve()
    raise FileNotFoundError("Could not find all existing train/validation/test split arrays.")


def load_splits():
    split_dir = find_split_dir()
    arrays = {
        "X_train": np.load(split_dir / "X_train.npy"),
        "y_train": np.load(split_dir / "y_train.npy"),
        "X_val": np.load(split_dir / "X_val.npy"),
        "y_val": np.load(split_dir / "y_val.npy"),
        "X_test": np.load(split_dir / "X_test.npy"),
        "y_test": np.load(split_dir / "y_test.npy"),
    }
    for name in ("X_train", "X_val", "X_test"):
        if arrays[name].shape[1:] != (INPUT_DIM,):
            raise ValueError(f"{name} has unexpected shape {arrays[name].shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    labels = set(np.concatenate([arrays["y_train"], arrays["y_val"], arrays["y_test"]]).tolist())
    if labels != set(range(NUM_CLASSES)):
        raise ValueError(f"Expected labels 0..{NUM_CLASSES - 1}, got {sorted(labels)}")
    return arrays, split_dir


def normalize_each_window(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    return ((X - means) / (stds + 1e-8)).astype(np.float32)


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).to(DEVICE),
        torch.from_numpy(y).to(DEVICE),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def augment_batch(X):
    batch_size, window_size = X.shape
    shifts = torch.randint(
        -MAX_SHIFT,
        MAX_SHIFT + 1,
        (batch_size,),
        device=X.device,
    )
    positions = torch.arange(window_size, device=X.device).unsqueeze(0)
    source_indices = (positions - shifts.unsqueeze(1)) % window_size
    shifted = X.gather(1, source_indices)
    return shifted + torch.randn_like(shifted) * NOISE_STD


def synchronize_device():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def evaluate(model, loader, criterion, collect=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            batch_size = y_batch.shape[0]
            total_loss += criterion(outputs, y_batch).item() * batch_size
            total_samples += batch_size
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
            if collect:
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    result = {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if collect:
        result.update({
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": np.concatenate(probabilities),
        })
    return result


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for X_batch, y_batch in train_loader:
            augmented = augment_batch(X_batch)
            optimizer.zero_grad()
            outputs = model(augmented)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            batch_size = y_batch.shape[0]
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            running_samples += batch_size

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch [{epoch:02d}/{MAX_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {row['val_loss']:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {row['val_acc']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    training_time = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": {
                "input_dim": INPUT_DIM,
                "hidden_dims": [128, 64],
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "max_shift": MAX_SHIFT,
                "noise_std": NOISE_STD,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "seed": SEED,
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        CHECKPOINT_PATH,
    )
    return history, best_epoch, training_time


def benchmark_inference(model, X, repeats=50, warmup=10):
    model.eval()
    batches = [X[index:index + BATCH_SIZE].to(DEVICE) for index in range(0, len(X), BATCH_SIZE)]
    with torch.no_grad():
        for _ in range(warmup):
            for batch in batches:
                model(batch)
    synchronize_device()
    durations = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            for batch in batches:
                model(batch)
            synchronize_device()
            durations.append(time.perf_counter() - start)
    per_sample_ms = np.asarray(durations) / len(X) * 1000.0
    return float(per_sample_ms.mean())


def write_history(history):
    with (RESULT_DIR / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def parse_metrics_file(path):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        match = re.match(r"([^:]+):\s*([-+]?\d*\.?\d+)", line)
        if match:
            values[match.group(1).strip()] = float(match.group(2))
    return values


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=20, ha="right")
    axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("Regularized MLP Confusion Matrix")
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return matrix


def save_curves(history):
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Regularized MLP Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_acc"] for row in history], label="Train Accuracy")
    axis.plot(epochs, [row["val_acc"] for row in history], label="Validation Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Regularized MLP Accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    global DEVICE
    set_seed(SEED)
    DEVICE = select_device()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    arrays, split_dir = load_splits()
    normalized = {name: normalize_each_window(arrays[name]) for name in ("X_train", "X_val", "X_test")}
    X_train = torch.from_numpy(normalized["X_train"]).to(DEVICE)
    X_val = torch.from_numpy(normalized["X_val"]).to(DEVICE)
    X_test = torch.from_numpy(normalized["X_test"]).to(DEVICE)
    y_train, y_val, y_test = arrays["y_train"], arrays["y_val"], arrays["y_test"]

    train_loader = make_loader(normalized["X_train"], y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(normalized["X_val"], y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(normalized["X_test"], y_test, BATCH_SIZE, shuffle=False)
    model = RegularizedMLP(INPUT_DIM, NUM_CLASSES, DROPOUT).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"Using device: {DEVICE}")
    print(f"Using split directory: {split_dir}")
    print(f"Train: {arrays['X_train'].shape}; Val: {arrays['X_val'].shape}; Test: {arrays['X_test'].shape}")
    print(f"Model: 1024 -> 128 -> 64 -> 4; Parameters: {parameter_count:,}")
    history, best_epoch, training_time = train_model(model, train_loader, val_loader)
    write_history(history)
    save_curves(history)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    train_eval = evaluate(model, make_loader(normalized["X_train"], y_train, BATCH_SIZE, False), criterion)
    val_eval = evaluate(model, val_loader, criterion)
    test_metrics = evaluate(model, test_loader, criterion, collect=True)
    inference_ms = benchmark_inference(model, X_val)
    matrix = save_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
    total_params = parameter_count
    final_train_val_gap = train_eval["accuracy"] - val_eval["accuracy"]
    baseline = parse_metrics_file(PROJECT_ROOT / "results" / "mlp" / "metrics.txt")
    cnn = parse_metrics_file(PROJECT_ROOT / "results" / "cnn" / "metrics.txt")

    metrics_path = RESULT_DIR / "metrics.txt"
    with metrics_path.open("w") as file:
        file.write("Regularized MLP Results\n")
        file.write("=======================\n")
        file.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        file.write(f"Macro Precision: {test_metrics['precision_macro']:.6f}\n")
        file.write(f"Macro Recall: {test_metrics['recall_macro']:.6f}\n")
        file.write(f"Macro F1: {test_metrics['f1_macro']:.6f}\n")
        file.write(f"Best Validation Accuracy: {val_eval['accuracy']:.6f}\n")
        file.write(f"Training Time: {training_time:.2f} seconds\n")
        file.write(f"Inference Time: {inference_ms:.4f} ms/sample\n")
        file.write(f"Total Parameters: {total_params}\n")
        file.write(f"Final Train-Validation Accuracy Gap: {final_train_val_gap:.6f}\n")
        file.write(f"Stopped Epoch: {history[-1]['epoch']}\n")
        file.write(f"Best Epoch by Validation Loss: {best_epoch}\n")

    comparison = [
        ["Old MLP", baseline.get("Test Accuracy", float("nan")), baseline.get("Macro F1", float("nan")), baseline.get("Total Parameters", float("nan")), baseline.get("Training Time", float("nan"))],
        ["Regularized MLP", test_metrics["accuracy"], test_metrics["f1_macro"], total_params, training_time],
        ["CNN", cnn.get("Test Accuracy", float("nan")), cnn.get("Macro F1", float("nan")), cnn.get("Total Parameters", float("nan")), cnn.get("Training Time", float("nan"))],
    ]
    print("\nComparison")
    print("Model | Test Accuracy | Macro F1 | Parameters | Training Time")
    for name, accuracy, f1, params, training in comparison:
        print(f"{name} | {accuracy:.6f} | {f1:.6f} | {params} | {training}")
    print(f"Final clean train accuracy: {train_eval['accuracy']:.6f}")
    print(f"Final validation accuracy: {val_eval['accuracy']:.6f}")
    print(f"Final train-validation gap: {final_train_val_gap:.6f}")
    print(f"Confusion matrix total: {int(matrix.sum())}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Results: {RESULT_DIR}")


if __name__ == "__main__":
    main()
