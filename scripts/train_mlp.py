import os
import random
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
from torch.utils.data import DataLoader, Dataset


# Make the repository root importable when run as: python scripts/train_mlp.py.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.mlp import MLP


SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
INPUT_DIM = 1024
NUM_CLASSES = 4
CLASS_NAMES = [
    "Normal",
    "Ball",
    "Inner Race",
    "Outer Race",
]

RESULT_DIR = PROJECT_DIR / "results" / "mlp"
MODEL_DIR = PROJECT_DIR / "models" / "mlp"
SPLIT_FILES = (
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class BearingDataset(Dataset):
    """Dataset returning the same raw 1,024-sample windows as the CNN."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # The MLP receives (batch, 1024); MLP.Flatten also supports a channel
        # dimension, but no channel or other feature is added here.
        return self.X[idx], self.y[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_split_dir():
    """Find the existing split directory without creating or copying data."""
    candidates = [
        PROJECT_DIR / "data" / "splits",
        PROJECT_DIR.parent / "data" / "data" / "splits",
        Path.cwd() / "data" / "splits",
        Path.cwd() / "data" / "data" / "splits",
    ]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)

        if all((candidate / filename).is_file() for filename in SPLIT_FILES):
            return candidate

    checked_paths = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find the existing CNN split files. Checked:\n"
        f"{checked_paths}"
    )


def load_splits():
    split_dir = find_split_dir()
    arrays = {
        filename.removesuffix(".npy"): np.load(split_dir / filename)
        for filename in SPLIT_FILES
    }

    if arrays["X_train"].ndim != 2 or arrays["X_train"].shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected raw windows shaped (samples, {INPUT_DIM}), "
            f"got {arrays['X_train'].shape}."
        )

    for split_name in ("X_val", "X_test"):
        if arrays[split_name].ndim != 2 or arrays[split_name].shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected {split_name} windows shaped (samples, {INPUT_DIM}), "
                f"got {arrays[split_name].shape}."
            )

    observed_labels = set(np.unique(arrays["y_train"]).tolist())
    observed_labels.update(np.unique(arrays["y_val"]).tolist())
    observed_labels.update(np.unique(arrays["y_test"]).tolist())
    if observed_labels != set(range(NUM_CLASSES)):
        raise ValueError(
            f"Expected labels 0 through {NUM_CLASSES - 1}, "
            f"got {sorted(observed_labels)}."
        )

    print(f"Using existing split directory: {split_dir}")
    return arrays


def evaluate_accuracy(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    return total_loss / len(loader), accuracy


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    best_model_path = MODEL_DIR / "best_mlp_model.pth"
    training_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        average_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_loss, val_accuracy = evaluate_accuracy(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Loss: {average_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    training_time = time.time() - training_start
    return best_model_path, best_val_acc, training_time


def synchronize_device():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def evaluate_test_set(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    synchronize_device()
    start_time = time.time()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    synchronize_device()
    total_inference_time = time.time() - start_time

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
        "inference_time": total_inference_time / len(all_labels),
        "y_true": all_labels,
        "y_pred": all_predictions,
    }


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    arrays = load_splits()
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")
    print(f"Number of classes: {NUM_CLASSES}")

    train_loader = DataLoader(
        BearingDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        BearingDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        BearingDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    model = MLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("\nStarting training...\n")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path, best_val_acc, training_time = train_model(
        model, train_loader, val_loader
    )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best model saved to: {best_model_path}")

    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_results = evaluate_test_set(model, test_loader)

    print("\n===== Test Results =====")
    print(f"Accuracy        : {test_results['accuracy']:.4f}")
    print(f"Macro Precision : {test_results['precision']:.4f}")
    print(f"Macro Recall    : {test_results['recall']:.4f}")
    print(f"Macro F1        : {test_results['f1']:.4f}")
    print(
        f"Inference Time  : "
        f"{test_results['inference_time'] * 1000:.4f} ms/sample"
    )
    print("\nConfusion Matrix:")
    print(test_results["confusion_matrix"])
    print("\nClassification Report:")
    print(
        classification_report(
            test_results["y_true"],
            test_results["y_pred"],
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    metrics_path = RESULT_DIR / "metrics.txt"
    with metrics_path.open("w") as file:
        file.write("MLP/DNN Preliminary Results\n")
        file.write("===========================\n")
        file.write(f"Test Accuracy: {test_results['accuracy']:.6f}\n")
        file.write(f"Macro Precision: {test_results['precision']:.6f}\n")
        file.write(f"Macro Recall: {test_results['recall']:.6f}\n")
        file.write(f"Macro F1: {test_results['f1']:.6f}\n")
        file.write(f"Best Validation Accuracy: {best_val_acc:.6f}\n")
        file.write(f"Training Time: {training_time:.2f} seconds\n")
        file.write(
            f"Inference Time: "
            f"{test_results['inference_time'] * 1000:.4f} ms/sample\n"
        )
        file.write(f"Total Parameters: {total_params}\n")

    cm = test_results["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(cm)
    axis.set_xticks(range(NUM_CLASSES))
    axis.set_yticks(range(NUM_CLASSES))
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("MLP Confusion Matrix")

    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"\nResults saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
