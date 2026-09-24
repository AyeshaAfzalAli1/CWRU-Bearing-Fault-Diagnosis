import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# Configuration
# =========================
SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

RESULT_DIR = os.path.join("results", "cnn")
MODEL_DIR = os.path.join("models", "cnn")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
        return self.X[idx].unsqueeze(0), self.y[idx]


# =========================
# 1D-CNN Model
# =========================
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


# =========================
# Evaluation
# =========================
def evaluate(model, loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    return accuracy_score(labels, predictions)


# =========================
# Training
# =========================
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_acc = 0.0

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_cnn_model.pth"
    )

    training_start = time.time()

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        val_acc = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Loss: {average_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # Save model only when validation performance improves
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                best_model_path
            )

    training_time = time.time() - training_start

    return best_model_path, best_val_acc, training_time


def evaluate_test_set(model, test_loader):
    model.eval()

    all_predictions = []
    all_labels = []
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )

    # Synchronize MPS before stopping timer
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()

    total_inference_time = time.time() - start_time

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    average_inference_time = (
        total_inference_time / len(all_labels)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "inference_time": average_inference_time,
        "y_true": all_labels,
        "y_pred": all_predictions
    }

# =========================
# Main
# =========================


def main():

    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    # Load common dataset splits
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")

    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")

    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")

    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # Create datasets
    train_dataset = BearingDataset(
        X_train,
        y_train
    )

    val_dataset = BearingDataset(
        X_val,
        y_val
    )

    test_dataset = BearingDataset(
        X_test,
        y_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Create model
    model = CNN1D(NUM_CLASSES).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Total parameters: {total_params:,}")
    print("\nStarting training...\n")

    # Train model
    best_model_path, best_val_acc, training_time = train_model(
        model,
        train_loader,
        val_loader
    )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best model saved to: {best_model_path}")
    # =========================
    # Final Test Evaluation
    # =========================

    print("\nEvaluating best model on test set...")

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE
        )
    )

    test_results = evaluate_test_set(
        model,
        test_loader
    )

    print("\n===== Test Results =====")
    print(
        f"Accuracy        : "
        f"{test_results['accuracy']:.4f}"
    )
    print(
        f"Macro Precision : "
        f"{test_results['precision']:.4f}"
    )
    print(
        f"Macro Recall    : "
        f"{test_results['recall']:.4f}"
    )
    print(
        f"Macro F1        : "
        f"{test_results['f1']:.4f}"
    )
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
            target_names=[
                "Normal",
                "Ball",
                "Inner Race",
                "Outer Race"
            ],
            digits=4
        )
    )


if __name__ == "__main__":
    main()
