import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mlp_regularized import train_mlp_reg as regularized
from experiments.mlp_regularized.mlp_reg import RegularizedMLP


OUTPUT_DIR = Path(__file__).resolve().parent / "results_clean_curves"
HISTORY_PATH = OUTPUT_DIR / "history_clean.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_curve_clean.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy_curve_clean.png"


def evaluate_clean(model, loader, criterion):
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return regularized.evaluate(model, loader, criterion)
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def train_with_clean_history(model, train_loader, val_loader, clean_train_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=regularized.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=regularized.LEARNING_RATE,
        weight_decay=regularized.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=regularized.MAX_EPOCHS,
    )
    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, regularized.MAX_EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            augmented = regularized.augment_batch(X_batch)
            optimizer.zero_grad()
            outputs = model(augmented)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), regularized.GRADIENT_CLIP_NORM)
            optimizer.step()

        val_metrics = regularized.evaluate(model, val_loader, criterion)
        clean_train_metrics = evaluate_clean(model, clean_train_loader, criterion)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss_clean": clean_train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc_clean": clean_train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= regularized.PATIENCE:
            break

    model.load_state_dict(best_state)
    return history, model


def save_history(history):
    with HISTORY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss_clean",
                "val_loss",
                "train_acc_clean",
                "val_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_plots(history):
    epochs = [row["epoch"] for row in history]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [row["train_loss_clean"] for row in history],
        label="Train (clean, eval mode)",
    )
    axis.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Regularized MLP Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [row["train_acc_clean"] for row in history],
        label="Train (clean, eval mode)",
    )
    axis.plot(epochs, [row["val_acc"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Regularized MLP Accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ACCURACY_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    regularized.set_seed(regularized.SEED)
    regularized.DEVICE = regularized.select_device()
    arrays, split_dir = regularized.load_splits()
    normalized = {
        name: regularized.normalize_each_window(arrays[name])
        for name in ("X_train", "X_val", "X_test")
    }

    train_loader = regularized.make_loader(
        normalized["X_train"], arrays["y_train"], regularized.BATCH_SIZE, shuffle=True
    )
    val_loader = regularized.make_loader(
        normalized["X_val"], arrays["y_val"], regularized.BATCH_SIZE, shuffle=False
    )
    test_loader = regularized.make_loader(
        normalized["X_test"], arrays["y_test"], regularized.BATCH_SIZE, shuffle=False
    )
    clean_train_loader = regularized.make_loader(
        normalized["X_train"], arrays["y_train"], regularized.BATCH_SIZE, shuffle=False
    )

    model = RegularizedMLP(
        regularized.INPUT_DIM,
        regularized.NUM_CLASSES,
        regularized.DROPOUT,
    ).to(regularized.DEVICE)
    history, best_model = train_with_clean_history(
        model, train_loader, val_loader, clean_train_loader
    )
    save_history(history)
    save_plots(history)

    criterion = nn.CrossEntropyLoss(label_smoothing=regularized.LABEL_SMOOTHING)
    test_metrics = regularized.evaluate(
        best_model, test_loader, criterion, collect=True
    )
    last = history[-1]
    print(f"Using split directory: {split_dir}")
    print(
        "Epoch 40: "
        f"clean train loss={last['train_loss_clean']:.6f}, "
        f"clean train accuracy={last['train_acc_clean']:.6f}, "
        f"validation loss={last['val_loss']:.6f}, "
        f"validation accuracy={last['val_acc']:.6f}"
    )
    print(
        "Best-validation-loss checkpoint test metrics: "
        f"accuracy={test_metrics['accuracy']:.6f}, "
        f"macro F1={test_metrics['f1_macro']:.6f}"
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
