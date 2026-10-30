import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from models.mlp_variants import ConfigurableMLP


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "mlp_experiments"
INPUT_DIM = 1024
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "Ball", "Inner Race", "Outer Race"]


class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


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
        PROJECT_DIR / "data" / "splits",
        PROJECT_DIR.parent / "data" / "data" / "splits",
        Path.cwd() / "data" / "splits",
        Path.cwd() / "data" / "data" / "splits",
    ]
    names = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in names):
            return candidate.resolve()
    raise FileNotFoundError("Could not locate all six existing split arrays.")


def load_splits():
    split_dir = find_split_dir()
    arrays = {name[:-4]: np.load(split_dir / name) for name in [
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"
    ]}
    for name in ("X_train", "X_val", "X_test"):
        if arrays[name].shape[1:] != (INPUT_DIM,):
            raise ValueError(f"{name} must have shape (samples, {INPUT_DIM}), got {arrays[name].shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("y_train", "y_val", "y_test"):
        if arrays[name].ndim != 1 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} is invalid")
    labels = set(np.concatenate([arrays["y_train"], arrays["y_val"], arrays["y_test"]]).tolist())
    if labels != set(range(NUM_CLASSES)):
        raise ValueError(f"Expected labels 0..{NUM_CLASSES - 1}, got {sorted(labels)}")
    return arrays, split_dir


def apply_preprocessing(arrays, mode):
    result = {key: value.copy() for key, value in arrays.items()}
    stats = {"mode": mode}
    if mode == "raw":
        stats.update({"mean": 0.0, "std": 1.0})
        return result, stats
    if mode != "standardized":
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    mean = float(result["X_train"].mean())
    std = float(result["X_train"].std())
    if std == 0.0:
        raise ValueError("Training standard deviation is zero")
    for name in ("X_train", "X_val", "X_test"):
        result[name] = ((result[name] - mean) / std).astype(np.float32)
    stats.update({"mean": mean, "std": std})
    return result, stats


def make_loaders(arrays, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset = BearingDataset(arrays["X_train"], arrays["y_train"])
    train_eval_dataset = BearingDataset(arrays["X_train"], arrays["y_train"])
    val_dataset = BearingDataset(arrays["X_val"], arrays["y_val"])
    test_dataset = BearingDataset(arrays["X_test"], arrays["y_test"])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0),
        DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def evaluate_model(model, loader, criterion, device, collect=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
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
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if collect:
        metrics.update({"y_true": y_true, "y_pred": y_pred, "probabilities": np.concatenate(probabilities)})
    return metrics


def is_better(candidate, best):
    if best is None:
        return True
    if candidate["f1_macro"] > best["f1_macro"] + 1e-12:
        return True
    return abs(candidate["f1_macro"] - best["f1_macro"]) <= 1e-12 and candidate["loss"] < best["loss"]


def train_one(config, arrays, seed, device=None, checkpoint_path=None, verbose=True):
    set_seed(seed)
    device = device or select_device()
    processed, preprocessing_stats = apply_preprocessing(arrays, config["preprocessing"])
    train_loader, train_eval_loader, val_loader, test_loader = make_loaders(processed, config["batch_size"], seed)
    model = ConfigurableMLP(
        input_dim=INPUT_DIM,
        hidden_dims=tuple(config["hidden_dims"]),
        num_classes=NUM_CLASSES,
        dropout=config["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = None
    if config["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
    history = []
    best_state = None
    best_metrics = None
    best_epoch = 0
    no_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y_batch.shape[0]
            seen += y_batch.shape[0]

        train_metrics = evaluate_model(model, train_eval_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "learning_rate": current_lr,
        }
        history.append(row)
        if verbose:
            print(
                f"Epoch [{epoch:02d}/{config['max_epochs']}] | "
                f"Train Loss {row['train_loss']:.4f} | Train Acc {row['train_accuracy']:.4f} | "
                f"Val Loss {row['val_loss']:.4f} | Val Acc {row['val_accuracy']:.4f} | "
                f"Val Macro F1 {row['val_f1_macro']:.4f}"
            )
        if is_better(val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": config,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_validation_metrics": best_metrics,
                        "preprocessing_stats": preprocessing_stats,
                    },
                    checkpoint_path,
                )
        else:
            no_improvement += 1
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        if no_improvement >= config["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "training_time": training_time,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "preprocessing_stats": preprocessing_stats,
        "test_loader": test_loader,
        "processed_arrays": processed,
        "device": device,
    }


def benchmark_inference(model, X, device, warmup=10, repeats=50, batch_size=64):
    model.eval()
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False, num_workers=0)
    batches = [batch.to(device) for batch in loader]
    with torch.no_grad():
        for _ in range(warmup):
            for batch in batches:
                model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            for batch in batches:
                model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
    total_samples = len(X)
    per_sample_ms = np.asarray(times) / total_samples * 1000.0
    return {
        "warmup_runs": warmup,
        "repeats": repeats,
        "mean_ms_per_sample": float(per_sample_ms.mean()),
        "std_ms_per_sample": float(per_sample_ms.std(ddof=1)),
        "median_ms_per_sample": float(np.median(per_sample_ms)),
        "p95_ms_per_sample": float(np.percentile(per_sample_ms, 95)),
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file, indent=2, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item))


def write_history(path, history):
    write_csv(path, history)


def save_predictions(path, y_true, y_pred, probabilities):
    rows = []
    for index, (true_label, predicted_label, probability) in enumerate(zip(y_true, y_pred, probabilities)):
        row = {"index": index, "true_label": int(true_label), "predicted_label": int(predicted_label)}
        row.update({f"probability_{name}": float(probability[class_index]) for class_index, name in enumerate(CLASS_NAMES)})
        rows.append(row)
    write_csv(path, rows)
    np.savez(path.with_suffix(".npz"), y_true=y_true, y_pred=y_pred, probabilities=probabilities)


def plot_training_curves(history, output_dir):
    epochs = [row["epoch"] for row in history]
    for metric, title, ylabel, filename in [
        ("loss", "Training and Validation Loss", "Loss", "loss_curves.png"),
        ("accuracy", "Training and Validation Accuracy", "Accuracy", "accuracy_curves.png"),
    ]:
        figure, axis = plt.subplots(figsize=(8, 5))
        if metric == "loss":
            train_values = [row["train_loss"] for row in history]
            val_values = [row["val_loss"] for row in history]
        else:
            train_values = [row["train_accuracy"] for row in history]
            val_values = [row["val_accuracy"] for row in history]
        axis.plot(epochs, train_values, label="Training")
        axis.plot(epochs, val_values, label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_confusion_matrices(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for matrix, title, filename, format_string in [
        (cm, "MLP Confusion Matrix", "confusion_matrix_raw.png", "d"),
        (normalized, "MLP Row-Normalized Confusion Matrix", "confusion_matrix_normalized.png", ".2f"),
    ]:
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")
        axis.set_title(title)
        for row in range(NUM_CLASSES):
            for column in range(NUM_CLASSES):
                axis.text(column, row, format(matrix[row, column], format_string), ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return cm, normalized


def plot_per_class_metrics(y_true, y_pred, output_dir):
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {metric: [report[name][metric] for name in CLASS_NAMES] for metric in ("precision", "recall", "f1-score")}
    x = np.arange(NUM_CLASSES)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, (metric, values) in enumerate(metrics.items()):
        axis.bar(x + (offset - 1) * width, values, width, label=metric.title())
    axis.set_xticks(x, CLASS_NAMES, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Per-Class Precision, Recall, and F1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return report


def plot_roc_pr(y_true, probabilities, output_dir):
    figure, axis = plt.subplots(figsize=(8, 6))
    all_fpr = np.linspace(0, 1, 200)
    interpolated = []
    roc_auc = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary = (y_true == class_index).astype(int)
        fpr, tpr, _ = roc_curve(binary, probabilities[:, class_index])
        roc_auc[class_name] = float(auc(fpr, tpr))
        interpolated.append(np.interp(all_fpr, fpr, tpr))
        axis.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc[class_name]:.3f})")
    macro_auc = float(np.mean(list(roc_auc.values())))
    axis.plot(all_fpr, np.mean(interpolated, axis=0), "k--", label=f"Macro mean (AUC={macro_auc:.3f})")
    axis.plot([0, 1], [0, 1], "gray", linestyle=":")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Multiclass One-vs-Rest ROC Curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    pr_auc = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary = (y_true == class_index).astype(int)
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, class_index])
        pr_auc[class_name] = float(auc(recall, precision))
        axis.plot(recall, precision, label=f"{class_name} (AUC={pr_auc[class_name]:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("One-vs-Rest Precision-Recall Curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"roc_auc": roc_auc, "macro_roc_auc": macro_auc, "pr_auc": pr_auc}


def plot_class_distribution(arrays, output_dir):
    counts = []
    for split in ("train", "val", "test"):
        labels = arrays[f"y_{split}"]
        counts.append([int(np.sum(labels == class_index)) for class_index in range(NUM_CLASSES)])
    counts = np.asarray(counts)
    x = np.arange(NUM_CLASSES)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, split in enumerate(("Train", "Validation", "Test")):
        axis.bar(x + (offset - 1) * width, counts[offset], width, label=split)
    axis.set_xticks(x, CLASS_NAMES, rotation=20, ha="right")
    axis.set_ylabel("Number of Windows")
    axis.set_title("Class Distribution by Split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return counts.tolist()


def plot_waveform_examples(X, y_true, y_pred, output_dir):
    correct = np.flatnonzero(y_true == y_pred)[:3]
    incorrect = np.flatnonzero(y_true != y_pred)[:3]
    for indices, prefix, title in [(correct, "correct", "Correctly Classified Waveforms"), (incorrect, "incorrect", "Incorrectly Classified Waveforms")]:
        if len(indices) == 0:
            continue
        figure, axes = plt.subplots(len(indices), 1, figsize=(10, max(3, 2.5 * len(indices))), squeeze=False)
        for axis, index in zip(axes[:, 0], indices):
            axis.plot(X[index], linewidth=0.7)
            axis.set_title(f"True: {CLASS_NAMES[int(y_true[index])]} | Predicted: {CLASS_NAMES[int(y_pred[index])]}" if prefix == "incorrect" else f"Class: {CLASS_NAMES[int(y_true[index])]}" )
            axis.set_xlabel("Sample")
            axis.set_ylabel("Amplitude")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_dir / f"{prefix}_waveforms.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_hyperparameters(rows, output_dir):
    valid = [row for row in rows if row.get("status") == "success"]
    if not valid:
        return
    labels = [str(row["trial_id"]) for row in valid]
    values = [float(row["best_val_f1_macro"]) for row in valid]
    colors = ["#2a9d8f" if row["preprocessing"] == "standardized" else "#457b9d" for row in valid]
    order = np.argsort(values)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(np.arange(len(order)), [values[index] for index in order], color=[colors[index] for index in order])
    axis.set_xticks(np.arange(len(order)), [labels[index] for index in order], rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Best Validation Macro F1")
    axis.set_title("Hyperparameter Trial Comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "hyperparameter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_baseline_comparison(baseline, tuned, output_dir):
    names = ["Baseline", "Selected MLP"]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    values = [[float(baseline[metric]), float(tuned[metric])] for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, [item[0] for item in values], width, label=names[0])
    axis.bar(x + width / 2, [item[1] for item in values], width, label=names[1])
    axis.set_xticks(x, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Original Baseline vs Selected MLP")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_vs_selected.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
