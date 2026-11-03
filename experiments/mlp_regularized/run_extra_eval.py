import csv
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mlp_regularized import train_mlp_reg as regularized
from experiments.mlp_regularized.mlp_reg import RegularizedMLP
from models.mlp import MLP


EXTRA_DIR = SCRIPT_DIR / "results_extra"
SEEDS = [42, 43, 44]
BATCH_SIZE = 128
CLASS_NAMES = ["Normal", "Ball", "Inner Race", "Outer Race"]
NUM_CLASSES = 4


def parse_metrics(path):
    values = {}
    pattern = re.compile(r"([^:]+):\s*([-+]?\d*\.?\d+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1).strip()] = float(match.group(2))
    return values


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_model(model, X, device):
    model.eval()
    batches = [X[index:index + BATCH_SIZE].to(device) for index in range(0, len(X), BATCH_SIZE)]
    with torch.no_grad():
        for _ in range(10):
            for batch in batches:
                model(batch)
    synchronize(device)
    durations = []
    with torch.no_grad():
        for _ in range(50):
            start = time.perf_counter()
            for batch in batches:
                model(batch)
            synchronize(device)
            durations.append(time.perf_counter() - start)
    per_sample_ms = np.asarray(durations) / len(X) * 1000.0
    return {
        "warmup_passes": 10,
        "timed_passes": 50,
        "batch_size": BATCH_SIZE,
        "mean_ms_per_sample": float(per_sample_ms.mean()),
        "std_ms_per_sample": float(per_sample_ms.std(ddof=1)),
        "median_ms_per_sample": float(np.median(per_sample_ms)),
        "p95_ms_per_sample": float(np.percentile(per_sample_ms, 95)),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_seed_model(seed, arrays, device):
    regularized.DEVICE = device
    processed = {name: regularized.normalize_each_window(arrays[name]) for name in ("X_train", "X_val", "X_test")}
    train_loader = regularized.make_loader(processed["X_train"], arrays["y_train"], BATCH_SIZE, shuffle=True)
    val_loader = regularized.make_loader(processed["X_val"], arrays["y_val"], BATCH_SIZE, shuffle=False)
    test_loader = regularized.make_loader(processed["X_test"], arrays["y_test"], BATCH_SIZE, shuffle=False)
    model = RegularizedMLP(regularized.INPUT_DIM, regularized.NUM_CLASSES, regularized.DROPOUT).to(device)
    seed_dir = EXTRA_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    regularized.CHECKPOINT_PATH = seed_dir / "best_model.pth"
    history, best_epoch, training_time = regularized.train_model(model, train_loader, val_loader)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=regularized.LABEL_SMOOTHING)
    train_eval_loader = regularized.make_loader(processed["X_train"], arrays["y_train"], BATCH_SIZE, shuffle=False)
    train_metrics = regularized.evaluate(model, train_eval_loader, criterion)
    val_metrics = regularized.evaluate(model, val_loader, criterion)
    test_metrics = regularized.evaluate(model, test_loader, criterion, collect=True)
    best_val_accuracy = max(row["val_acc"] for row in history)
    stopped_epoch = history[-1]["epoch"]
    clean_train_accuracy = train_metrics["accuracy"]
    train_val_gap = clean_train_accuracy - val_metrics["accuracy"]
    summary = {
        "seed": seed,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": precision_recall_fscore_support(
            test_metrics["y_true"], test_metrics["y_pred"], average="weighted", zero_division=0
        )[2],
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "clean_train_accuracy": clean_train_accuracy,
        "train_val_gap": train_val_gap,
        "training_time_seconds": training_time,
    }
    return summary, model, processed, test_metrics


def save_classification_outputs(y_true, y_pred, probabilities):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
    )
    (EXTRA_DIR / "classification_report.txt").write_text(report)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("\nSeed 42 per-class report")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    return matrix


def save_roc_outputs(y_true, probabilities):
    macro_auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    per_class = {}
    figure, axis = plt.subplots(figsize=(8, 6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_labels = (y_true == class_index).astype(int)
        class_auc = roc_auc_score(binary_labels, probabilities[:, class_index])
        per_class[class_name] = float(class_auc)
        false_positive_rate, true_positive_rate, _ = roc_curve(binary_labels, probabilities[:, class_index])
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} (AUC={class_auc:.4f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"Regularized MLP ROC Curves (Macro AUC={macro_auc:.4f})")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(EXTRA_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    lines = [f"Macro ROC-AUC: {macro_auc:.6f}"]
    lines.extend(f"{name} ROC-AUC: {value:.6f}" for name, value in per_class.items())
    (EXTRA_DIR / "roc_auc.txt").write_text("\n".join(lines) + "\n")
    return macro_auc, per_class


def write_timing(timings, device):
    lines = [f"Device: {device}", "Timing method: model.eval(), torch.no_grad(), test set preloaded on device, batch size 128, 10 warmup passes, 50 timed passes, device synchronization before/after.", ""]
    for name, result in timings.items():
        lines.append(name)
        if isinstance(result, str):
            lines.append(result)
        else:
            lines.append(f"Mean ms/sample: {result['mean_ms_per_sample']:.6f}")
            lines.append(f"Std ms/sample: {result['std_ms_per_sample']:.6f}")
            lines.append(f"Median ms/sample: {result['median_ms_per_sample']:.6f}")
            lines.append(f"P95 ms/sample: {result['p95_ms_per_sample']:.6f}")
        lines.append("")
    (EXTRA_DIR / "timing.txt").write_text("\n".join(lines))


def write_comparison(summary_rows, timings, device):
    old_mlp = parse_metrics(PROJECT_ROOT / "results" / "mlp" / "metrics.txt")
    cnn = parse_metrics(PROJECT_ROOT / "results" / "cnn" / "metrics.txt")
    mean_accuracy = np.mean([row["test_accuracy"] for row in summary_rows])
    std_accuracy = np.std([row["test_accuracy"] for row in summary_rows], ddof=1)
    mean_f1 = np.mean([row["test_f1_macro"] for row in summary_rows])
    std_f1 = np.std([row["test_f1_macro"] for row in summary_rows], ddof=1)
    seed_42 = next(row for row in summary_rows if row["seed"] == 42)
    regularized_timing = timings["Regularized MLP"]
    cnn_timing = timings.get("CNN", "Skipped: checkpoint not found.")
    cnn_timing_value = cnn_timing if isinstance(cnn_timing, str) else f"{cnn_timing['mean_ms_per_sample']:.6f}"
    table = [
        "# Extra Evaluation Comparison",
        "",
        f"Timing device: `{device}`",
        "",
        "| Model | Test accuracy | Macro F1 | Parameters | ms/sample inference |",
        "|---|---:|---:|---:|---:|",
        f"| Original MLP | {old_mlp.get('Test Accuracy', float('nan')):.6f} | {old_mlp.get('Macro F1', float('nan')):.6f} | {int(old_mlp.get('Total Parameters', 0)):,} | {old_mlp.get('Inference Time', float('nan')):.4f} |",
        f"| Regularized MLP | {mean_accuracy:.6f} +/- {std_accuracy:.6f} (seed 42: {seed_42['test_accuracy']:.6f}) | {mean_f1:.6f} +/- {std_f1:.6f} (seed 42: {seed_42['test_f1_macro']:.6f}) | {parameter_count(RegularizedMLP()) if device == 'cpu' else 140100:,} | {regularized_timing['mean_ms_per_sample']:.6f} |",
        f"| CNN | {cnn.get('Test Accuracy', float('nan')):.6f} | {cnn.get('Macro F1', float('nan')):.6f} | {int(cnn.get('Total Parameters', 0)):,} | {cnn_timing_value} |",
        "",
        "Training times are not directly comparable (different epochs, data loading and early stopping).",
    ]
    (EXTRA_DIR / "comparison_table.md").write_text("\n".join(table) + "\n")
    print("\n" + "\n".join(table))


def main():
    if EXTRA_DIR.exists():
        existing_files = [path for path in EXTRA_DIR.rglob("*") if path.is_file()]
        if existing_files:
            raise FileExistsError(f"Refusing to overwrite existing output files: {EXTRA_DIR}")
    else:
        EXTRA_DIR.mkdir(parents=True)
    regularized.set_seed(42)
    device = regularized.select_device()
    arrays, split_dir = regularized.load_splits()
    print(f"Using device: {device}")
    print(f"Using split directory: {split_dir}")
    summary_rows = []
    seed_results = {}
    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")
        regularized.set_seed(seed)
        summary, model, processed, test_metrics = load_seed_model(seed, arrays, device)
        summary_rows.append(summary)
        seed_results[seed] = (model, processed, test_metrics)
        print(summary)
    fields = list(summary_rows[0].keys())
    write_csv(EXTRA_DIR / "seeds_summary.csv", summary_rows, fields)
    print("\nThree-seed summary (sample standard deviation):")
    for field in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_f1_weighted", "best_val_accuracy", "clean_train_accuracy", "train_val_gap"]:
        values = np.asarray([row[field] for row in summary_rows], dtype=float)
        print(f"{field}: {values.mean():.6f} +/- {values.std(ddof=1):.6f}")

    seed_42_model, processed_42, seed_42_test = seed_results[42]
    seed_42_checkpoint = torch.load(EXTRA_DIR / "seed_42" / "best_model.pth", map_location=device, weights_only=False)
    seed_42_model.load_state_dict(seed_42_checkpoint["model_state_dict"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=regularized.LABEL_SMOOTHING)
    test_loader_42 = regularized.make_loader(processed_42["X_test"], arrays["y_test"], BATCH_SIZE, shuffle=False)
    seed_42_test = regularized.evaluate(seed_42_model, test_loader_42, criterion, collect=True)
    save_classification_outputs(seed_42_test["y_true"], seed_42_test["y_pred"], seed_42_test["probabilities"])
    macro_auc, class_auc = save_roc_outputs(seed_42_test["y_true"], seed_42_test["probabilities"])

    raw_test = torch.from_numpy(arrays["X_test"])
    normalized_test = torch.from_numpy(processed_42["X_test"])
    timings = {
        "Regularized MLP": benchmark_model(seed_42_model, normalized_test, device),
    }
    original_checkpoint = PROJECT_ROOT / "models" / "mlp" / "best_mlp_model.pth"
    if original_checkpoint.is_file():
        original_model = MLP().to(device)
        original_model.load_state_dict(torch.load(original_checkpoint, map_location=device, weights_only=False))
        timings["Original MLP"] = benchmark_model(original_model, raw_test, device)
    else:
        timings["Original MLP"] = "Skipped: checkpoint not found."
    cnn_checkpoint = PROJECT_ROOT / "models" / "cnn" / "best_cnn_model.pth"
    if not cnn_checkpoint.is_file():
        timings["CNN"] = "Skipped: checkpoint not found."
    write_timing(timings, device)
    write_comparison(summary_rows, timings, device)
    print(f"\nSeed 42 macro ROC-AUC: {macro_auc:.6f}")
    print("Per-class ROC-AUC:", {name: round(value, 6) for name, value in class_auc.items()})
    print("\nFiles created:")
    for path in sorted(EXTRA_DIR.rglob("*")):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
