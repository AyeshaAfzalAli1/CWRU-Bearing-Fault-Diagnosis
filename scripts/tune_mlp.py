import itertools
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.mlp_variants import ConfigurableMLP
from scripts.mlp_experiment import (
    CLASS_NAMES,
    INPUT_DIM,
    NUM_CLASSES,
    RESULTS_DIR,
    benchmark_inference,
    load_splits,
    plot_baseline_comparison,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_hyperparameters,
    plot_per_class_metrics,
    plot_roc_pr,
    plot_training_curves,
    plot_waveform_examples,
    save_predictions,
    select_device,
    train_one,
    write_csv,
    write_json,
    write_history,
)


MAX_EPOCHS = 60
PATIENCE = 8
SEARCH_SEED = 42
ROBUSTNESS_SEEDS = [42, 43, 44]


def parse_baseline_metrics():
    path = PROJECT_DIR / "results" / "mlp" / "metrics.txt"
    values = {}
    patterns = {
        "accuracy": "Test Accuracy",
        "precision_macro": "Macro Precision",
        "recall_macro": "Macro Recall",
        "f1_macro": "Macro F1",
    }
    for line in path.read_text().splitlines():
        for key, label in patterns.items():
            if line.startswith(label + ":"):
                values[key] = float(line.split(":", 1)[1].strip())
    return values


def build_search_configs():
    architecture_options = [(128, 64), (256, 128), (512, 256)]
    dropout_options = [0.2, 0.3, 0.5]
    learning_rate_options = [0.001, 0.0005, 0.0001]
    weight_decay_options = [0.0, 0.00001, 0.0001, 0.001]
    batch_size_options = [32, 64, 128]
    scheduler_options = ["fixed", "plateau"]
    pool = [
        {
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler": scheduler,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        }
        for hidden_dims, dropout, learning_rate, weight_decay, batch_size, scheduler in itertools.product(
            architecture_options,
            dropout_options,
            learning_rate_options,
            weight_decay_options,
            batch_size_options,
            scheduler_options,
        )
    ]
    baseline = {
        "hidden_dims": [256, 128],
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 64,
        "scheduler": "fixed",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
    }
    pool = [config for config in pool if config != baseline]
    random.Random(SEARCH_SEED).shuffle(pool)
    return [baseline] + pool[:19]


def run_trial(trial_id, config, preprocessing, arrays, device, tuning_dir):
    full_config = dict(config)
    full_config["preprocessing"] = preprocessing
    print(f"\n===== Trial {trial_id}: {preprocessing} {full_config} =====")
    trial_dir = tuning_dir / "trials" / f"trial_{trial_id:03d}_{preprocessing}"
    started = time.perf_counter()
    try:
        result = train_one(full_config, arrays, seed=SEARCH_SEED, device=device, verbose=True)
        duration = time.perf_counter() - started
        write_history(trial_dir / "training_history.csv", result["history"])
        metrics = result["best_validation"]
        payload = {
            "trial_id": trial_id,
            "config": full_config,
            "best_epoch": result["best_epoch"],
            "best_validation": metrics,
            "training_time_seconds": duration,
            "parameter_count": result["parameter_count"],
            "preprocessing_stats": result["preprocessing_stats"],
            "status": "success",
        }
        write_json(trial_dir / "trial_record.json", payload)
        return {
            "trial_id": trial_id,
            **full_config,
            "preprocessing": preprocessing,
            "best_epoch": result["best_epoch"],
            "best_val_loss": metrics["loss"],
            "best_val_accuracy": metrics["accuracy"],
            "best_val_precision_macro": metrics["precision_macro"],
            "best_val_recall_macro": metrics["recall_macro"],
            "best_val_f1_macro": metrics["f1_macro"],
            "training_time_seconds": duration,
            "parameter_count": result["parameter_count"],
            "status": "success",
        }
    except Exception as error:
        payload = {"trial_id": trial_id, "config": full_config, "status": "failed", "error": repr(error)}
        write_json(trial_dir / "trial_record.json", payload)
        print(f"Trial failed: {error}")
        return {
            "trial_id": trial_id,
            **full_config,
            "preprocessing": preprocessing,
            "status": "failed",
            "error": repr(error),
        }


def choose_best(rows):
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        raise RuntimeError("All tuning trials failed")
    return max(successful, key=lambda row: (float(row["best_val_f1_macro"]), -float(row["best_val_loss"])))


def run_robustness(selected, arrays, device, output_dir):
    rows = []
    config = {key: selected[key] for key in ("hidden_dims", "dropout", "learning_rate", "weight_decay", "batch_size", "scheduler", "max_epochs", "patience", "preprocessing")}
    for seed in ROBUSTNESS_SEEDS:
        print(f"\n===== Robustness seed {seed} =====")
        result = train_one(config, arrays, seed=seed, device=device, verbose=True)
        metrics = result["best_validation"]
        rows.append({
            "seed": seed,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision_macro": metrics["precision_macro"],
            "val_recall_macro": metrics["recall_macro"],
            "val_f1_macro": metrics["f1_macro"],
            "best_epoch": result["best_epoch"],
            "training_time_seconds": result["training_time"],
        })
    write_csv(output_dir / "robustness_results.csv", rows)
    summary = {}
    for metric in ("val_loss", "val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    write_json(output_dir / "robustness_summary.json", summary)
    return rows, summary


def write_audit_report(path, split_dir, arrays, selected, baseline, final_metrics, robustness_summary, trial_rows, latency):
    successful = [row for row in trial_rows if row.get("status") == "success"]
    raw_rows = [row for row in successful if row["preprocessing"] == "raw"]
    standardized_rows = [row for row in successful if row["preprocessing"] == "standardized"]
    raw_best = choose_best(raw_rows) if raw_rows else None
    standardized_best = choose_best(standardized_rows) if standardized_rows else None
    report = f"""# MLP Experiment Audit Report

## Dataset and preprocessing

The experiment reused the existing split arrays at `{split_dir}` without regenerating or modifying them. The input consists of raw 1,024-sample vibration windows and four integer labels: Normal, Ball, Inner Race, and Outer Race. No per-window normalization, augmentation, FFT features, or handcrafted features were used.

Two preprocessing modes were evaluated independently: raw input, and global standardization using only the training-set mean and standard deviation. Validation and test data used the training statistics.

Shapes were train `{arrays['X_train'].shape}`, validation `{arrays['X_val'].shape}`, and test `{arrays['X_test'].shape}`. The current split is window-level rather than recording-level; source recording identity was discarded by the existing preprocessing pipeline. Therefore, this report does not claim recording-independent evaluation, and windows from the same `.mat` recording may occur in multiple splits.

## Model and training

The model is fully connected only. Candidate architectures were 1024-128-64-4, 1024-256-128-4, and 1024-512-256-4, with ReLU activations and configurable dropout. Adam and CrossEntropyLoss were used. Search runs used at most 60 epochs with early stopping patience 8.

Selection used validation macro F1 as the primary criterion and validation loss as the secondary criterion. Test data was not loaded during tuning or robustness selection. The final test evaluation was performed once after selection using seed 42.

## Search and selection

Twenty hyperparameter configurations were sampled, and each was evaluated with both raw and standardized inputs. The selected trial was `{selected['trial_id']}` using `{selected['preprocessing']}` preprocessing. Its validation macro F1 was `{float(selected['best_val_f1_macro']):.6f}` and validation loss was `{float(selected['best_val_loss']):.6f}`.

The best raw validation F1 was `{float(raw_best['best_val_f1_macro']):.6f}` and the best standardized validation F1 was `{float(standardized_best['best_val_f1_macro']):.6f}`.

## Robustness

The selected configuration was retrained with seeds 42, 43, and 44 using training and validation data. Validation macro F1 mean and standard deviation were `{robustness_summary['val_f1_macro_mean']:.6f} +/- {robustness_summary['val_f1_macro_std']:.6f}`. Validation accuracy was `{robustness_summary['val_accuracy_mean']:.6f} +/- {robustness_summary['val_accuracy_std']:.6f}`. This robustness analysis did not change the selected configuration.

## Final evaluation

The final model achieved test accuracy `{final_metrics['accuracy']:.6f}`, macro precision `{final_metrics['precision_macro']:.6f}`, macro recall `{final_metrics['recall_macro']:.6f}`, and macro F1 `{final_metrics['f1_macro']:.6f}`. The original baseline metrics were accuracy `{baseline['accuracy']:.6f}`, macro precision `{baseline['precision_macro']:.6f}`, macro recall `{baseline['recall_macro']:.6f}`, and macro F1 `{baseline['f1_macro']:.6f}`.

The selected model had {final_metrics['parameter_count']:,} trainable parameters. Training took {final_metrics['training_time_seconds']:.2f} seconds. Warmed-up repeated inference latency on validation windows was {latency['mean_ms_per_sample']:.6f} ms/sample mean, with median {latency['median_ms_per_sample']:.6f} ms/sample and p95 {latency['p95_ms_per_sample']:.6f} ms/sample.

## Interpretation and limitations

The confusion matrix and per-class metrics should be interpreted with particular attention to Ball, Inner Race, and Outer Race confusions. A tuned validation improvement does not establish recording-independent generalization because the inherited split is window-level. Future work should retain recording IDs and use group-disjoint splits, then repeat the comparison across multiple independent recording partitions.
"""
    path.write_text(report)


def main():
    arrays, split_dir = load_splits()
    device = select_device()
    print(f"Using device: {device}")
    print(f"Using splits: {split_dir}")
    print(f"Train: {arrays['X_train'].shape}; Val: {arrays['X_val'].shape}; Test: {arrays['X_test'].shape}")
    tuning_dir = RESULTS_DIR / "tuning"
    final_dir = RESULTS_DIR / "final"
    baseline_dir = RESULTS_DIR / "baseline"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline = parse_baseline_metrics()
    write_json(baseline_dir / "baseline_metrics.json", baseline)
    configs = build_search_configs()
    rows = []
    trial_id = 1
    for config in configs:
        for preprocessing in ("raw", "standardized"):
            rows.append(run_trial(trial_id, config, preprocessing, arrays, device, tuning_dir))
            trial_id += 1
    write_csv(tuning_dir / "tuning_results.csv", rows)
    write_json(tuning_dir / "tuning_results.json", rows)
    plot_hyperparameters(rows, tuning_dir)
    selected = choose_best(rows)
    write_json(tuning_dir / "best_config.json", selected)
    print(f"\nSelected trial: {selected['trial_id']} with validation macro F1 {selected['best_val_f1_macro']:.6f}")

    robustness_rows, robustness_summary = run_robustness(selected, arrays, device, final_dir)

    final_config = {key: selected[key] for key in ("hidden_dims", "dropout", "learning_rate", "weight_decay", "batch_size", "scheduler", "max_epochs", "patience", "preprocessing")}
    final_checkpoint = final_dir / "best_model.pth"
    print("\n===== Final training with seed 42 =====")
    final_result = train_one(final_config, arrays, seed=42, device=device, checkpoint_path=final_checkpoint, verbose=True)
    write_history(final_dir / "training_history.csv", final_result["history"])
    write_json(final_dir / "best_config.json", final_config)

    test_metrics = __import__("scripts.mlp_experiment", fromlist=["evaluate_model"]).evaluate_model(
        final_result["model"],
        final_result["test_loader"],
        __import__("torch").nn.CrossEntropyLoss(),
        device,
        collect=True,
    )
    latency = benchmark_inference(final_result["model"], arrays["X_val"], device)
    report = __import__("sklearn.metrics", fromlist=["classification_report"]).classification_report(
        test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    roc_pr = plot_roc_pr(test_metrics["y_true"], test_metrics["probabilities"], final_dir)
    cm, normalized_cm = plot_confusion_matrices(test_metrics["y_true"], test_metrics["y_pred"], final_dir)
    plot_per_class_metrics(test_metrics["y_true"], test_metrics["y_pred"], final_dir)
    plot_training_curves(final_result["history"], final_dir)
    plot_class_distribution(arrays, final_dir)
    plot_waveform_examples(arrays["X_test"], test_metrics["y_true"], test_metrics["y_pred"], final_dir)
    plot_baseline_comparison(baseline, test_metrics, final_dir)

    final_metrics = {
        "device": device,
        "train_shape": list(arrays["X_train"].shape),
        "val_shape": list(arrays["X_val"].shape),
        "test_shape": list(arrays["X_test"].shape),
        "parameter_count": final_result["parameter_count"],
        "best_epoch": final_result["best_epoch"],
        "best_validation_loss": final_result["best_validation"]["loss"],
        "best_validation_accuracy": final_result["best_validation"]["accuracy"],
        "best_validation_precision_macro": final_result["best_validation"]["precision_macro"],
        "best_validation_recall_macro": final_result["best_validation"]["recall_macro"],
        "best_validation_f1_macro": final_result["best_validation"]["f1_macro"],
        "training_time_seconds": final_result["training_time"],
        "preprocessing_stats": final_result["preprocessing_stats"],
        "accuracy": test_metrics["accuracy"],
        "precision_macro": test_metrics["precision_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "f1_macro": test_metrics["f1_macro"],
        "precision_weighted": test_metrics["precision_weighted"],
        "recall_weighted": test_metrics["recall_weighted"],
        "f1_weighted": test_metrics["f1_weighted"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": normalized_cm.tolist(),
        "roc_pr": roc_pr,
        "inference_latency": latency,
        "selected_config": final_config,
    }
    write_json(final_dir / "final_metrics.json", final_metrics)
    save_predictions(final_dir / "predictions.csv", test_metrics["y_true"], test_metrics["y_pred"], test_metrics["probabilities"])
    comparison_rows = [{"model": "original_baseline", **baseline, "status": "existing"}]
    comparison_rows.extend({"model": f"trial_{row['trial_id']}", **row} for row in rows)
    comparison_rows.append(
        {
            "model": "selected_final",
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
            "status": "final_test",
        }
    )
    write_csv(RESULTS_DIR / "experiment_comparison.csv", comparison_rows)
    write_json(RESULTS_DIR / "best_config.json", selected)
    write_json(final_dir / "robustness_results.json", robustness_rows)
    report_path = RESULTS_DIR / "audit_report.md"
    write_audit_report(report_path, split_dir, arrays, selected, baseline, final_metrics, robustness_summary, rows, latency)

    lines = [
        "MLP Final Metrics",
        "=================",
        f"Device: {device}",
        f"Best epoch: {final_result['best_epoch']}",
        f"Test accuracy: {test_metrics['accuracy']:.6f}",
        f"Macro precision: {test_metrics['precision_macro']:.6f}",
        f"Macro recall: {test_metrics['recall_macro']:.6f}",
        f"Macro F1: {test_metrics['f1_macro']:.6f}",
        f"Weighted precision: {test_metrics['precision_weighted']:.6f}",
        f"Weighted recall: {test_metrics['recall_weighted']:.6f}",
        f"Weighted F1: {test_metrics['f1_weighted']:.6f}",
        f"Training time seconds: {final_result['training_time']:.2f}",
        f"Mean inference ms/sample: {latency['mean_ms_per_sample']:.6f}",
        f"Checkpoint: {final_checkpoint}",
    ]
    (final_dir / "final_metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n===== Final Results =====")
    print("\n".join(lines))
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
