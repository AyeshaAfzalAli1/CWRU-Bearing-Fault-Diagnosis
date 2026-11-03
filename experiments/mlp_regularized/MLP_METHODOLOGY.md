# Regularized MLP Methodology and Results

## 1. Experiment covered

This document describes the latest regularized MLP experiment in
`experiments/mlp_regularized/`. The clean-curve rerun used the same training
recipe and seed as the original experiment. It added clean evaluation metrics
for plotting, but did not change the model, optimizer, hyperparameters, data
splits, normalization, or test evaluation.

The main implementation is in:

- `experiments/mlp_regularized/mlp_reg.py`
- `experiments/mlp_regularized/train_mlp_reg.py`
- `experiments/mlp_regularized/run_clean_curves.py`

The saved model checkpoint is `best_mlp_reg_model.pth`.

## 2. Dataset preparation

The dataset is based on the Case Western Reserve University (CWRU) bearing
vibration data. The preprocessing script
`scripts/prepare_cwru.py`:

1. Reads MATLAB `.mat` files from the normal and fault-data directories.
2. Selects the drive-end vibration signal whose variable name ends in
   `DE_time`.
3. Extracts non-overlapping windows of 1,024 samples using a step size of
   1,024.
4. Assigns one of four integer labels:

   | Label | Class |
   |---:|---|
   | 0 | Normal |
   | 1 | Ball |
   | 2 | Inner Race |
   | 3 | Outer Race |

5. Saves the combined arrays as `data/processed/X.npy` and
   `data/processed/y.npy`.

The MLP therefore receives one-dimensional raw vibration windows with shape
`(number_of_samples, 1024)`.

## 3. Dataset splitting

`scripts/split_dataset_fixed.py` creates the saved train, validation, and test
arrays using stratified random splitting with seed 42:

1. 10% is held out as the test set.
2. The remaining 90% is split into approximately 70% training and 20%
   validation data.

The regularized MLP uses these existing files:

- `data/data/splits/X_train.npy`
- `data/data/splits/y_train.npy`
- `data/data/splits/X_val.npy`
- `data/data/splits/y_val.npy`
- `data/data/splits/X_test.npy`
- `data/data/splits/y_test.npy`

The observed split sizes are:

| Split | Samples | Shape |
|---|---:|---|
| Train | 4,147 | `(4147, 1024)` |
| Validation | 1,186 | `(1186, 1024)` |
| Test | 593 | `(593, 1024)` |

The test class distribution is:

| Class | Test samples |
|---|---:|
| Normal | 166 |
| Ball | 142 |
| Inner Race | 142 |
| Outer Race | 143 |

## 4. Per-window normalization

Each 1,024-sample window is normalized independently:

```text
normalized_window = (window - window_mean) / (window_std + 1e-8)
```

The mean and standard deviation are calculated separately for each window.
The same operation is applied to training, validation, and test windows.
There is no global scaler fitted using validation or test data.

## 5. MLP architecture

The model is a fully connected classifier:

```text
Input:       1024 values
Linear:      1024 -> 128
BatchNorm:   128
ReLU
Dropout:     0.5
Linear:      128 -> 64
BatchNorm:   64
ReLU
Dropout:     0.5
Linear:      64 -> 4
```

The final layer produces four logits, one for each bearing class. The predicted
class is the index of the largest logit:

```python
prediction = outputs.argmax(dim=1)
```

The model has **140,100 trainable parameters**.

## 6. Training procedure

The training configuration is:

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Batch size | 128 |
| Maximum epochs | 40 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Loss | Cross-entropy with label smoothing |
| Label smoothing | 0.1 |
| Dropout | 0.5 |
| Scheduler | Cosine annealing |
| Gradient clipping | Maximum norm 1.0 |
| Early-stopping patience | 8 epochs |

### Training augmentation

For every training batch, the normalized windows are augmented before the
forward pass:

1. A random circular shift between -100 and +100 samples is applied.
2. Gaussian noise with standard deviation 0.05 is added.

The labels are unchanged by this augmentation.

### Regularization

Dropout is active during training. AdamW also applies weight decay during
parameter updates. However, the displayed loss is only the label-smoothed
cross-entropy loss; an explicit L1 or L2 penalty is not added to the reported
loss value.

## 7. Validation and checkpoint selection

After each training epoch, the model is evaluated on the clean validation
windows with:

- `model.eval()`
- dropout disabled
- no augmentation
- `torch.no_grad()`
- the same label-smoothed cross-entropy criterion

The checkpoint with the lowest validation loss is saved. The best
validation-loss checkpoint was from **epoch 36**. Training continued through
epoch 40 because the patience limit was not reached.

## 8. Clean training curves

The original training curves measured training loss and accuracy on augmented
batches while dropout was active. Those values are not directly comparable to
the clean validation values.

The latest clean-curve run added an evaluation pass at the end of each epoch.
This extra pass:

- uses the complete clean training set;
- uses `model.eval()`;
- disables dropout;
- uses no augmentation;
- uses `torch.no_grad()`;
- does not perform an optimizer or scheduler update.

The clean training metrics and validation metrics are saved in
`results_clean_curves/history_clean.csv`. The plots are:

- `results_clean_curves/loss_curve_clean.png`
- `results_clean_curves/accuracy_curve_clean.png`

At epoch 40:

| Metric | Clean train | Validation |
|---|---:|---:|
| Loss | 0.486725 | 0.491189 |
| Accuracy | 94.3815% | 94.6880% |

These values are much closer because both sides are measured in evaluation
mode on clean, normalized data. The small remaining difference is normal
validation variation and does not imply that the curves must be identical.

## 9. Test evaluation

After training, the best-validation-loss checkpoint is evaluated once on the
held-out test set of 593 samples. Dropout is disabled and no augmentation is
used.

The resulting test metrics are:

| Metric | Result |
|---|---:|
| Accuracy | **92.7487%** |
| Macro precision | **92.9016%** |
| Macro recall | **92.4431%** |
| Macro F1 | **92.5105%** |
| Trainable parameters | **140,100** |

Rounded for a results table:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| Regularized MLP | 92.75% | 92.90% | 92.44% | 92.51% | 140,100 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 1.0000 | 1.0000 | 1.0000 | 166 |
| Ball | 1.0000 | 0.9577 | 0.9784 | 142 |
| Inner Race | 0.9062 | 0.8169 | 0.8593 | 142 |
| Outer Race | 0.8098 | 0.9231 | 0.8627 | 143 |

The test confusion matrix uses rows as true classes and columns as predicted
classes, in the order Normal, Ball, Inner Race, Outer Race:

```text
[[166,   0,   0,   0],
 [  0, 136,   1,   5],
 [  0,   0, 116,  26],
 [  0,   0,  11, 132]]
```

The saved confusion matrix is generated from test predictions, not validation
predictions.

## 10. Why the original curves showed validation better than training

The original epoch training metrics were measured under harder conditions:

- dropout was active;
- every training batch was shifted and noised;
- the measurements were made during the training-mode forward passes.

Validation metrics were measured under easier, clean conditions:

- dropout was disabled;
- no augmentation was applied;
- the model was in evaluation mode.

Consequently, validation accuracy could be higher and validation loss lower
than the reported training values. This was primarily a metric-comparison
issue, not evidence by itself of model failure or unusual overfitting.

## 11. Limitations and issues

### Window-level rather than recording-level splitting

The split is performed after windows are generated, using random stratified
sample splitting. Recording identity is not retained by the preprocessing
pipeline. Therefore, windows from the same source recording can potentially
occur in training, validation, and test sets.

This can make the reported performance more optimistic than a
recording-independent evaluation. A stronger evaluation would retain source
recording IDs and use group-disjoint train, validation, and test splits.

### Duplicate windows across splits

An audit found exact duplicate windows across the saved arrays:

```text
Train-validation: 130
Train-test:         65
Validation-test:    17
```

This is a concrete form of leakage in the current saved split and further
reduces confidence that the test score represents performance on entirely
unseen recordings or windows.

### Training and validation history are different measurements

The original `results/history.csv` contains augmented training-mode metrics,
whereas `results_clean_curves/history_clean.csv` contains clean evaluation-mode
training metrics. They should not be compared as if they were measured under
the same conditions. The clean curves are the appropriate visualization for
comparing training and validation behavior.

### Single fixed seed and split

The headline test result is from seed 42 and one fixed split. It does not
provide a confidence interval over independent recording partitions. Repeated
seeds can show optimization variability, but they do not solve the
recording-overlap problem.

### No independent external test set

The test data is a held-out portion of the same processed dataset, not an
independent bearing experiment or unseen recording collection. The result
should therefore be presented as performance on this saved split, subject to
the leakage limitations above.

## 12. Source of reported values

The reported values are from:

- `experiments/mlp_regularized/results/metrics.txt`
- `experiments/mlp_regularized/results_extra/classification_report.txt`
- `experiments/mlp_regularized/results_clean_curves/history_clean.csv`
- `experiments/mlp_regularized/results/confusion_matrix.png`
- `experiments/mlp_regularized/train_mlp_reg.py`
- `experiments/mlp_regularized/run_clean_curves.py`

