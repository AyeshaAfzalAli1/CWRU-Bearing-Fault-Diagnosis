# Extra Evaluation Comparison

Timing device: `cpu`

| Model | Test accuracy | Macro F1 | Parameters | ms/sample inference |
|---|---:|---:|---:|---:|
| Original MLP | 0.883642 | 0.878651 | 295,812 | 0.0444 |
| Regularized MLP | 0.933671 +/- 0.005922 (seed 42: 0.927487) | 0.931391 +/- 0.006068 (seed 42: 0.925105) | 140,100 | 0.017973 |
| CNN | 0.996627 | 0.996479 | 1,051,940 | Skipped: checkpoint not found. |

Training times are not directly comparable (different epochs, data loading and early stopping).
