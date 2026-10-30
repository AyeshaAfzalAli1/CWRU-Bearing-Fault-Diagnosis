# MLP Experiment Audit Report

## Dataset and preprocessing

The experiment reused the existing split arrays at `C:\Users\Aparna Karanth\Desktop\DL_Project\data\data\splits` without regenerating or modifying them. The input consists of raw 1,024-sample vibration windows and four integer labels: Normal, Ball, Inner Race, and Outer Race. No per-window normalization, augmentation, FFT features, or handcrafted features were used.

Two preprocessing modes were evaluated independently: raw input, and global standardization using only the training-set mean and standard deviation. Validation and test data used the training statistics.

Shapes were train `(4147, 1024)`, validation `(1186, 1024)`, and test `(593, 1024)`. The current split is window-level rather than recording-level; source recording identity was discarded by the existing preprocessing pipeline. Therefore, this report does not claim recording-independent evaluation, and windows from the same `.mat` recording may occur in multiple splits.

## Model and training

The model is fully connected only. Candidate architectures were 1024-128-64-4, 1024-256-128-4, and 1024-512-256-4, with ReLU activations and configurable dropout. Adam and CrossEntropyLoss were used. Search runs used at most 60 epochs with early stopping patience 8.

Selection used validation macro F1 as the primary criterion and validation loss as the secondary criterion. Test data was not loaded during tuning or robustness selection. The final test evaluation was performed once after selection using seed 42.

## Search and selection

Twenty hyperparameter configurations were sampled, and each was evaluated with both raw and standardized inputs. The selected trial was `37` using `raw` preprocessing. Its validation macro F1 was `0.944713` and validation loss was `0.157755`.

The best raw validation F1 was `0.944713` and the best standardized validation F1 was `0.895980`.

## Robustness

The selected configuration was retrained with seeds 42, 43, and 44 using training and validation data. Validation macro F1 mean and standard deviation were `0.939744 +/- 0.005847`. Validation accuracy was `0.942102 +/- 0.005614`. This robustness analysis did not change the selected configuration.

## Final evaluation

The final model achieved test accuracy `0.919056`, macro precision `0.917898`, macro recall `0.915715`, and macro F1 `0.915966`. The original baseline metrics were accuracy `0.883642`, macro precision `0.878741`, macro recall `0.878915`, and macro F1 `0.878651`.

The selected model had 295,812 trainable parameters. Training took 115.49 seconds. Warmed-up repeated inference latency on validation windows was 0.049745 ms/sample mean, with median 0.051439 ms/sample and p95 0.075466 ms/sample.

## Interpretation and limitations

The confusion matrix and per-class metrics should be interpreted with particular attention to Ball, Inner Race, and Outer Race confusions. A tuned validation improvement does not establish recording-independent generalization because the inherited split is window-level. Future work should retain recording IDs and use group-disjoint splits, then repeat the comparison across multiple independent recording partitions.
