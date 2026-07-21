# Model Evaluation

We evaluate model performance using comprehensive classification metrics.

## Performance Metrics

Validation evaluations write logs and save standard metrics under `reports/evaluation/ metrics.json`:
- **Primary metric**: ROC-AUC (Area under the Receiver Operating Characteristic).
- **Secondary metrics**: PR-AUC (Precision-Recall curve area), F1 Score, Matthews Correlation Coefficient (MCC), Log Loss, and Brier Score.

## Threshold Optimization

- In fraud detection, the default 0.5 probability threshold is rarely optimal due to skewed class balances.
- The pipeline scans probabilities to optimize the threshold targeting a custom utility function (e.g. maximizing MCC or F1).
- The selected cutoff is written to `artifacts/thresholds/best_threshold.json` to guide deployment inference.
