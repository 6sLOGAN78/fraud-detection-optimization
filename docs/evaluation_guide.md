# Model Evaluation and Diagnostic Diagnostics Guide

Comprehensive guide on evaluation metrics, probability calibration, and financial net savings cost matrices.

---

## 📊 Performance Metrics

* **ROC-AUC**: Evaluates overall ranking capability across probability thresholds.
* **PR-AUC**: Measures precision-recall trade-off under heavy class imbalance.
* **F1-Score & MCC**: Matthew's Correlation Coefficient evaluates true binary classification capability.
* **KS Statistic**: Kolmogorov-Smirnov test measures maximum separation between fraud and legitimate CDFs.

---

## 💸 Financial Cost-Loss Matrix

$$\text{Net Savings} = \text{Fraud Captured} \times V_{\text{avg}} - \text{False Positives} \times C_{\text{friction}}$$

* **Average Fraud Value ($V_{\text{avg}}$)**: $150.00
* **Customer Friction Cost ($C_{\text{friction}}$)**: $15.00
