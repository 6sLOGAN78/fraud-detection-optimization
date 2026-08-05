# Enterprise Feature Store & Feature Selection Engine (`src/features/`)

The `src/features/` package builds the production feature store and executes multi-strategy feature selection.

---

## 📊 Feature Visualizations & Selection Analysis

### 1. Global SHAP Feature Importance
The feature selection engine identifies key drivers of fraud probability using SHAP importance. `TransactionAmt` (transaction amount), `card1`, and `card2` emerge as the primary discriminators.

![Global SHAP Feature Importance](../../reports/images/shap_feature_importance.png)

### 2. Multi-Collinearity & Feature Correlation Matrix
Heatmap of Pearson correlation coefficients across engineered feature families. Highly correlated features are pruned to avoid model instability and overfitting.

![Pearson Correlation Heatmap](../../reports/eda/correlation/plots/pearson_heatmap.png)

---

## 💡 Feature Family Formulations

```
                       ┌───────────────────────────────┐
                       │   Raw Transaction Attributes  │
                       └───────────────┬───────────────┘
                                       │
         ┌───────────────────┬─────────┴─────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼
  [Frequency Encoding]  [Aggregations]      [Ratios/Diffs]      [Interactions]
   `card1_fq`            `amt_mean_card1`   `amt_ratio_card1`   `card1_x_email`
```

1. **Frequency Encodings (`*_fq`)**: Computes value occurrence counts across high-cardinality columns.
2. **Grouped Aggregations (`*_mean`, `*_std`)**: Calculates user-level statistics grouped by card IDs.
3. **Ratios & Differences (`*_ratio`, `*_diff`)**: Quantifies deviation of current transaction from user historical average:
   $$\text{Ratio} = \frac{\text{TransactionAmt}}{\text{Mean\_Amt\_card1} + \epsilon}$$
4. **Multi-Strategy Feature Selection**: Evaluates SHAP importance, Mutual Information, Boruta, and Recursive Feature Elimination (RFE) to prune non-informative features from 404 down to the optimal feature subset.
