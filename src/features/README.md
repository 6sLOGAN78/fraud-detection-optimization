# Enterprise Feature Store & Feature Selection Engine (`src/features/`)

The `src/features/` package builds the production feature store and executes multi-strategy feature selection.

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
