# Feature Engineering Dictionary

Definitions and mathematical formulations of engineered feature families in the Feature Store.

---

## 📈 Feature Families

### 1. Frequency Encodings (`*_fq`)
Counts occurrences of high-cardinality values across the training set:
$$\text{Feature\_FQ} = \text{Count}(\text{card1})$$

### 2. Grouped Aggregations (`*_mean`, `*_std`)
Aggregates numeric attributes grouped by user payment card identifiers:
$$\text{TransactionAmt\_mean\_card1} = \mathbb{E}[\text{TransactionAmt} \mid \text{card1}]$$

### 3. Ratio Features (`*_ratio`)
Ratio of current transaction value to user historical mean:
$$\text{Amt\_to\_Mean\_card1} = \frac{\text{TransactionAmt}}{\mathbb{E}[\text{TransactionAmt} \mid \text{card1}] + \epsilon}$$

### 4. Difference Features (`*_diff`)
Difference between current transaction value and user historical mean:
$$\text{Amt\_Diff\_card1} = \text{TransactionAmt} - \mathbb{E}[\text{TransactionAmt} \mid \text{card1}]$$

### 5. Interaction Features (`*_x_*`)
Combinations of card and domain attributes (e.g. `card1_x_P_emaildomain`).
