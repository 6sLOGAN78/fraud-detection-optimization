# Model Training

We partition datasets and train models using strict validation guardrails.

## Cross Validation Strategy

To prevent data leakage and simulate real production evaluation:
- **GroupKFold Split**: We partition based on the transaction timestamp proxy (`TransactionDT`) to ensure that models do not make predictions on historical data using features trained on future entries.
- **Out of Fold Predictions**: Predictions for validation folds are collected to produce an OOF target table saved under `artifacts/oof/`.

## Model Architectures

The framework supports three prominent gradient boosting frameworks selectable in configurations:
1. **LightGBM**: Default fast model using histogram-based binning.
2. **CatBoost**: Excels at handling categorical coordinates natively.
3. **XGBoost**: Robust traditional tree booting framework.
