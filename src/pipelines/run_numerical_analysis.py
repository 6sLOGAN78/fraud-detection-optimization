"""Pipeline script to execute Numerical Feature Analysis on merged datasets."""

from pathlib import Path

import pandas as pd

from src.eda.numerical import NumericalFeatureAnalyzer


def main() -> None:
    """Main pipeline execution entry point."""
    # Define workspace path targets
    project_root = Path.cwd()
    train_path = project_root / "data" / "interim" / "train_merged.parquet"
    test_path = project_root / "data" / "interim" / "test_merged.parquet"
    report_dir = project_root / "reports" / "eda" / "numerical"

    print(f"Loading datasets:\n - Train: {train_path}\n - Test: {test_path}")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Initialize and execute Numerical Analysis
    print("Initializing Numerical Feature Diagnostics...")
    analyzer = NumericalFeatureAnalyzer(
        df_train=df_train,
        df_test=df_test,
        target_col="isFraud",
    )
    analyzer.analyze_all(report_dir)
    print(f"Numerical feature analysis completed. Outputs saved to {report_dir}")


if __name__ == "__main__":
    main()
