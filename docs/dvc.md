# DVC Pipeline Orchestration

We use DVC (Data Version Control) to coordinate data flow and guarantee reproducibility of the full ML pipeline.

## Pipeline Directed Acyclic Graph (DAG)

The DAG comprises the following sequence:

```text
validate_raw -> merge -> optimize_memory -> eda -> feature_engineering -> feature_selection -> train -> evaluate -> explain -> submit
```

## Running the Pipeline

Execute the pipeline end-to-end via command line:
```bash
dvc repro
```
This is mapped to the Makefile short-cut:
```bash
make run-pipeline
```

## Stage Definitions

Each stage is defined in `dvc.yaml`:
- **validate_raw**: Ensures files match standard structural schemas before execution.
- **merge**: Pairs transactional data with identity reports.
- **optimize_memory**: Converts standard data types (e.g. float64 to float32) to prevent out-of-memory errors on large tables.
- **eda**: Saves reports on missing rates and columns correlation details.
- **feature_engineering**: Outputs aggregate features, frequencies and target averages.
- **feature_selection**: Cuts off features with low relevance or large correlations.
- **train**: Trains CV folds and writes models.
- **evaluate**: Measures validation performance and optimizes the binary decision threshold.
- **explain**: Runs SHAP values on trained models.
- **submit**: Assembles final submission parquet / CSV.
