# Feature Engineering

We construct high-fidelity features for fraud detection. Engineered inputs are saved as reusable parquet partitions in `data/feature_store/`.

## Feature Classes

1. **Card and Client Aggregations**:
   - Computes statistical moments (mean, standard deviation) of transaction amounts grouped by Card IDs and customer proxies.
   - Captured in `aggregation_features.parquet`.

2. **Frequency Encodings**:
   - Counts occurrences of values (e.g. Card IP, system addresses, email addresses, browser details).
   - Captured in `frequency_features.parquet`.

3. **Time-Based Groups**:
   - Extracts diurnal statistics (hour of the day, day of the week) from transaction timestamps.
   - Captured in `time_features.parquet` (if created) or inside client aggregations.

4. **Identity and Device Details**:
   - Normalizes browser strings, OS targets, and device models.
   - Captured in `identity_features.parquet`.
