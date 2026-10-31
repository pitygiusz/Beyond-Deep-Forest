# Recorded results

These JSON files contain configurations, per-fold metrics, and aggregate metrics from breast-cancer experiments with one split seed (42). They are archived outputs; new runs write an ignored `latest.json` by default.

- [breast_cancer_seed42_projection_only.json](breast_cancer_seed42_projection_only.json) records the earlier standard-versus-projection comparison.
- [breast_cancer_seed42_four_way.json](breast_cancer_seed42_four_way.json) records the corrected four-way comparison, including selected interaction features in retained multi-layer models.

The four-way run used five stratified folds, two RF/ExtraTrees pairs per layer, 50 trees per forest, up to five layers, 16 projected features per forest, and up to six IFC features per layer. Mean accuracy was 0.9684 for standard Deep Forest, 0.9649 for projections, 0.9684 for the IFC adaptation, and 0.9666 for the combination. These runs are exploratory because they cover one dataset and one split seed, and supervised IFC features may make the internal cascade-depth criterion optimistic.

The recorded outputs were generated with a locally compiled Deep Forest fork and a NumPy 2.5 compatibility patch before the repository switched to the published package. The current [requirements](../../../requirements.txt) pin NumPy 2.3.5 to retain the upstream percentile API, so a fresh run need not be bit-for-bit identical to these archived files.
