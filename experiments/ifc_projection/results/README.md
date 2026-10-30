# Recorded results

These JSON files contain per-fold metrics, configuration, and aggregate metrics
from breast-cancer experiments with one split seed (42). They are archived
outputs; new runs write an ignored `latest.json` by default.

- [breast_cancer_seed42_projection_only.json](breast_cancer_seed42_projection_only.json):
  the earlier standard versus projection comparison.
- [breast_cancer_seed42_four_way.json](breast_cancer_seed42_four_way.json):
  the corrected four-way comparison, using the IFC adapter with selected
  interactions in the retained multi-layer models.

The four-way run used five stratified folds, two RF/ExtraTrees pairs per layer,
50 trees per forest, up to five layers, 16 projected features per forest, and
up to six IFC features per layer. Mean accuracy was 0.9684 (standard), 0.9649
(projection), 0.9684 (IFC adaptation), and 0.9666 (combined). These runs are
exploratory. They cover one dataset and one split seed; the supervised IFC
feature construction may make the internal depth criterion optimistic.
