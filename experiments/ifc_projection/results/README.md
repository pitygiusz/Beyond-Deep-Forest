# Recorded results

These JSON files contain configurations, per-fold metrics, and aggregate metrics from breast-cancer experiments with one split seed (42). They are archived outputs; new runs write a timestamped comparison file by default.

- [breast_cancer_seed42_projection_only.json](breast_cancer_seed42_projection_only.json) records the earlier standard-versus-projection comparison.
- [breast_cancer_seed42_four_way.json](breast_cancer_seed42_four_way.json) records the corrected four-way comparison, including selected interaction features in retained multi-layer models.
- [utility_feedback_breast_cancer_seed42.json](utility_feedback_breast_cancer_seed42.json) records a two-fold pilot of the utility-selected IFC variant. It includes split indices, source/dependency fingerprints, selected groups, and per-fold metrics.

The four-way run used five stratified folds, two RF/ExtraTrees pairs per layer, 50 trees per forest, up to five layers, 16 projected features per forest, and up to six IFC features per layer. Mean accuracy was 0.9684 for standard Deep Forest, 0.9649 for projections, 0.9684 for the IFC adaptation, and 0.9666 for the combination. These runs predate out-of-fold IFC features, raw-value projections, and external depth selection, so fresh results are not directly comparable.

These archived outputs predate the current dependency set in [pyproject.toml](../../../pyproject.toml). Fresh runs may produce different values.

The utility pilot is a new, small iteration using the current code. Utility selection improved mean balanced accuracy from 0.9483 to 0.9492 and log-loss from 0.1866 to 0.1785 over standard Deep Forest on two folds; accuracy was unchanged. It selected only forest probability groups and no IFC rules. Mean fit time rose from 0.19s to 3.11s per fold. Raising the minimum utility gain from 0.001 to 0.005 retained the same scores and removed one group from a layer. These results are exploratory and do not establish a dataset-level benefit.
