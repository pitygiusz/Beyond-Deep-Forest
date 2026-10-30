# Legacy experiments

This directory preserves earlier exploratory work on outlier-pruned Deep
Forest and transformer-augmented Deep Forest. The original overview is retained
in [README_OLD.md](README_OLD.md), and the original write-ups are under
[reports/](reports/).

The old overview contains claims that have not been reconciled with every
saved JSON result. Treat it as an archive, not the current project summary.

## Contents

| Study | Runner | Results | Report |
| --- | --- | --- | --- |
| Outlier pruning | [run_experiment.py](run_experiment.py) | [experiment_results.json](experiment_results.json) | [report](reports/experiment_report.md) |
| Inter-layer transformer | [run_transformer_experiment.py](run_transformer_experiment.py) | [transformer_experiment_results.json](transformer_experiment_results.json) | [report](reports/transformer_experiment_report.md) |
| MAGIC comparison | [run_magictelescope_experiment.py](run_magictelescope_experiment.py) | [magictelescope_experiment_results.json](magictelescope_experiment_results.json) | [report](reports/magictelescope_experiment_report.md) |

The local `deep_forest_py310/` checkout used by these scripts is preserved in
this directory but ignored by the parent Git repository. The old repository
stored it as a Git link without a `.gitmodules` mapping. The active experiment
uses a separate pinned clone and build.

With the archived local `deep_forest_py310/` checkout available, run the old
scripts from this directory if needed: `evaluate_model.py` uses a
working-directory-relative import path. Recreating this archive from a fresh
checkout also requires the old fork and its local NumPy compatibility change;
the active experiment has a separate reproducible setup.
