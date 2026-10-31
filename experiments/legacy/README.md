# Legacy experiments

This directory preserves the earlier outlier-pruned and transformer-augmented Deep Forest studies and their original write-ups under [reports/](reports/). Some claims in those reports have not been reconciled with every saved JSON result, so treat them as an archive rather than the current project summary.

## Contents

| Study | Runner | Results | Report |
| --- | --- | --- | --- |
| Outlier pruning | [run_experiment.py](run_experiment.py) | [experiment_results.json](experiment_results.json) | [report](reports/experiment_report.md) |
| Inter-layer transformer | [run_transformer_experiment.py](run_transformer_experiment.py) | [transformer_experiment_results.json](transformer_experiment_results.json) | [report](reports/transformer_experiment_report.md) |
| MAGIC comparison | [run_magictelescope_experiment.py](run_magictelescope_experiment.py) | [magictelescope_experiment_results.json](magictelescope_experiment_results.json) | [report](reports/magictelescope_experiment_report.md) |

## Run an archived study

The scripts import the published `deep-forest-py310` package directly and no longer require a local Deep Forest checkout. Create the project virtual environment using the [root setup guide](../../README.md#environment), then install the [legacy requirements](requirements.txt) for PyTorch and pandas. For example, from the repository root run:

```bash
.venv/bin/python experiments/legacy/run_experiment.py --n_trees 50
.venv/bin/python experiments/legacy/run_transformer_experiment.py --n_trees 50 --d_token 16 --n_heads 2 --transformer_epochs 15
.venv/bin/python experiments/legacy/run_magictelescope_experiment.py --n_trees 35 --n_estimators 4
```

The MAGIC runner fetches its dataset through OpenML, so that run also needs network access.
