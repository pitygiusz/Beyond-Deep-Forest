# Beyond Deep Forest

This repository studies whether changes to the Deep Forest cascade improve
classification performance, runtime, or stability compared with a standard
cascade and tree-ensemble baselines. It includes the model implementations,
experiment runners, saved benchmark results, and analysis notebooks.

## Models

The [`deep_forests`](deep_forests/) package provides four scikit-learn-style
Deep Forest classifiers:

- **Bare** — standard cascade baseline.
- **Projection** — adds fixed random nonlinear projections to the forest
  inputs.
- **IFC** — adds interaction features extracted from forest decision rules.
- **Combined** — uses both projections and interaction features.

The benchmark also compares **Random Forest** and **Extra Trees**.
See the [method notes and architecture diagrams](deep_forests/README.md) for
implementation details and a classifier usage example.

## Repository map

- [`deep_forests/`](deep_forests/) — classifier implementations, diagrams, and
  focused tests.
- [`experiments/run.py`](experiments/run.py) — tabular benchmark, including
  cross-validated random search and held-out test evaluation.
- [`experiments/run_non_tabular.py`](experiments/run_non_tabular.py) — fixed
  Deep Forest comparisons on IMDB, MNIST, and Olivetti/ORL.
- [`experiments/models.py`](experiments/models.py) — model definitions,
  defaults, and search spaces.
- [`experiments/prepare_data.py`](experiments/prepare_data.py) — dataset
  loading, cleaning, splitting, and preprocessing.
- [`raw_data/`](raw_data/) — checked-in source data; downloaded datasets and
  preprocessing caches are stored here at runtime.
- [`results/`](results/) — per-run outputs and consolidated tabular and
  non-tabular result tables.
- [`analysis/`](analysis/) — notebooks for aggregating results, comparing
  models, and statistical analysis.

## Setup

The project targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) to
manage the locked environment:

```bash
uv sync --python 3.12
```

Some datasets are downloaded from OpenML, the UCI repository, Hugging Face, or
scikit-learn on first use. They are cached under `raw_data/` and are not
committed.

## Run the tabular benchmark

Run all registered tabular datasets with the default settings:

```bash
uv run -m experiments.run
```

Each dataset gets a stratified train/test split. For each model, the runner
compares its default configuration with randomly sampled configurations using
stratified cross-validation on the training split, ranking candidates by
balanced accuracy (three folds by default). Preprocessing is fitted within
each CV fold. The best configuration and the defaults are then evaluated on
the held-out test split; the test data is not used for model selection.

Start with a smaller run while checking the setup:

```bash
uv run -m experiments.run \
  --datasets yeast madelon \
  --models bare projection random_forest \
  --trials 5
```

Useful options include `--datasets`, `--models`, `--trials`, `--cv-folds`,
`--model-seeds`, `--seed`, `--test-size`, `--n-jobs`, and `--output`. The
default search uses 10 candidates, including the default configuration, and
the default test evaluation uses model seeds 0 through 4.

Each run writes `settings.json`, `search.csv`, and `test.csv` to a timestamped
folder under `results/`. Pass `--output results/<run-folder>` to resume an
interrupted run with the same settings; completed dataset/model pairs are
skipped.

## Run the non-tabular comparisons

The dedicated runner evaluates each of the four Deep Forest variants on IMDB,
MNIST, and ORL, using one fit per dataset and variant:

```bash
uv run -m experiments.run_non_tabular
```

For these runs, the forest configuration uses four estimators per layer and
500 trees per estimator. Use `--datasets`, `--models`, `--seed`, `--n-jobs`,
or `--output` to select the run configuration. Results are saved in a
timestamped `non_tabular_*` folder under `results/`. The shared data registry
also includes CIFAR-10, which can be selected with `experiments.run`.

## Results and analysis

- [`results/all_tabular_results.csv`](results/all_tabular_results.csv) and
  [`results/all_nontabular_results.csv`](results/all_nontabular_results.csv)
  contain consolidated benchmark results.
- [`analysis/consolidated_analysis.ipynb`](analysis/consolidated_analysis.ipynb)
  compares test scores, ranks and statistical tests, seed variability, tuning
  effects, runtimes, and selected hyperparameters.
- [`analysis/aggregate.ipynb`](analysis/aggregate.ipynb) aggregates benchmark
  metrics and examines model gains, tuning, and runtime.
- [`analysis/friedman.ipynb`](analysis/friedman.ipynb) focuses on the Friedman
  test across datasets.

The consolidated analysis notebook documents its tabular comparison as
covering 20 datasets. Individual run folders retain their settings and raw
search and test rows for reproducibility.

## Tests

Run the repository tests with:

```bash
uv run pytest
```
