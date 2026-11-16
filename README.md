# Beyond Deep Forest

This repository compares four cascade-forest architectures and includes other exploratory studies.

## Layout

- [Active IFC and projection experiment](experiments/ifc_projection/README.md) compares standard Deep Forest, fixed random nonlinear projections, an adaptation of hiDF-style interaction feature concatenation (IFC), and their combination.
- [Other experiments](experiments/other/README.md) include outlier-pruning and transformer studies and their results.

## Environment

Install [uv](https://docs.astral.sh/uv/) and use Python 3.12. The dependencies are specified in [pyproject.toml](pyproject.toml) and locked in [uv.lock](uv.lock). 

```bash
uv sync --python 3.12
```

The four-way comparison uses the default dependencies. To include PyTorch for the archived transformer experiments, run:

```bash
uv sync --extra other
```

## Run

From the repository root, run a small built-in dataset comparison with:

```bash
uv run -m experiments.ifc_projection.run_experiment
```

Tune a small tabular pilot before running the full workflow. The
`experiments.run_tabular` entry point excludes the text/image datasets
(`imdb`, `mnist`, `cifar10`, and `orl`):

```bash
uv run -m experiments.run_tabular --phase tune --datasets yeast madelon --modes bare projection random_forest --tuning-trials 5
```

The benchmark has separate tuning and final-evaluation phases. By default,
the tuning phase covers all tabular datasets and model modes, uses 20
trials per dataset/model, and writes a timestamped tuning JSON artifact:

```bash
uv run -m experiments.run_tabular --phase tune --tuning-method bayesian
```

Use `experiments.run` instead to include every dataset in the registry:

```bash
uv run -m experiments.run --phase tune --tuning-method bayesian
```

Tuning is per dataset and model. The runner makes a stratified 80/20
train/test split, then scores every candidate using 3-fold stratified
cross-validation on the training portion. Preprocessing is fitted separately
inside each fold, and each cascade selects its depth using only that fold's
training data. The held-out test portion is not used to select parameters.
Use `--tuning-method random` for random search, or adjust the search budget
with `--tuning-trials` and CV folds with `--cv-folds`. Random Forest,
ExtraTrees, HistGradientBoosting, and the Deep Forest variants (including
`utility_ifc`) have model-specific search spaces. Current settings are tried
as an initial configuration.

The tuning command prints the JSON artifact path. Pass that path to the final
phase; it fits both default and tuned configurations and evaluates them on the
same test split:

```bash
uv run -m experiments.run_tabular --phase final --tuning-file results/tabular_tuning_bayesian_<timestamp>.json
```

For an all-dataset tuning artifact, run the final phase through
`experiments.run` as well.

The final phase writes a timestamped CSV and `.metadata.json` sidecar.
It evaluates seed `0` once by default; pass `--seeds 0 1 2 3 4` for repeated
stratified holdouts. The tuning seed, test fraction, and fixed model settings
must match the tuning artifact. Results include split indices and hashes,
selected parameters, CV scores, and final test metrics. Use `--tuning-file`
with the same path to resume a tuning run; incompatible settings, code, or
data are rejected.

The older CSV and JSON results remain archived. Their values should not be
combined with the new protocol without matching provenance.
