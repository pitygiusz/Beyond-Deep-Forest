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

Run a tabular pilot before the full benchmark:

```bash
uv run -m experiments.run_tabular --datasets yeast madelon --seeds 0 1
```

The tabular runner saves a new timestamped CSV and a matching `.metadata.json`.
The sidecar records code and dependency fingerprints, exact train/test indices,
prepared-data hashes, model parameters, projection-width validation scores,
and validation scores used to select cascade depth. A run can be resumed with
`--output` pointing to its CSV; mismatched code, settings, or data are rejected.
For projected variants, widths `0`, `4`, and `16` are compared on an inner
training split by default. Width zero lets validation reject projections.
The outer test split is used only for final metrics. The runner also includes
random-forest, ExtraTrees, and histogram gradient boosting baselines; the
default includes histogram gradient boosting. Each baseline uses a small
three-fold inner grid search scored by balanced accuracy; the chosen settings
and inner score are saved in the sidecar.

The older CSV and JSON results remain archived. Their values should not be
combined with the new protocol without matching provenance.
