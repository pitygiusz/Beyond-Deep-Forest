# Beyond Deep Forest

This repository contains an active four-way cascade-forest experiment and an archive of earlier exploratory work.

## Layout

- [Active IFC and projection experiment](experiments/ifc_projection/README.md) compares standard Deep Forest, fixed random nonlinear projections, an adaptation of hiDF-style interaction feature concatenation (IFC), and their combination.
- [Legacy experiments](experiments/legacy/README.md) preserve the earlier outlier-pruning and transformer studies and their results.

## Environment

Use uv with Python 3.12 so the project environment selects the cascade-forest package rather than the unrelated computer-vision package named `deepforest` in WUT.
The [pyproject.toml](pyproject.toml) installs the published [`deep-forest-py310` package](https://pypi.org/project/deep-forest-py310/) and pins NumPy 2.3.5, whose [documented API](https://numpy.org/doc/2.3/reference/generated/numpy.percentile.html) still accepts that package's deprecated `interpolation=` argument.
The package is distributed as source and requires a C compiler; uv builds it with its declared isolated build dependencies.

```bash
uv sync --python 3.12
```

The active comparison needs only the default dependencies.
To install the archived transformer and MAGIC dependencies, run:

```bash
uv sync --extra legacy
```

## Run

From the repository root, run the active four-way comparison with:

```bash
uv run -m experiments.ifc_projection.run_experiment
```

Run the short API example with `uv run -m experiments.ifc_projection.smoke_experiment --variant all`.
The full run writes `experiments/ifc_projection/results/latest.json`, and the previous five-fold result is archived in [breast_cancer_seed42_four_way.json](experiments/ifc_projection/results/breast_cancer_seed42_four_way.json).
See the [active experiment guide](experiments/ifc_projection/README.md) for the class API and options and the [legacy guide](experiments/legacy/README.md) for earlier runners.
