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

From the repository root, run the active four-way comparison with:

```bash
uv run -m experiments.ifc_projection.run_experiment
```
