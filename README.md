# Beyond Deep Forest

This repository contains an active four-way cascade-forest experiment and an archive of earlier exploratory work.

## Layout

- [Active IFC and projection experiment](experiments/ifc_projection/README.md) compares standard Deep Forest, fixed random nonlinear projections, an adaptation of hiDF-style interaction feature concatenation (IFC), and their combination.
- [Legacy experiments](experiments/legacy/README.md) preserve the earlier outlier-pruning and transformer studies and their results.

## Environment

Use Python 3.12 and a dedicated virtual environment because the unrelated computer-vision package named `deepforest` has the same import name as the cascade-forest package. The project installs the published [`deep-forest-py310` package](https://pypi.org/project/deep-forest-py310/) rather than keeping source clones in the repository. The package uses NumPy's old `interpolation=` percentile argument, so [requirements.txt](requirements.txt) pins NumPy 2.3.5, whose [documented API](https://numpy.org/doc/2.3/reference/generated/numpy.percentile.html) still accepts it. The package is distributed as source and needs a C compiler, so install its [build requirements](requirements-build.txt) before the runtime requirements.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python -m pip install --no-build-isolation -r requirements.txt
```

The active comparison needs only those packages. To run the archived transformer and MAGIC experiments, install the additional [legacy requirements](experiments/legacy/requirements.txt):

```bash
.venv/bin/python -m pip install -r experiments/legacy/requirements.txt
```

## Run

From the repository root, run the active four-way comparison with:

```bash
.venv/bin/python experiments/ifc_projection/run_experiment.py
```

The run writes `experiments/ifc_projection/results/latest.json`. The VS Code task **Run Deep Forest IFC and projection comparison** uses the same virtual environment and runner. The previous five-fold result is archived in [breast_cancer_seed42_four_way.json](experiments/ifc_projection/results/breast_cancer_seed42_four_way.json). See the [active experiment guide](experiments/ifc_projection/README.md) for methods and options and the [legacy guide](experiments/legacy/README.md) for earlier runners.
