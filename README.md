# Beyond Deep Forest

This repository contains an active four-way cascade-forest experiment and an
archive of earlier exploratory work.

## Layout

- [Active IFC and projection experiment](experiments/ifc_projection/README.md):
  standard Deep Forest, fixed random nonlinear projections, an adaptation of
  hiDF-style interaction feature concatenation (IFC), and their combination.
- [Archived experiments](legacy/README.md): outlier pruning, a transformer
  module, and earlier benchmark results. These are preserved for reference.

## Run the active experiment

From `experiments/ifc_projection/`, clone the pinned dependencies and run
`python prepare_upstream.py` as described in its README. Then run:

```bash
python run_experiment.py
```

The run writes `results/latest.json`. A VS Code task named **Run Deep Forest
IFC and projection comparison** performs the same run from the workspace root.
The recorded five-fold breast-cancer comparison is in
[results/breast_cancer_seed42_four_way.json](experiments/ifc_projection/results/breast_cancer_seed42_four_way.json).

The upstream source checkouts and compiled build are local, ignored
dependencies. They are not committed to this repository. Their exact revisions
and setup commands are documented in the active experiment README.
