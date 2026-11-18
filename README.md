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

## Run

The experiment in [experiments/run.py](experiments/run.py) runs in one command:

```bash
uv run -m experiments.run --trials 10 --model-seeds 0 1 2 3 4
```

Each dataset gets one stratified 80/20 train/test split. For each model:

1. **Random search.** The default settings and `trials - 1` random
   configurations are scored by balanced accuracy in 3-fold stratified CV on
   the training part. Preprocessing is fitted inside each fold; the test part
   is not used. Search spaces are defined in
   [experiments/models.py](experiments/models.py).
2. **Test evaluation.** The defaults and the best configuration are fitted on
   the whole training part, once per model seed, and scored on the test part.
   The split stays fixed, because the best configuration was chosen on its
   training rows.

By default this covers all tabular datasets and all models (five Deep Forest
variants, Random Forest, ExtraTrees, HistGradientBoosting). Try a small pilot
first:

```bash
uv run -m experiments.run --datasets yeast madelon --models bare projection random_forest --trials 5
```

Non-tabular datasets (`imdb`, `mnist`, `cifar10`, `orl`) can be added with
`--datasets`. Other options: `--cv-folds`, `--seed`, `--test-size`, `--n-jobs`.

Results go to `results/run_<timestamp>/`:

- `settings.json`: the settings of the run.
- `search.csv`: one row per candidate configuration with its CV score
  (`candidate` 0 is the defaults).
- `test.csv`: one row per dataset, model, parameter set (`default` / `tuned`),
  and model seed. The `candidate` column links back to `search.csv`.

To resume an interrupted run, repeat the command with
`--output results/run_<timestamp>`. Finished dataset/model pairs are skipped.
