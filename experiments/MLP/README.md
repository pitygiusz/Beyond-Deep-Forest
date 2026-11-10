# Dataset-adapted MLP

This experiment trains a fully connected neural network on the same default datasets, 70/30 stratified split, seeds, and metrics as `experiments.run`. It produces two paired rows for each dataset and seed:

- `standardized` uses `prepare_data.get_train_test`, exactly matching the feature matrix passed to the Deep Forest models.
- `raw` keeps numeric inputs unscaled as a separate MLP ablation; it still median-imputes missing values and one-hot encodes categoricals.

Both modes use the same split seed. IMDB reviews become 5,000-dimensional training-fold TF-IDF vectors in both modes. ORL, MNIST, and CIFAR-10 images are flattened into 1D pixel vectors. Sequential sEMG and acoustic GTZAN feature vectors (including flattened MFCC matrices) use the sequential profile when those datasets are added to the data loader.

The MLP uses ten tapered hidden layers by default, matching the Deep Forest runner's `max_layers=10` ceiling. Widths depend on the input family: tabular starts at 128 units, text and sequential features at 256, and images at 512, then taper toward 16 units. Use `--n-layers` to select 1 to 10 hidden layers. CIFAR-10 remains selectable but is not part of the forest runner's default dataset list. Images do not retain spatial structure, while flattened sequential/audio features do not retain explicit temporal or spectral structure. `sEMG` and `GTZAN` are not currently registered in `prepare_data.py`, so they cannot yet be selected by the experiment runner.

The `standardized` rows are the closest input-level comparison because they use the same transformed features as the forests. The `raw` rows show how the MLP behaves without numeric scaling. This is a matched-protocol baseline, not a capacity-equivalent model: a forest cascade layer and a fully connected neural layer have different computation and parameterization. The forest may also stop before its ten-layer ceiling, while the MLP's hidden-layer count is fixed for each run. These runs use explicit configurations rather than automatic hyperparameter search; tuning only the MLP would give it a larger selection budget than the current forest runs.

## Standard venv setup

Use Python 3.12. From the repository root in PowerShell, create or repair the project venv and install only the MLP dependencies:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements-mlp.txt
```

This MLP requirements file avoids installing the Deep Forest package and its build requirements. Run all supported datasets and five seeds with:

```bash
.\.venv\Scripts\python.exe -m experiments.MLP.run_experiment
```

Select datasets or seeds and adjust training length with:

```bash
.\.venv\Scripts\python.exe -m experiments.MLP.run_experiment --datasets yeast imdb mnist --seeds 0 1 --n-layers 10 --max-iter 200
```

Results append to `experiments/MLP/results/mlp_benchmark.csv`, leaving previous `mlp.csv` results untouched. Completed dataset/seed/mode/configuration combinations are skipped when the command is rerun. Select one preprocessing mode with `--input-modes standardized` or `--input-modes raw`. The model factory is available as `experiments.MLP.make_mlp`.