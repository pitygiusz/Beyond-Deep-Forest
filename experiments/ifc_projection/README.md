# IFC and fixed random projections in Deep Forest

This experiment compares four architectures on identical stratified folds: standard Deep Forest, fixed random nonlinear projections, an adaptation of hiDF-style interaction feature concatenation (IFC), and both changes together. The implementation imports the published `deep-forest-py310==0.1.9` package specified in [requirements.txt](../../requirements.txt); it needs no cloned repository or local build copy.

## Methods

Each projected RF or ExtraTrees estimator receives its layer input plus the output of its own seeded, untrained, two-layer Gaussian random mapping with LeakyReLU activations. The random weights stay fixed between fitting and prediction, and the package's binner maps the added features to the `uint8` format required by its custom trees. Forest training, out-of-bag probability features, cascade growth, and early stopping come from `deep-forest-py310`.

The [IFC adapter](interaction_features.py) follows the rule-interaction procedure in the authors' [hiDF source](https://github.com/lyushenhuan/chen2021improving/blob/e1d8d31a8628819d73e75741880d0850c39dfbd1/hiDF/hiDF_utils_threshold.py): bootstrap forests, signed decision rules, random intersection trees, stability filtering, and activated linear-combination features. The adapter also retains two-condition sub-rules of each intersection so repeatable low-order patterns can be matched across bootstrap forests. Selected interaction features join the probability vector passed to the next cascade layer; in the combined model, the interaction extractor can use both raw and projected features from one forest's input. The outer test fold is never passed to the interaction extractor.

This is an adaptation of hiDF's feature generation, not a run of its original classifier. The original hiDF package has legacy compiled-tree and prediction-path issues under the current Python environment, so it is cited as a method source and is not installed or imported. The projection component is inspired by Deep Isolation Forest but remains part of a supervised cascade classifier.

## Setup and run

Create the virtual environment and install dependencies using the [repository setup instructions](../../README.md#environment). From this directory, run the default five-fold comparison with:

```bash
../../.venv/bin/python run_experiment.py
```

To compare multiple split seeds, use `--seeds 42 43 44`. For a small diagnostic run, use:

```bash
../../.venv/bin/python run_experiment.py --folds 2 --n-trees 30 --max-layers 2 --interaction-bootstraps 2 --rit-trees 8
```

The defaults are two RF/ExtraTrees pairs per layer, 50 trees per forest, at most five layers, 16 projected features per forest, at most six IFC features per layer, three bootstrap forests, and 20 random intersection trees per bootstrap. The runner writes per-fold records and aggregate metrics to the ignored `results/latest.json`; the recorded outputs and their provenance are in [results/README.md](results/README.md). The workspace's VS Code task runs the same default comparison after the virtual environment is installed.

## Interpretation

The four-way comparison isolates the two additions, but IFC uses training labels and its training-row features can make the cascade's internal out-of-bag depth criterion optimistic. Treat the archived one-dataset, one-seed results as exploratory until interaction features are cross-fitted and the comparison is repeated across datasets and seeds. The focused checks in [test_random_projected_deep_forest.py](test_random_projected_deep_forest.py) cover seeded projections, interaction selection on a real training fold, and two-layer combined prediction.
