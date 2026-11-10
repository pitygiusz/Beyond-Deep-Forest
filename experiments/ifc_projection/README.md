# IFC and fixed random projections in Deep Forest

This experiment compares four architectures on identical stratified folds: standard Deep Forest, fixed random nonlinear projections, an adaptation of hiDF-style interaction feature concatenation (IFC), and both changes together. The implementation uses `deep-forest-py310==0.1.9`, as specified in [pyproject.toml](../../pyproject.toml).

## Architecture diagrams

These schematics use an example with six input features, four forests, and three classes. The actual feature and class counts depend on the dataset.

### Standard Deep Forest (`bare`)

![Standard Deep Forest cascade with class probabilities and raw features passed to the next layer](diagrams/deep_forest_baseline.png)

### Fixed random projections (`projection`)

![Deep Forest cascade with a separate fixed random projection before each forest](diagrams/deep_forest_projection.svg)

### Interaction feature concatenation (`ifc`)

![Deep Forest cascade with selected interaction features added to the next layer input](diagrams/deep_forest_ifc.svg)

### Projections and IFC (`combined`)

![Deep Forest cascade combining fixed random projections and interaction features](diagrams/deep_forest_combined.svg)

## Methods

Each projected RF or ExtraTrees estimator receives its layer input plus the output of its own seeded, untrained, two-layer Gaussian random mapping with LeakyReLU activations. The random weights stay fixed between fitting and prediction, and the package's binner maps the added features to the `uint8` format required by its custom trees. Forest training, out-of-bag probability features, cascade growth, and early stopping come from `deep-forest-py310`.

The [IFC adapter](interaction_features.py) follows the rule-interaction procedure in the authors' [hiDF source](https://github.com/lyushenhuan/chen2021improving/blob/e1d8d31a8628819d73e75741880d0850c39dfbd1/hiDF/hiDF_utils_threshold.py): bootstrap forests, signed decision rules, random intersection trees, stability filtering, and activated linear-combination features. The adapter also retains two-condition sub-rules of each intersection so repeatable low-order patterns can be matched across bootstrap forests. Selected interaction features join the probability vector passed to the next cascade layer; in the combined model, the interaction extractor can use both raw and projected features from one forest's input. The outer test fold is never passed to the interaction extractor.

This adapts hiDF's feature generation within a supervised cascade classifier. The projection component is inspired by Deep Isolation Forest.

## Setup and run

Sync the uv-managed project environment using the [repository setup instructions](../../README.md#environment).
From the repository root, run the default five-fold comparison with:

```bash
uv run -m experiments.ifc_projection.run_experiment
```

To compare multiple split seeds, use `--seeds 42 43 44`.
To compare only two model variants, use `--variants bare combined`.
For a small diagnostic run, use:

```bash
uv run -m experiments.ifc_projection.run_experiment --folds 2 --n-trees 30 --max-layers 2 --interaction-bootstraps 2 --rit-trees 8
```

The defaults are two RF/ExtraTrees pairs per layer, 50 trees per forest, at most five layers, 16 projected features per forest, at most six IFC features per layer, three bootstrap forests, and 20 random intersection trees per bootstrap.
The runner writes per-fold records and aggregate metrics to `results/latest.json`; the archived outputs are described in [results/README.md](results/README.md).

## Reusable classifier

Import [DeepForestVariantClassifier](models.py) and choose `variant="bare"`, `"projection"`, `"ifc"`, or `"combined"`.
The class supports scikit-learn's `get_params`, `set_params`, `clone`, `fit`, `predict`, `predict_proba`, and `score` interfaces.

```python
from experiments.ifc_projection import DeepForestVariantClassifier

model = DeepForestVariantClassifier(variant="combined", n_trees=50, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

The runnable [smoke script](smoke_experiment.py) trains a small model with this API and accepts `--variant all` to try all four variants.
Run it from the repository root with `uv run -m experiments.ifc_projection.smoke_experiment --variant all`.

## Interpretation

The four-way comparison isolates the two additions, but IFC uses training labels and its training-row features can make the cascade's internal out-of-bag depth criterion optimistic. Treat the archived one-dataset, one-seed results as exploratory until interaction features are cross-fitted and the comparison is repeated across datasets and seeds. The focused checks in [test_random_projected_deep_forest.py](test_random_projected_deep_forest.py) cover seeded projections, interaction selection on a real training fold, and two-layer combined prediction.
