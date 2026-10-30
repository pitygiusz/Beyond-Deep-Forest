# IFC and fixed random projections in Deep Forest

This experiment compares four architectures on identical stratified folds:

1. Standard Deep Forest with probability feature concatenation.
2. Deep Forest with fixed random nonlinear projections before each base forest.
3. Deep Forest with hiDF-style interaction-based feature concatenation (IFC).
4. Deep Forest with both projections and IFC.

Each RF and ExtraTrees forest gets its own seeded, untrained two-layer random
network. The implementation uses scikit-learn's `GaussianRandomProjection` for
each linear map and applies LeakyReLU after each map. Its output is concatenated
with that layer's input before the forest fits or predicts. The same fitted map
is reused at inference. Forest training, out-of-bag probability features,
cascade growth, and early stopping come from a fresh build of `deepforest`
package. No gradients are used.

For IFC, `interaction_features.py` adapts the core procedure in the cloned
hiDF repository: bootstrap forests, signed decision rules, random intersection
trees, stability filtering, and activated linear-combination features. The
adapter also retains two-condition sub-rules of each intersection so stable
low-order patterns can be matched across bootstrap forests. These
features are appended to the probability vector passed to the **next** cascade
layer. In the combined model, the interaction extractor sees the raw and
projected features used by one of the layer's forests. IFC features are
discovered from the outer training fold only.

This is a modern **adaptation** of hiDF's feature generation, not its original
classifier. The cloned package does not import under the current Python 3.12
environment because its compiled tree extension is absent, and its checked-in
classifier lacks a working `predict_proba` path. The adaptation uses the
maintained scikit-learn forest and tree API while keeping the source clone
unchanged. The projection component is DIF-inspired; it is not Deep Isolation
Forest.

## Fresh upstream sources

Both `upstream/` checkouts are ignored local dependencies. They should not be
committed or copied into the parent repository. From this directory, recreate
them at the recorded revisions with:

```bash
mkdir -p upstream
git clone https://github.com/simonprovost/deep_forest_py310.git upstream/deep_forest_py310
git -C upstream/deep_forest_py310 checkout --detach 1a724924a01d5dd966425dde9c38d99746a4039e
git clone https://github.com/lyushenhuan/chen2021improving.git upstream/hiDF
git -C upstream/hiDF checkout --detach e1d8d31a8628819d73e75741880d0850c39dfbd1
```

The hiDF source clone documents the method adapted in
`interaction_features.py`; the Python runtime does not import it. Only the
Deep Forest source is built for execution.

The checked-in Deep Forest source needs compiled extensions. Run the preparation
script with the configured Python environment:

```bash
python prepare_upstream.py
```

This copies the clean source into the ignored `.build/` directory, changes
NumPy's obsolete `interpolation=` percentile argument to `method=` in that
copy, then compiles it. The unmodified fork fails with NumPy 2.5 before forest
training starts. The experiment imports the build copy and does not import
the archived `../../legacy/deep_forest_py310` checkout or change either clone.

Run the four-way comparison on identical stratified folds:

```bash
python run_experiment.py --dataset breast_cancer --folds 5 --seeds 42 43 44
```

The default is two RF/ExtraTrees pairs per layer, 50 trees per forest, up to
five layers, 16 projected features per forest, and up to six IFC features per
layer. IFC uses three bootstrap forests and 20 random intersection trees per
bootstrap. The runner writes per-fold records and aggregate metrics to
`results/latest.json`. Recorded outputs, including the earlier projection-only
run, are described in [results/README.md](results/README.md). For a quick check:

```bash
python run_experiment.py --folds 2 --n-trees 30 --max-layers 2 --interaction-bootstraps 2 --rit-trees 8
```

Each forest also bins its projected features with the package's existing
`Binner`, fitted only on that training fold, so the custom tree backend receives
the required `uint8` features. The projections are learned only in the sense
that their input means are calculated from the training fold; all projection
weights remain random and fixed. The four-way comparison isolates the two
components, but IFC is supervised and its training-row features can make the
internal out-of-bag depth criterion optimistic. Interpret outer-fold test
results as exploratory until this is addressed by a cross-fitted IFC scheme.

## Validation

The focused checks in `test_random_projected_deep_forest.py` cover seeded
projections, selected interactions on a real training fold, and a two-layer
combined cascade. The full archived comparison used the default command with
the configured Python 3.12 environment. The original hiDF source clone remains
unchanged.
