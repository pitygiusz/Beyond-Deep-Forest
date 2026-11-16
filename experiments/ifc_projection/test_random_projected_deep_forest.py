import numpy as np
from unittest.mock import patch
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import StratifiedKFold

from experiments.ifc_projection import DeepForestVariantClassifier
from experiments.ifc_projection.interaction_features import StableInteractionFeatures
from experiments.ifc_projection.random_projected_deep_forest import (
    FixedRandomNonlinearMap,
    InteractionClassificationCascadeLayer,
    RandomProjectedCascadeForestClassifier,
)


def test_random_map_is_fixed_and_seeded():
    X = np.arange(60, dtype=np.float32).reshape(12, 5)
    first = FixedRandomNonlinearMap(width=4, depth=2, random_state=13).fit(X)
    same_seed = FixedRandomNonlinearMap(width=4, depth=2, random_state=13).fit(X)
    other_seed = FixedRandomNonlinearMap(width=4, depth=2, random_state=14).fit(X)

    np.testing.assert_array_equal(first.transform(X), first.transform(X))
    np.testing.assert_array_equal(first.transform(X), same_seed.transform(X))
    assert not np.array_equal(first.transform(X), other_seed.transform(X))


def test_projected_cascade_uses_distinct_fixed_maps():
    X, y = make_classification(n_samples=90, n_features=8, n_informative=5, random_state=3)
    model = RandomProjectedCascadeForestClassifier(
        n_estimators=2,
        n_trees=50,
        max_layers=2,
        projection_width=4,
        projection_depth=2,
        random_state=7,
        n_jobs=1,
        verbose=0,
    )
    model.fit(X, y)

    layer = model[0]
    forests = list(layer.estimators_.values())
    assert len(forests) == 4
    assert all(forest.estimator.estimator_.n_features_ == X.shape[1] + 4 for forest in forests)
    assert not np.array_equal(
        forests[0].random_map.transform(X), forests[1].random_map.transform(X)
    )
    probability = model.predict_proba(X[:5])
    assert probability.shape == (5, 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)


def test_projection_is_fitted_on_unbinned_raw_features():
    X, y = make_classification(n_samples=90, n_features=8, n_informative=5, random_state=3)
    X = X * np.arange(1, 9) + 10.0
    model = RandomProjectedCascadeForestClassifier(
        n_estimators=1, n_trees=30, max_layers=1, projection_width=4,
        random_state=7, n_jobs=1, verbose=0,
    ).fit(X, y)
    forest = next(iter(model[0].estimators_.values()))
    np.testing.assert_allclose(forest.random_map.mean_.ravel(), X.mean(axis=0), rtol=1e-5)
    assert forest.random_map.n_features_in_ == X.shape[1]
    assert forest.estimator.estimator_.n_features_ == X.shape[1] + 4
    assert model.predict_proba(X[:4]).shape == (4, 2)


def test_zero_projection_budget_matches_unprojected_cascade():
    X, y = make_classification(n_samples=90, n_features=8, n_informative=5, random_state=3)
    projected = RandomProjectedCascadeForestClassifier(
        n_estimators=1, n_trees=30, max_layers=1, projection_width=0,
        use_projection=True, random_state=7, n_jobs=1, verbose=0,
    ).fit(X, y)
    unprojected = RandomProjectedCascadeForestClassifier(
        n_estimators=1, n_trees=30, max_layers=1, use_projection=False,
        random_state=7, n_jobs=1, verbose=0,
    ).fit(X, y)
    np.testing.assert_allclose(projected.predict_proba(X[:10]), unprojected.predict_proba(X[:10]))


def test_ifc_training_features_exclude_each_rows_label():
    class MembershipExtractor:
        def __init__(self, n_features_out, **_kwargs):
            self.n_features_out = n_features_out

        def fit(self, X, y):
            self.seen = set(X[:, 0])
            self.interactions_ = [0]
            return self

        def transform(self, X):
            return np.repeat(
                np.isin(X[:, 0], list(self.seen))[:, None].astype(np.float32),
                self.n_features_out, axis=1,
            )

    X = np.column_stack((np.arange(60), np.arange(60) % 5)).astype(np.uint8)
    y = np.arange(60) % 2
    layer = InteractionClassificationCascadeLayer(
        layer_idx=0, n_outputs=2, criterion="gini", n_estimators=1,
        n_trees=30, use_projection=False, interaction_limit=2,
        interaction_folds=3, random_state=7, n_jobs=1, verbose=0,
    )
    with patch(
        "experiments.ifc_projection.random_projected_deep_forest.StableInteractionFeatures",
        MembershipExtractor,
    ):
        training_features = layer.fit_transform(X, y)
        np.testing.assert_array_equal(training_features[:, -2:], 0)
        np.testing.assert_array_equal(layer.interaction_generator_.transform(X), 1)


def test_interaction_features_are_stable_and_usable_on_new_rows():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5, random_state=3)
    first = StableInteractionFeatures(
        n_features_out=4, n_bootstraps=2, n_trees=5,
        n_rit_trees=8, stability_threshold=0.5, random_state=17,
    ).fit(X[:90], y[:90])
    repeated = StableInteractionFeatures(
        n_features_out=4, n_bootstraps=2, n_trees=5,
        n_rit_trees=8, stability_threshold=0.5, random_state=17,
    ).fit(X[:90], y[:90])
    assert len(first.interactions_) > 0
    np.testing.assert_array_equal(first.transform(X[90:]), repeated.transform(X[90:]))


def test_combined_cascade_passes_ifc_features_between_layers():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5, random_state=3)
    model = RandomProjectedCascadeForestClassifier(
        n_estimators=2, n_trees=30, max_layers=2, projection_width=4,
        use_interactions=True, interaction_limit=4, interaction_bootstraps=2,
        interaction_trees=5, rit_trees=8, stability_threshold=0.5,
        random_state=7, n_jobs=1, verbose=0,
    ).fit(X, y)
    assert model.n_layers_ == 2
    assert len(model[0].interaction_generator_.interactions_) > 0
    assert model[0].is_final_ is False
    assert model[1].is_final_ is True
    probability = model.predict_proba(X[:5])
    assert probability.shape == (5, 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)


def test_ifc_selects_features_on_breast_cancer_training_fold():
    X, y = load_breast_cancer(return_X_y=True)
    train_index, _ = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    seed = int(np.random.SeedSequence([42001, 0, 999]).generate_state(1)[0])
    extractor = StableInteractionFeatures(random_state=seed).fit(X[train_index], y[train_index])
    assert len(extractor.interactions_) > 0


def test_public_estimator_is_cloneable_and_predicts():
    X, y = make_classification(n_samples=100, n_features=8, n_informative=5, random_state=3)
    estimator = DeepForestVariantClassifier(
        variant="combined", n_estimators=2, n_trees=30, max_layers=2,
        projection_width=4, interaction_limit=4, interaction_bootstraps=2,
        interaction_trees=5, rit_trees=8, random_state=7, n_jobs=1,
    )
    model = clone(estimator).fit(X, y)
    assert model.get_params()["variant"] == "combined"
    assert model.predict(X[:5]).shape == (5,)
    assert model.predict_proba(X[:5]).shape == (5, 2)


def test_depth_is_selected_on_separate_validation_rows():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5, random_state=3)
    model = DeepForestVariantClassifier(
        variant="ifc", n_estimators=1, n_trees=30, max_layers=3,
        interaction_limit=2, interaction_bootstraps=2, interaction_trees=3,
        rit_trees=4, interaction_folds=2, random_state=7, n_jobs=1,
    ).fit(X, y)
    assert len(model.depth_validation_scores_) == 3
    assert model.chosen_depth_ == np.argmax(model.depth_validation_scores_) + 1
    assert model.n_layers_ == model.chosen_depth_
