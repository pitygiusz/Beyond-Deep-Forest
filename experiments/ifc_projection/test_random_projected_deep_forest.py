import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import StratifiedKFold

from interaction_features import StableInteractionFeatures
from random_projected_deep_forest import (
    FixedRandomNonlinearMap,
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
