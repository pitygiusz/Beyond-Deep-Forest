"""Deep Forest with a fixed random nonlinear input map for each forest.

The cascade, out-of-bag probabilities, and stopping rule come from a build of
the fresh ``upstream/deep_forest_py310`` checkout. Each base forest sees the
original layer input plus a distinct untrained random projection of that input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from interaction_features import StableInteractionFeatures

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / ".build" / "deep_forest_py310"))

from deepforest import CascadeForestClassifier
from deepforest._binner import Binner
from deepforest._layer import ClassificationCascadeLayer


class FixedRandomNonlinearMap:
    """A fixed-width random network with LeakyReLU after each projection."""

    def __init__(
        self,
        width: int = 16,
        depth: int = 2,
        negative_slope: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        if not 0 <= negative_slope < 1:
            raise ValueError("negative_slope must be in [0, 1)")
        self.width = width
        self.depth = depth
        self.negative_slope = negative_slope
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> FixedRandomNonlinearMap:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0, keepdims=True)
        states = np.random.default_rng(self.random_state).integers(
            0, 2**32 - 1, size=self.depth, dtype=np.uint32
        )
        self.projections_ = []
        hidden = X - self.mean_
        for seed in states:
            projection = GaussianRandomProjection(
                n_components=self.width, random_state=int(seed)
            )
            hidden = projection.fit_transform(hidden)
            hidden = np.where(hidden >= 0, hidden, self.negative_slope * hidden)
            self.projections_.append(projection)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "projections_"):
            raise AttributeError("Fit the random map before transforming data")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        hidden = X - self.mean_
        for projection in self.projections_:
            hidden = projection.transform(hidden)
            hidden = np.where(hidden >= 0, hidden, self.negative_slope * hidden)
        return np.ascontiguousarray(hidden, dtype=np.float32)


class ProjectedEstimator:
    """Adapter preserving the local Deep Forest estimator's OOB interface."""

    def __init__(
        self,
        estimator,
        random_map: FixedRandomNonlinearMap,
        n_bins: int,
        bin_subsample: int,
        bin_type: str,
    ) -> None:
        self.estimator = estimator
        self.random_map = random_map
        self.projection_binner = Binner(
            n_bins=n_bins, bin_subsample=bin_subsample, bin_type=bin_type
        )

    def _map_input(self, X: np.ndarray) -> np.ndarray:
        original = np.asarray(X, dtype=np.uint8)
        projected = self.projection_binner.transform(self.random_map.transform(original))
        return np.ascontiguousarray(np.hstack((original, projected)), dtype=np.uint8)

    def fit_transform(self, X, y, sample_weight=None):
        self.random_map.fit(X)
        self.projection_binner.fit(self.random_map.transform(X))
        return self.estimator.fit_transform(self._map_input(X), y, sample_weight)

    @property
    def oob_decision_function_(self):
        return self.estimator.oob_decision_function_

    def predict(self, X):
        return self.estimator.predict(self._map_input(X))


class RandomProjectedClassificationCascadeLayer(ClassificationCascadeLayer):
    def __init__(
        self,
        *args,
        projection_width: int = 16,
        projection_depth: int = 2,
        negative_slope: float = 0.1,
        projection_bins: int = 255,
        projection_bin_subsample: int = 200000,
        projection_bin_type: str = "percentile",
        use_projection: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.projection_width = projection_width
        self.projection_depth = projection_depth
        self.negative_slope = negative_slope
        self.projection_bins = projection_bins
        self.projection_bin_subsample = projection_bin_subsample
        self.projection_bin_type = projection_bin_type
        self.use_projection = use_projection

    def _make_estimator(self, estimator_idx, estimator_name):
        estimator = super()._make_estimator(estimator_idx, estimator_name)
        if not self.use_projection:
            return estimator
        if self.random_state is None:
            seed = None
        else:
            seed = int(
                np.random.SeedSequence(
                    [self.random_state, self.layer_idx, estimator_idx, int(estimator_name == "erf")]
                ).generate_state(1)[0]
            )
        random_map = FixedRandomNonlinearMap(
            width=self.projection_width,
            depth=self.projection_depth,
            negative_slope=self.negative_slope,
            random_state=seed,
        )
        return ProjectedEstimator(
            estimator,
            random_map,
            n_bins=self.projection_bins,
            bin_subsample=self.projection_bin_subsample,
            bin_type=self.projection_bin_type,
        )


class InteractionClassificationCascadeLayer(RandomProjectedClassificationCascadeLayer):
    """Add stable hiDF-style rule interactions to the next cascade layer."""

    def __init__(
        self,
        *args,
        interaction_limit: int = 6,
        interaction_bootstraps: int = 3,
        interaction_trees: int = 10,
        rit_trees: int = 20,
        rit_depth: int = 3,
        stability_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.interaction_limit = interaction_limit
        self.interaction_bootstraps = interaction_bootstraps
        self.interaction_trees = interaction_trees
        self.rit_trees = rit_trees
        self.rit_depth = rit_depth
        self.stability_threshold = stability_threshold
        self.is_final_ = False

    def _interaction_input(self, X):
        if self.use_projection:
            first_forest = next(iter(self.estimators_.values()))
            return first_forest._map_input(X)
        return X

    def fit_transform(self, X, y, sample_weight=None):
        probability_features = super().fit_transform(X, y, sample_weight)
        seed = None if self.random_state is None else int(
            np.random.SeedSequence([self.random_state, self.layer_idx, 999]).generate_state(1)[0]
        )
        self.interaction_generator_ = StableInteractionFeatures(
            n_features_out=self.interaction_limit,
            n_bootstraps=self.interaction_bootstraps,
            n_trees=self.interaction_trees,
            n_rit_trees=self.rit_trees,
            rit_depth=self.rit_depth,
            stability_threshold=self.stability_threshold,
            random_state=seed,
        )
        interaction_features = self.interaction_generator_.fit_transform(
            self._interaction_input(X), y
        )
        return np.hstack((probability_features, interaction_features))

    def transform(self, X):
        probability_features = self.predict_full(X)
        if self.is_final_:
            return probability_features
        interaction_features = self.interaction_generator_.transform(
            self._interaction_input(X)
        )
        return np.hstack((probability_features, interaction_features))


class RandomProjectedCascadeForestClassifier(CascadeForestClassifier):
    """Cascade forest with optional fixed projections and hiDF-style IFC."""

    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs=None,
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
        projection_width=16,
        projection_depth=2,
        negative_slope=0.1,
        use_projection=True,
        use_interactions=False,
        interaction_limit=6,
        interaction_bootstraps=3,
        interaction_trees=10,
        rit_trees=20,
        rit_depth=3,
        stability_threshold=0.5,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.projection_width = projection_width
        self.projection_depth = projection_depth
        self.negative_slope = negative_slope
        self.use_projection = use_projection
        self.use_interactions = use_interactions
        self.interaction_limit = interaction_limit
        self.interaction_bootstraps = interaction_bootstraps
        self.interaction_trees = interaction_trees
        self.rit_trees = rit_trees
        self.rit_depth = rit_depth
        self.stability_threshold = stability_threshold

    @property
    def n_aug_features_(self):
        return super().n_aug_features_ + (self.interaction_limit if self.use_interactions else 0)

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        if self.use_interactions:
            for layer_idx in range(self.n_layers_):
                self._get_layer(layer_idx).is_final_ = layer_idx == self.n_layers_ - 1
        return self

    def _make_layer(self, **layer_args):
        options = dict(
            projection_width=self.projection_width,
            projection_depth=self.projection_depth,
            negative_slope=self.negative_slope,
            projection_bins=self.n_bins,
            projection_bin_subsample=self.bin_subsample,
            projection_bin_type=self.bin_type,
            use_projection=self.use_projection,
        )
        if self.use_interactions:
            return InteractionClassificationCascadeLayer(
                **layer_args,
                **options,
                interaction_limit=self.interaction_limit,
                interaction_bootstraps=self.interaction_bootstraps,
                interaction_trees=self.interaction_trees,
                rit_trees=self.rit_trees,
                rit_depth=self.rit_depth,
                stability_threshold=self.stability_threshold,
            )
        return RandomProjectedClassificationCascadeLayer(**layer_args, **options)
