"""Deep Forest with a fixed random nonlinear input map for each forest.

The cascade, out-of-bag probabilities, and stopping rule come from the
``deep-forest-py310`` package. Each base forest sees its binned layer input
plus a distinct projection of the unbinned raw features. The projected values
are binned only after the random map has been applied.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.random_projection import GaussianRandomProjection

from .interaction_features import StableInteractionFeatures
from .utility_feedback import UtilitySelectedFeedback

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
        raw_fit_X: np.ndarray,
        n_bins: int,
        bin_subsample: int,
        bin_type: str,
    ) -> None:
        self.estimator = estimator
        self.random_map = random_map
        self.raw_fit_X = raw_fit_X
        self.projection_binner = Binner(
            n_bins=n_bins, bin_subsample=bin_subsample, bin_type=bin_type
        )

    def _map_input(self, X: np.ndarray, raw_X: np.ndarray) -> np.ndarray:
        original = np.asarray(X, dtype=np.uint8)
        raw_X = np.asarray(raw_X, dtype=np.float32)
        if len(raw_X) != len(original):
            raise ValueError("Raw and binned inputs must have the same rows")
        projected = self.projection_binner.transform(self.random_map.transform(raw_X))
        return np.ascontiguousarray(np.hstack((original, projected)), dtype=np.uint8)

    def fit_transform(self, X, y, sample_weight=None):
        raw_X = self.raw_fit_X
        self.random_map.fit(raw_X)
        self.projection_binner.fit(self.random_map.transform(raw_X))
        result = self.estimator.fit_transform(self._map_input(X, raw_X), y, sample_weight)
        self.raw_fit_X = None
        return result

    @property
    def oob_decision_function_(self):
        return self.estimator.oob_decision_function_

    def predict(self, X, raw_X):
        return self.estimator.predict(self._map_input(X, raw_X))


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
        raw_fit_X: np.ndarray | None = None,
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
        self.raw_fit_X = raw_fit_X

    def _make_estimator(self, estimator_idx, estimator_name):
        estimator = super()._make_estimator(estimator_idx, estimator_name)
        if not self.use_projection or self.projection_width == 0:
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
            raw_fit_X=self.raw_fit_X,
            n_bins=self.projection_bins,
            bin_subsample=self.projection_bin_subsample,
            bin_type=self.projection_bin_type,
        )

    def predict_full(self, X):
        if not self.use_projection or self.projection_width == 0:
            return super().predict_full(X)
        raw_X = self.raw_predict_X
        pred = np.zeros((len(X), self.n_outputs * self.n_estimators))
        for idx, stored in enumerate(self.estimators_.values()):
            estimator = self.buffer.load_estimator(stored) if self.partial_mode else stored
            left, right = idx * self.n_outputs, (idx + 1) * self.n_outputs
            pred[:, left:right] = estimator.predict(X, raw_X)
        return pred


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
        interaction_folds: int = 3,
        use_utility_selection: bool = False,
        utility_feature_budget: int = 16,
        utility_cv_folds: int = 3,
        utility_min_gain: float = 0.005,
        utility_cost_penalty: float = 0.001,
        utility_max_groups: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.interaction_limit = interaction_limit
        self.interaction_bootstraps = interaction_bootstraps
        self.interaction_trees = interaction_trees
        self.rit_trees = rit_trees
        self.rit_depth = rit_depth
        self.stability_threshold = stability_threshold
        self.interaction_folds = interaction_folds
        self.use_utility_selection = use_utility_selection
        self.utility_feature_budget = utility_feature_budget
        self.utility_cv_folds = utility_cv_folds
        self.utility_min_gain = utility_min_gain
        self.utility_cost_penalty = utility_cost_penalty
        self.utility_max_groups = utility_max_groups
        self.is_final_ = False

    def _interaction_input(self, X):
        if self.use_projection and self.projection_width > 0:
            first_forest = next(iter(self.estimators_.values()))
            if self.partial_mode:
                first_forest = self.buffer.load_estimator(first_forest)
            raw_X = self.raw_fit_X if self.raw_fit_X is not None else self.raw_predict_X
            return first_forest._map_input(X, raw_X)
        return X

    def _new_interaction_generator(self, seed):
        return StableInteractionFeatures(
            n_features_out=self.interaction_limit,
            n_bootstraps=self.interaction_bootstraps,
            n_trees=self.interaction_trees,
            n_rit_trees=self.rit_trees,
            rit_depth=self.rit_depth,
            stability_threshold=self.stability_threshold,
            random_state=seed,
        )

    def fit_transform(self, X, y, sample_weight=None):
        probability_features = ClassificationCascadeLayer.fit_transform(self, X, y, sample_weight)
        seed = None if self.random_state is None else int(
            np.random.SeedSequence([self.random_state, self.layer_idx, 999]).generate_state(1)[0]
        )
        if self.interaction_folds < 2 or self.interaction_folds > len(y):
            raise ValueError("interaction_folds must be between 2 and the sample count")
        interaction_input = self._interaction_input(X)
        interaction_features = np.zeros((len(y), self.interaction_limit), dtype=np.float32)
        folds = KFold(n_splits=self.interaction_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, holdout_idx) in enumerate(folds.split(interaction_input)):
            fold_seed = None if seed is None else int(
                np.random.SeedSequence([seed, fold]).generate_state(1)[0]
            )
            extractor = self._new_interaction_generator(fold_seed)
            extractor.fit(interaction_input[train_idx], np.asarray(y)[train_idx])
            interaction_features[holdout_idx] = extractor.transform(interaction_input[holdout_idx])
        # The full-data extractor supplies the same rank-ordered channels at inference.
        self.interaction_generator_ = self._new_interaction_generator(seed)
        self.interaction_generator_.fit(interaction_input, y)
        self.raw_fit_X = None
        candidates = np.hstack((probability_features, interaction_features))
        if not self.use_utility_selection:
            return candidates

        n_classes = self.n_outputs
        groups = [
            (f"forest_{idx}", np.arange(idx * n_classes, (idx + 1) * n_classes))
            for idx in range(probability_features.shape[1] // n_classes)
        ]
        interaction_offset = probability_features.shape[1]
        groups.extend(
            (f"ifc_{idx}", np.asarray([interaction_offset + idx]))
            for idx in range(interaction_features.shape[1])
        )
        self.utility_selector_ = UtilitySelectedFeedback(
            n_features_out=max(self.utility_feature_budget, n_classes),
            n_splits=self.utility_cv_folds,
            min_gain=self.utility_min_gain,
            cost_penalty=self.utility_cost_penalty,
            max_groups=self.utility_max_groups,
            random_state=seed,
        ).fit(X, candidates, y, groups)
        return self.utility_selector_.transform(candidates)

    def transform(self, X):
        probability_features = self.predict_full(X)
        if self.is_final_:
            return probability_features
        interaction_features = self.interaction_generator_.transform(
            self._interaction_input(X)
        )
        candidates = np.hstack((probability_features, interaction_features))
        if self.use_utility_selection:
            return self.utility_selector_.transform(candidates)
        return candidates


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
        interaction_folds=3,
        use_utility_selection=False,
        utility_feature_budget=16,
        utility_cv_folds=3,
        utility_min_gain=0.005,
        utility_cost_penalty=0.001,
        utility_max_groups=4,
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
        self.interaction_folds = interaction_folds
        self.use_utility_selection = use_utility_selection
        self.utility_feature_budget = utility_feature_budget
        self.utility_cv_folds = utility_cv_folds
        self.utility_min_gain = utility_min_gain
        self.utility_cost_penalty = utility_cost_penalty
        self.utility_max_groups = utility_max_groups
        if self.use_utility_selection and not self.use_interactions:
            raise ValueError("Utility selection currently requires IFC candidate features")

    def fit(self, X, y, sample_weight=None):
        if self.projection_width < 0:
            raise ValueError("projection_width must be nonnegative")
        self.raw_fit_X = (
            np.asarray(X, dtype=np.float32)
            if self.use_projection and self.projection_width > 0 else None
        )
        try:
            super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.raw_fit_X = None
            for layer in getattr(self, "layers_", {}).values():
                layer.raw_fit_X = None
        if self.use_interactions:
            for layer_idx in range(self.n_layers_):
                self._get_layer(layer_idx).is_final_ = layer_idx == self.n_layers_ - 1
        return self

    @property
    def n_aug_features_(self):
        if self.use_utility_selection:
            return max(self.utility_feature_budget, self.n_outputs_)
        return super().n_aug_features_ + (self.interaction_limit if self.use_interactions else 0)

    def predict_proba(self, X):
        raw_X = np.asarray(X, dtype=np.float32)
        if self.use_projection and self.projection_width > 0:
            for layer in self.layers_.values():
                layer.raw_predict_X = raw_X
        try:
            return super().predict_proba(X)
        finally:
            if self.use_projection and self.projection_width > 0:
                for layer in self.layers_.values():
                    layer.raw_predict_X = None

    def _make_layer(self, **layer_args):
        options = dict(
            projection_width=self.projection_width,
            projection_depth=self.projection_depth,
            negative_slope=self.negative_slope,
            projection_bins=self.n_bins,
            projection_bin_subsample=self.bin_subsample,
            projection_bin_type=self.bin_type,
            use_projection=self.use_projection,
            raw_fit_X=self.raw_fit_X,
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
                interaction_folds=self.interaction_folds,
                use_utility_selection=self.use_utility_selection,
                utility_feature_budget=self.utility_feature_budget,
                utility_cv_folds=self.utility_cv_folds,
                utility_min_gain=self.utility_min_gain,
                utility_cost_penalty=self.utility_cost_penalty,
                utility_max_groups=self.utility_max_groups,
            )
        return RandomProjectedClassificationCascadeLayer(**layer_args, **options)
