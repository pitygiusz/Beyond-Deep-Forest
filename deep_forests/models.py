"""Scikit-learn-style access to the four Deep Forest variants."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

try:
    from deepforest import CascadeForestClassifier
except ImportError as exc:
    raise ImportError(
        "CascadeForestClassifier is unavailable in this Python environment. "
        "From the project root, run 'uv sync' and then use 'uv run'. "
        "The WUT environment may import an unrelated package named deepforest."
    ) from exc

from .random_projected_deep_forest import RandomProjectedCascadeForestClassifier


class DeepForestVariantClassifier(ClassifierMixin, BaseEstimator):
    """A cascade forest selected by ``variant``.

    Parameters
    ----------
    variant : {"bare", "projection", "ifc", "combined", "utility_ifc"}
        Select standard Deep Forest, a feature variant, or utility-selected IFC.
    n_estimators : int
        Number of RF/ExtraTrees pairs in each cascade layer.
    n_trees : int
        Number of trees in each forest.

    Other parameters are forwarded to the relevant cascade implementation.
    The IFC variants use the project's hiDF-style interaction adaptation.
    utility_ifc selects probability and rule groups by inner-fold utility.
    """

    VARIANTS = ("bare", "projection", "ifc", "combined", "utility_ifc")

    def __init__(
        self,
        variant: str = "bare",
        n_estimators: int = 2,
        n_trees: int = 100,
        max_layers: int = 20,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        n_bins: int = 255,
        bin_type: str = "percentile",
        n_tolerant_rounds: int = 2,
        delta: float = 1e-5,
        backend: str = "custom",
        projection_width: int = 16,
        projection_depth: int = 2,
        negative_slope: float = 0.1,
        interaction_limit: int = 6,
        interaction_bootstraps: int = 3,
        interaction_trees: int = 10,
        rit_trees: int = 20,
        rit_depth: int = 3,
        stability_threshold: float = 0.5,
        interaction_folds: int = 3,
        utility_feature_budget: int = 16,
        utility_cv_folds: int = 3,
        utility_min_gain: float = 0.005,
        utility_cost_penalty: float = 0.001,
        utility_max_groups: int = 4,
        validation_fraction: float = 0.2,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.variant = variant
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.max_layers = max_layers
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.n_bins = n_bins
        self.bin_type = bin_type
        self.n_tolerant_rounds = n_tolerant_rounds
        self.delta = delta
        self.backend = backend
        self.projection_width = projection_width
        self.projection_depth = projection_depth
        self.negative_slope = negative_slope
        self.interaction_limit = interaction_limit
        self.interaction_bootstraps = interaction_bootstraps
        self.interaction_trees = interaction_trees
        self.rit_trees = rit_trees
        self.rit_depth = rit_depth
        self.stability_threshold = stability_threshold
        self.interaction_folds = interaction_folds
        self.utility_feature_budget = utility_feature_budget
        self.utility_cv_folds = utility_cv_folds
        self.utility_min_gain = utility_min_gain
        self.utility_cost_penalty = utility_cost_penalty
        self.utility_max_groups = utility_max_groups
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _new_model(self, max_layers, n_tolerant_rounds):
        cascade_args = dict(
            n_estimators=self.n_estimators,
            n_trees=self.n_trees,
            max_layers=max_layers,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            n_bins=self.n_bins,
            bin_type=self.bin_type,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=self.delta,
            backend=self.backend,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        if self.variant == "bare":
            return CascadeForestClassifier(**cascade_args)
        return RandomProjectedCascadeForestClassifier(
            **cascade_args,
            projection_width=self.projection_width,
            projection_depth=self.projection_depth,
            negative_slope=self.negative_slope,
            use_projection=self.variant in ("projection", "combined"),
            use_interactions=self.variant in ("ifc", "combined", "utility_ifc"),
            use_utility_selection=self.variant == "utility_ifc",
            interaction_limit=self.interaction_limit,
            interaction_bootstraps=self.interaction_bootstraps,
            interaction_trees=self.interaction_trees,
            rit_trees=self.rit_trees,
            rit_depth=self.rit_depth,
            stability_threshold=self.stability_threshold,
            interaction_folds=self.interaction_folds,
            utility_feature_budget=self.utility_feature_budget,
            utility_cv_folds=self.utility_cv_folds,
            utility_min_gain=self.utility_min_gain,
            utility_cost_penalty=self.utility_cost_penalty,
            utility_max_groups=self.utility_max_groups,
        )

    def fit(self, X, y, sample_weight=None):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        X, y = np.asarray(X), np.asarray(y)
        self.depth_validation_scores_ = []
        if self.validation_fraction and self.max_layers > 1:
            indices = np.arange(len(y))
            fit_idx, val_idx = train_test_split(
                indices, test_size=self.validation_fraction, stratify=y,
                random_state=self.random_state,
            )
            weights = None if sample_weight is None else np.asarray(sample_weight)[fit_idx]
            candidate = self._new_model(self.max_layers, self.max_layers)
            candidate.fit(X[fit_idx], y[fit_idx], sample_weight=weights)
            trained_layers = candidate.n_layers_
            for depth in range(1, trained_layers + 1):
                candidate.n_layers_ = depth
                if self.variant in ("ifc", "combined"):
                    for idx, layer in enumerate(candidate.layers_.values()):
                        layer.is_final_ = idx == depth - 1
                prediction = candidate.predict(X[val_idx])
                self.depth_validation_scores_.append(
                    float(balanced_accuracy_score(y[val_idx], prediction))
                )
            self.chosen_depth_ = int(np.argmax(self.depth_validation_scores_)) + 1
            model = self._new_model(self.chosen_depth_, self.chosen_depth_)
        else:
            self.chosen_depth_ = None
            model = self._new_model(self.max_layers, self.n_tolerant_rounds)
        model.fit(X, y, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_layers_ = model.n_layers_
        self.interaction_counts_by_layer_ = (
            [len(model[layer].interaction_generator_.interactions_) for layer in range(model.n_layers_)]
            if self.variant in ("ifc", "combined", "utility_ifc") else []
        )
        self.selected_feedback_by_layer_ = (
            [list(model[layer].utility_selector_.selected_groups_)
             for layer in range(model.n_layers_)]
            if self.variant == "utility_ifc" else []
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)
