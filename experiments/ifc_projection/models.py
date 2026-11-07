"""Scikit-learn-style access to the four Deep Forest variants."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    from deepforest import CascadeForestClassifier
except ImportError as exc:
    raise ImportError(
        "CascadeForestClassifier is unavailable in this Python environment. "
        "From the project root, run 'uv sync' and then use "
        "'uv run -m experiments.ifc_projection.smoke_experiment'. "
        "The WUT environment may import an unrelated package named deepforest."
    ) from exc

from .random_projected_deep_forest import RandomProjectedCascadeForestClassifier


class DeepForestVariantClassifier(ClassifierMixin, BaseEstimator):
    """A cascade forest selected by ``variant``.

    Parameters
    ----------
    variant : {"bare", "projection", "ifc", "combined"}
        Select standard Deep Forest or one of the three modifications.
    n_estimators : int
        Number of RF/ExtraTrees pairs in each cascade layer.
    n_trees : int
        Number of trees in each forest.

    Other parameters are forwarded to the relevant cascade implementation.
    The IFC variants use the project's hiDF-style interaction adaptation.
    """

    VARIANTS = ("bare", "projection", "ifc", "combined")

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
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")

        cascade_args = dict(
            n_estimators=self.n_estimators,
            n_trees=self.n_trees,
            max_layers=self.max_layers,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            n_bins=self.n_bins,
            bin_type=self.bin_type,
            n_tolerant_rounds=self.n_tolerant_rounds,
            delta=self.delta,
            backend=self.backend,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        if self.variant == "bare":
            model = CascadeForestClassifier(**cascade_args)
        else:
            model = RandomProjectedCascadeForestClassifier(
                **cascade_args,
                projection_width=self.projection_width,
                projection_depth=self.projection_depth,
                negative_slope=self.negative_slope,
                use_projection=self.variant in ("projection", "combined"),
                use_interactions=self.variant in ("ifc", "combined"),
                interaction_limit=self.interaction_limit,
                interaction_bootstraps=self.interaction_bootstraps,
                interaction_trees=self.interaction_trees,
                rit_trees=self.rit_trees,
                rit_depth=self.rit_depth,
                stability_threshold=self.stability_threshold,
            )

        model.fit(X, y, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_layers_ = model.n_layers_
        self.interaction_counts_by_layer_ = (
            [len(model[layer].interaction_generator_.interactions_) for layer in range(model.n_layers_)]
            if self.variant in ("ifc", "combined") else []
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)
