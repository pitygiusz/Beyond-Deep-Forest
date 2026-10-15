"""
Implementation of Deep Forest with Outlier Forest Pruning from the Augmented Vector.

Original Architecture:
At each cascade layer, an ensemble of forests (e.g. 4 Random Forests + 4 Completely Random Forests)
outputs class probability vectors. All M probability vectors are concatenated with the original
features to form the Augmented Vector.

Modified Architecture:
At each cascade layer, the consensus probability distribution is computed across all M forests.
The divergence (e.g., Euclidean distance) of each forest's prediction from the consensus is measured.
The 2 outlier forests exhibiting the largest divergence are removed from the Augmented Vector,
passing only the (M - 2) consensus forests' probability vectors to the subsequent layer.
"""

import sys
import os
import time
import numpy as np
from sklearn.base import is_classifier, ClassifierMixin
from sklearn.metrics import accuracy_score

# Ensure the local deepforest package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "deep_forest_py310")))

from deepforest import CascadeForestClassifier
from deepforest._layer import ClassificationCascadeLayer, _build_estimator
from deepforest import _utils


class OutlierPrunedClassificationCascadeLayer(ClassificationCascadeLayer):
    """
    Cascade layer that identifies outlier forests based on prediction divergence from
    the ensemble consensus, and removes them from the augmented vector.
    """

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
        n_outliers_removed=2,
        divergence_metric="euclidean",
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_outliers_removed = n_outliers_removed
        self.divergence_metric = divergence_metric

        # Outlier tracking metadata
        self.estimator_keys_ = []
        self.forest_divergences_ = []
        self.outlier_indices_ = []
        self.kept_indices_ = []

    def fit_transform(self, X, y, sample_weight=None):
        self._validate_params()
        n_samples, self.n_features = X.shape

        X_aug = []
        self.estimator_keys_ = []
        oob_decision_function_full = np.zeros((n_samples, self.n_outputs))

        # 1. Fit Random Forests (self.n_estimators // 2)
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "rf",
                self._make_estimator(estimator_idx, "rf"),
                oob_decision_function_full,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "rf")
            self.estimators_.update({key: _estimator})
            self.estimator_keys_.append(key)

        # 2. Fit Extra-Trees / Completely Random Forests (self.n_estimators // 2)
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "erf",
                self._make_estimator(estimator_idx, "erf"),
                oob_decision_function_full,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "erf")
            self.estimators_.update({key: _estimator})
            self.estimator_keys_.append(key)

        total_estimators = len(X_aug)

        # 3. Compute Consensus (Ensemble Mean Probability)
        # Shape: (n_samples, n_outputs)
        consensus_proba = np.mean(X_aug, axis=0)

        # 4. Compute Divergence of each forest from consensus
        divergences = []
        for idx in range(total_estimators):
            est_proba = X_aug[idx]
            if self.divergence_metric == "euclidean":
                # Average Euclidean distance across samples
                dist = np.mean(np.sqrt(np.sum((est_proba - consensus_proba) ** 2, axis=1)))
            elif self.divergence_metric == "mse":
                dist = np.mean((est_proba - consensus_proba) ** 2)
            elif self.divergence_metric == "l1":
                dist = np.mean(np.sum(np.abs(est_proba - consensus_proba), axis=1))
            else:
                dist = np.mean(np.sqrt(np.sum((est_proba - consensus_proba) ** 2, axis=1)))
            divergences.append(float(dist))

        self.forest_divergences_ = divergences

        # 5. Identify the Outlier Forests (largest divergence)
        sorted_indices = np.argsort(divergences)
        n_outliers = min(self.n_outliers_removed, max(0, total_estimators - 1))
        n_kept = total_estimators - n_outliers

        self.outlier_indices_ = sorted_indices[n_kept:].tolist()
        # Keep remaining indices sorted for stable feature column positions
        self.kept_indices_ = sorted(sorted_indices[:n_kept].tolist())

        if self.verbose > 0:
            outlier_names = [self.estimator_keys_[i] for i in self.outlier_indices_]
            kept_names = [self.estimator_keys_[i] for i in self.kept_indices_]
            print(f"[{_utils.ctime()}] Layer {self.layer_idx} Outlier Forests: {outlier_names}")
            print(f"[{_utils.ctime()}] Layer {self.layer_idx} Kept Forests:    {kept_names}")

        # 6. Set OOB decision function from consensus (kept) forests
        kept_probas = [X_aug[i] for i in self.kept_indices_]
        self.oob_decision_function_ = np.mean(kept_probas, axis=0)
        y_pred = np.argmax(self.oob_decision_function_, axis=1)
        self.val_performance_ = accuracy_score(y, y_pred, sample_weight=sample_weight)

        # 7. Form Augmented Vector using ONLY kept consensus forests
        X_aug_pruned = np.hstack(kept_probas)
        return X_aug_pruned

    def predict_full(self, X):
        """
        Return the concatenated predictions from only the kept consensus base estimators,
        pruning the outlier forests identified during training.
        """
        pred_list = []
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key_short = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key_short, self.layer_idx))
            if self.partial_mode:
                estimator = self.buffer.load_estimator(estimator)
            pred_list.append(estimator.predict(X))

        # Select only the kept consensus forests
        if self.kept_indices_ is not None and len(self.kept_indices_) > 0:
            pred_pruned = np.hstack([pred_list[i] for i in self.kept_indices_])
        else:
            pred_pruned = np.hstack(pred_list)

        return pred_pruned

    def transform(self, X):
        return self.predict_full(X)


class OutlierPrunedCascadeForestClassifier(CascadeForestClassifier):
    """
    Deep Forest Classifier that removes the two outlier forests from the
    augmented feature vector at each cascade level.
    """

    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=4,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs=None,
        backend="custom",
        n_tolerant_rounds=2,
        delta=0.0,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
        n_outliers_removed=2,
        divergence_metric="euclidean",
    ):
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
        self.n_outliers_removed = n_outliers_removed
        self.divergence_metric = divergence_metric

    @property
    def n_aug_features_(self):
        """
        Number of augmented features appended to raw features.
        Reflects pruning of outlier forests.
        """
        total_estimators = 2 * self.n_estimators
        n_kept = max(1, total_estimators - self.n_outliers_removed)
        return n_kept * self.n_outputs_

    def _make_layer(self, **layer_args):
        """Instantiate the outlier-pruned cascade layer."""
        return OutlierPrunedClassificationCascadeLayer(
            **layer_args,
            n_outliers_removed=self.n_outliers_removed,
            divergence_metric=self.divergence_metric,
        )

    def get_pruning_report(self):
        """
        Returns a detailed summary of outlier pruning across all cascade layers.
        """
        if not self.is_fitted_:
            raise AttributeError("Model has not been fitted yet.")

        report = []
        for l_idx in range(self.n_layers_):
            layer = self._get_layer(l_idx)
            info = {
                "layer_idx": l_idx,
                "n_total_estimators": len(layer.estimator_keys_),
                "n_outliers_removed": len(layer.outlier_indices_),
                "n_kept_estimators": len(layer.kept_indices_),
                "estimators": [
                    {
                        "name": layer.estimator_keys_[i],
                        "divergence": layer.forest_divergences_[i],
                        "is_outlier": i in layer.outlier_indices_,
                    }
                    for i in range(len(layer.estimator_keys_))
                ],
                "val_performance": layer.val_performance_,
            }
            report.append(info)
        return report
