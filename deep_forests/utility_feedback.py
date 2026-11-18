"""Select cascade feedback by conditional validation utility and feature cost."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


class UtilitySelectedFeedback:
    """Greedily retain feedback groups that improve inner-fold balanced accuracy.

    Forest probabilities stay grouped by forest. IFC features are singleton
    groups. Each addition is scored conditional on the already selected
    groups, so redundant signals have to earn their incremental value. The
    output has fixed width for the surrounding cascade's preallocated arrays.
    """

    def __init__(
        self,
        n_features_out: int,
        n_splits: int = 3,
        min_gain: float = 0.005,
        cost_penalty: float = 0.001,
        max_groups: int = 4,
        random_state: int | None = None,
    ) -> None:
        if n_features_out < 1 or n_splits < 2 or max_groups < 1:
            raise ValueError("Feature budget, fold count, and group limit must be positive")
        if min_gain < 0 or cost_penalty < 0:
            raise ValueError("Utility thresholds must be nonnegative")
        self.n_features_out = n_features_out
        self.n_splits = n_splits
        self.min_gain = min_gain
        self.cost_penalty = cost_penalty
        self.max_groups = max_groups
        self.random_state = random_state

    def _score(self, context, candidate, y, selected):
        design = np.hstack((context, candidate[:, selected])) if selected else context
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(design, y)):
            counts = np.bincount(y[train_idx].astype(int))
            weights = len(train_idx) / (len(counts) * counts[y[train_idx].astype(int)])
            model = HistGradientBoostingClassifier(
                max_iter=50,
                max_leaf_nodes=15,
                min_samples_leaf=max(2, min(10, len(train_idx) // 10)),
                l2_regularization=1.0,
                random_state=None if self.random_state is None else self.random_state + fold,
            )
            model.fit(design[train_idx], y[train_idx], sample_weight=weights)
            scores.append(balanced_accuracy_score(y[valid_idx], model.predict(design[valid_idx])))
        return float(np.mean(scores))

    def fit(self, context, candidates, y, groups):
        context = np.asarray(context, dtype=np.float32)
        candidates = np.asarray(candidates, dtype=np.float32)
        y = np.asarray(y)
        if context.ndim != 2 or candidates.ndim != 2 or len(context) != len(y) or len(candidates) != len(y):
            raise ValueError("Expected matching 2D context/candidate arrays and 1D labels")
        _, y_encoded = np.unique(y, return_inverse=True)
        if self.n_splits > np.bincount(y_encoded).min():
            raise ValueError("n_splits cannot exceed the number of samples in the rarest class")

        normalized_groups = [(str(name), np.asarray(columns, dtype=int)) for name, columns in groups]
        if any(not len(columns) or columns.min() < 0 or columns.max() >= candidates.shape[1]
               for _, columns in normalized_groups):
            raise ValueError("Candidate groups contain invalid feature indices")
        if len(set(np.concatenate([columns for _, columns in normalized_groups]))) != sum(
            len(columns) for _, columns in normalized_groups
        ):
            raise ValueError("Candidate groups must not overlap")

        self.groups_ = normalized_groups
        self.selected_groups_ = []
        self.selected_indices_ = []
        self.baseline_score_ = self._score(context, candidates, y_encoded, [])
        current_score = self.baseline_score_
        remaining = list(range(len(self.groups_)))
        history = []
        while remaining and len(self.selected_groups_) < self.max_groups:
            best = None
            for group_idx in remaining:
                name, columns = self.groups_[group_idx]
                trial = self.selected_indices_ + columns.tolist()
                if len(trial) > self.n_features_out:
                    continue
                raw_score = self._score(context, candidates, y_encoded, trial)
                gain = raw_score - current_score
                net_gain = gain - self.cost_penalty * len(columns) / self.n_features_out
                history.append({"group": name, "score": raw_score,
                                "gain": gain, "net_gain": net_gain})
                if best is None or net_gain > best[0]:
                    best = (net_gain, raw_score, group_idx)
            if best is None or best[0] <= self.min_gain:
                break
            _, current_score, group_idx = best
            name, columns = self.groups_[group_idx]
            self.selected_groups_.append(name)
            self.selected_indices_.extend(columns.tolist())
            remaining.remove(group_idx)

        self.selected_score_ = current_score
        self.selection_history_ = history
        self.n_features_in_ = candidates.shape[1]
        return self

    def transform(self, candidates):
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != self.n_features_in_:
            raise ValueError("Candidate features differ from those used during fit")
        output = np.zeros((len(candidates), self.n_features_out), dtype=np.float64)
        count = len(self.selected_indices_)
        if count:
            output[:, :count] = candidates[:, self.selected_indices_]
        return output
