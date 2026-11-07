"""A modern, small adaptation of hiDF's interaction feature construction.

The source method is in
``https://github.com/lyushenhuan/chen2021improving/blob/e1d8d31a8628819d73e75741880d0850c39dfbd1/hiDF/hiDF_utils_threshold.py``:
bootstrap forests -> signed root-to-leaf rules -> random intersection trees
-> stability filtering -> activated linear-combination (ALC) features.
The cloned implementation cannot be imported under the current Python 3.12
environment, so this module uses scikit-learn's maintained forest and tree API.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class StableInteractionFeatures:
    """Generate fixed-width IFC features from stable signed tree-rule overlaps."""

    def __init__(
        self,
        n_features_out: int = 6,
        n_bootstraps: int = 3,
        n_trees: int = 10,
        n_rit_trees: int = 20,
        rit_depth: int = 3,
        stability_threshold: float = 0.5,
        sample_fraction: float = 0.7,
        random_state: int | None = None,
    ) -> None:
        if min(n_features_out, n_bootstraps, n_trees, n_rit_trees, rit_depth) < 1:
            raise ValueError("Feature, bootstrap, tree, and depth counts must be positive")
        if not 0 < stability_threshold <= 1 or not 0 < sample_fraction <= 1:
            raise ValueError("stability_threshold and sample_fraction must be in (0, 1]")
        self.n_features_out = n_features_out
        self.n_bootstraps = n_bootstraps
        self.n_trees = n_trees
        self.n_rit_trees = n_rit_trees
        self.rit_depth = rit_depth
        self.stability_threshold = stability_threshold
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    @staticmethod
    def _leaf_paths(forest):
        paths = []
        weights = []
        for tree in forest.estimators_:
            structure = tree.tree_
            stack = [(0, ())]
            while stack:
                node, path = stack.pop()
                left = int(structure.children_left[node])
                if left == -1:
                    if len({condition[0] for condition in path}) >= 2:
                        paths.append(path)
                        weights.append(float(structure.weighted_n_node_samples[node]))
                    continue
                right = int(structure.children_right[node])
                feature = int(structure.feature[node])
                threshold = float(structure.threshold[node])
                stack.append((left, path + ((feature, -1, threshold),)))
                stack.append((right, path + ((feature, 1, threshold),)))
        return paths, np.asarray(weights, dtype=float)

    @staticmethod
    def _intersect(first, second):
        """Intersect threshold rules, retaining the stricter bound per sign."""
        bounds = {}
        for feature, direction, threshold in first + second:
            key = (feature, direction)
            if key in bounds:
                bounds[key] = min(bounds[key], threshold) if direction == -1 else max(bounds[key], threshold)
            else:
                bounds[key] = threshold
        for feature in {feature for feature, _ in bounds}:
            lower = bounds.get((feature, 1), -np.inf)
            upper = bounds.get((feature, -1), np.inf)
            if lower >= upper:
                return ()
        return tuple((feature, direction, bounds[(feature, direction)]) for feature, direction in sorted(bounds))

    def _rit_candidates(self, paths, weights, rng):
        if not paths:
            return {}
        probabilities = weights / weights.sum()
        found = defaultdict(list)

        def draw():
            return paths[int(rng.choice(len(paths), p=probabilities))]

        def grow(rule, depth):
            if len({condition[0] for condition in rule}) >= 2:
                # Retain the RIT rule and its two-condition sub-rules. The
                # latter make repeatable low-order interactions discoverable
                # when full paths vary across bootstrap forests.
                for candidate in (rule, *combinations(rule, 2)):
                    if len({condition[0] for condition in candidate}) < 2:
                        continue
                    key = tuple((feature, direction) for feature, direction, _ in candidate)
                    found[key].append(tuple(threshold for _, _, threshold in candidate))
            if depth >= self.rit_depth:
                return
            for _ in range(2):
                overlap = self._intersect(rule, draw())
                if overlap:
                    grow(overlap, depth + 1)

        for _ in range(self.n_rit_trees):
            grow(self._intersect((), draw()), 1)
        return found

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Expected a 2D X and matching 1D labels")
        self.n_features_in_ = X.shape[1]
        bootstrap_states = np.random.SeedSequence(self.random_state).spawn(self.n_bootstraps)
        by_bootstrap = defaultdict(list)
        importances = []

        for bootstrap_state in bootstrap_states:
            rng = np.random.default_rng(bootstrap_state)
            sample = rng.choice(len(y), size=max(2, int(self.sample_fraction * len(y))), replace=True)
            forest = RandomForestClassifier(
                n_estimators=self.n_trees,
                max_depth=8,
                min_samples_leaf=2,
                random_state=int(rng.integers(0, 2**32 - 1)),
                n_jobs=1,
            )
            forest.fit(X[sample], y[sample])
            importances.append(forest.feature_importances_)
            paths, weights = self._leaf_paths(forest)
            candidates = self._rit_candidates(paths, weights, rng)
            for key, thresholds in candidates.items():
                by_bootstrap[key].append(np.median(np.asarray(thresholds), axis=0))

        self.feature_weights_ = np.mean(importances, axis=0)
        self.candidate_count_ = len(by_bootstrap)
        ranked = []
        for key, threshold_vectors in by_bootstrap.items():
            stability = len(threshold_vectors) / self.n_bootstraps
            if stability >= self.stability_threshold:
                thresholds = tuple(np.median(np.asarray(threshold_vectors), axis=0))
                ranked.append((stability, key, thresholds))
        ranked.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
        self.interactions_ = ranked[: self.n_features_out]
        return self

    def transform(self, X):
        if not hasattr(self, "interactions_"):
            raise AttributeError("Fit interaction features before transforming data")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        output = np.zeros((len(X), self.n_features_out), dtype=np.float32)
        for column, (_, key, thresholds) in enumerate(self.interactions_):
            for (feature, direction), threshold in zip(key, thresholds):
                output[:, column] += (
                    direction * self.feature_weights_[feature] * (X[:, feature] - threshold)
                )
            np.maximum(output[:, column], 0, out=output[:, column])
        return output

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
