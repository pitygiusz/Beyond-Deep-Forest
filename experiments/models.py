"""The compared models, their default settings, and their random-search spaces."""

from scipy.stats import loguniform, uniform
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler

from deep_forests import DeepForestVariantClassifier


DEEP_FORESTS = DeepForestVariantClassifier.VARIANTS
BASELINES = ("random_forest", "extra_trees", "hist_gradient_boosting")
MODELS = DEEP_FORESTS + BASELINES

USES_PROJECTION = ("projection", "combined")
USES_INTERACTIONS = ("ifc", "combined", "utility_ifc")


def build_model(name, params, seed, n_jobs=1):
    if name in DEEP_FORESTS:
        return DeepForestVariantClassifier(variant=name, **params, random_state=seed, n_jobs=n_jobs)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if name == "extra_trees":
        return ExtraTreesClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def default_params(name):
    if name in ("random_forest", "extra_trees"):
        return {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None}
    if name == "hist_gradient_boosting":
        return {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
                "min_samples_leaf": 20, "l2_regularization": 0.0}

    params = {"n_estimators": 2, "n_trees": 50, "max_layers": 5, "max_depth": None, "min_samples_leaf": 1}
    if name in USES_PROJECTION:
        params.update(projection_width=16, projection_depth=2, negative_slope=0.1)
    if name in USES_INTERACTIONS:
        params.update(interaction_limit=6, interaction_bootstraps=3, interaction_trees=10,
                      rit_trees=20, rit_depth=3, stability_threshold=0.5, interaction_folds=3)
    if name == "utility_ifc":
        params.update(utility_feature_budget=16, utility_cv_folds=3, utility_min_gain=0.005,
                      utility_cost_penalty=0.001, utility_max_groups=4)
    return params


def search_space(name):
    """Values to sample from. Parameters not listed stay at default_params(name)."""
    if name in ("random_forest", "extra_trees"):
        return {"max_features": ["sqrt", "log2", 0.5, 0.75], "min_samples_leaf": list(range(1, 11))}
    if name == "hist_gradient_boosting":
        return {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": list(range(7, 64))}

    space = {"max_depth": [None, 10, 20, 40], "min_samples_leaf": list(range(1, 6))}
    if name in USES_PROJECTION:
        space["projection_width"] = [0, 4, 16]
    if name in USES_INTERACTIONS:
        space["interaction_limit"] = list(range(1, 13))
    if name in ("ifc", "utility_ifc"):
        space["stability_threshold"] = uniform(0.2, 0.6)  # 0.2 to 0.8
    return space


def candidates(name, n, seed):
    """The default settings followed by up to n - 1 random configurations.

    Fewer are returned when the space is small: sampled copies of the
    defaults are dropped, and a space of lists only is sampled without repeats.
    """
    defaults = default_params(name)
    sampled = ParameterSampler(search_space(name), n - 1, random_state=seed) if n > 1 else []
    others = [defaults | params for params in sampled]
    return [defaults] + [params for params in others if params != defaults]
