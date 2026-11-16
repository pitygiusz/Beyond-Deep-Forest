"""Four-way Deep Forest, projection, IFC, and combined comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

EXPERIMENT_ROOT = Path(__file__).resolve().parent

from experiments.reproducibility import array_sha256, provenance
from .models import DeepForestVariantClassifier


def evaluate(model, X_train, y_train, X_test, y_test):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)
    predict_seconds = perf_counter() - start

    if probability.shape[1] == 2:
        auc = roc_auc_score(y_test, probability[:, 1])
    else:
        auc = roc_auc_score(y_test, probability, multi_class="ovr", average="macro")
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "recall_by_class": recall_score(y_test, predicted, average=None).tolist(),
        "roc_auc": float(auc),
        "log_loss": float(log_loss(y_test, probability)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_layers": int(getattr(model, "n_layers_", 0)),
        "interaction_counts_by_layer": getattr(model, "interaction_counts_by_layer_", []),
        "selected_feedback_by_layer": getattr(model, "selected_feedback_by_layer_", []),
        "depth_validation_scores": getattr(model, "depth_validation_scores_", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("breast_cancer", "digits"), default="breast_cancer")
    parser.add_argument(
        "--variants", nargs="+", choices=DeepForestVariantClassifier.VARIANTS,
        default=list(DeepForestVariantClassifier.VARIANTS),
    )
    parser.add_argument("--baselines", nargs="*", choices=("random_forest", "extra_trees", "hist_gradient_boosting"),
                        default=["hist_gradient_boosting"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-estimators", type=int, default=2, help="RF/ExtraTrees pairs per layer")
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--projection-depth", type=int, default=2)
    parser.add_argument("--negative-slope", type=float, default=0.1)
    parser.add_argument("--interaction-limit", type=int, default=6)
    parser.add_argument("--interaction-bootstraps", type=int, default=3)
    parser.add_argument("--interaction-trees", type=int, default=10)
    parser.add_argument("--rit-trees", type=int, default=20)
    parser.add_argument("--rit-depth", type=int, default=3)
    parser.add_argument("--stability-threshold", type=float, default=0.5)
    parser.add_argument("--interaction-folds", type=int, default=3)
    parser.add_argument("--utility-feature-budget", type=int, default=16)
    parser.add_argument("--utility-cv-folds", type=int, default=3)
    parser.add_argument("--utility-min-gain", type=float, default=0.005)
    parser.add_argument("--utility-cost-penalty", type=float, default=0.001)
    parser.add_argument("--utility-max-groups", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPERIMENT_ROOT / "results" / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.json"

    if args.folds < 2 or args.n_estimators < 1 or args.n_trees < 1 or args.max_layers < 1:
        parser.error("folds must be at least 2; forest counts, trees and layers must be positive")
    if args.projection_width < 0 or args.interaction_folds < 2 or not 0 <= args.validation_fraction < 1:
        parser.error("projection width must be nonnegative, interaction folds >=2, and validation fraction in [0,1)")

    data = load_breast_cancer() if args.dataset == "breast_cancer" else load_digits()
    X, y = data.data, data.target
    records = []
    splits = []
    forest_args = dict(
        n_estimators=args.n_estimators,
        n_trees=args.n_trees,
        max_layers=args.max_layers,
        n_jobs=args.n_jobs,
        verbose=0,
    )
    projection_args = dict(
        projection_width=args.projection_width,
        projection_depth=args.projection_depth,
        negative_slope=args.negative_slope,
    )
    interaction_args = dict(
        interaction_limit=args.interaction_limit,
        interaction_bootstraps=args.interaction_bootstraps,
        interaction_trees=args.interaction_trees,
        rit_trees=args.rit_trees,
        rit_depth=args.rit_depth,
        stability_threshold=args.stability_threshold,
        interaction_folds=args.interaction_folds,
        validation_fraction=args.validation_fraction,
        utility_feature_budget=args.utility_feature_budget,
        utility_cv_folds=args.utility_cv_folds,
        utility_min_gain=args.utility_min_gain,
        utility_cost_penalty=args.utility_cost_penalty,
        utility_max_groups=args.utility_max_groups,
    )
    output_names = {
        "bare": "standard",
        "projection": "random_projection",
        "ifc": "ifc",
        "combined": "ifc_random_projection",
        "utility_ifc": "utility_selected_ifc",
    }
    model_variants = tuple((output_names[variant], variant) for variant in args.variants)
    def baseline(name, seed):
        total_trees = 2 * args.n_estimators * args.n_trees
        if name == "random_forest":
            estimator = RandomForestClassifier(n_estimators=total_trees,
                                               n_jobs=args.n_jobs, random_state=seed)
            grid = {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}
        elif name == "extra_trees":
            estimator = ExtraTreesClassifier(n_estimators=total_trees,
                                             n_jobs=args.n_jobs, random_state=seed)
            grid = {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}
        else:
            estimator = HistGradientBoostingClassifier(max_iter=200,
                                                       learning_rate=0.05, random_state=seed)
            grid = {"max_leaf_nodes": [15, 31], "min_samples_leaf": [10, 30]}
        return GridSearchCV(estimator, grid, scoring="balanced_accuracy",
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            refit=True, n_jobs=1)

    for seed in args.seeds:
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(folds.split(X, y), start=1):
            split_id = f"{seed}:{fold}"
            splits.append({"split_id": split_id, "train_indices": train_index.tolist(),
                           "test_indices": test_index.tolist()})
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            forest_seed = seed * 1000 + fold
            for name, variant in model_variants:
                model = DeepForestVariantClassifier(
                    variant=variant,
                    **forest_args,
                    random_state=forest_seed,
                    **projection_args,
                    **interaction_args,
                )
                metrics = evaluate(model, X_train, y_train, X_test, y_test)
                records.append({"seed": seed, "fold": fold, "split_id": split_id,
                                "model": name, "parameters": model.get_params(), **metrics})
                print(f"seed={seed} fold={fold} {name}: accuracy={metrics['accuracy']:.4f} layers={metrics['n_layers']}")
            for name in args.baselines:
                model = baseline(name, forest_seed)
                metrics = evaluate(model, X_train, y_train, X_test, y_test)
                records.append({"seed": seed, "fold": fold, "split_id": split_id,
                                "model": name, "search_grid": model.param_grid,
                                "best_parameters": model.best_params_,
                                "best_inner_score": float(model.best_score_), **metrics})
                print(f"seed={seed} fold={fold} {name}: accuracy={metrics['accuracy']:.4f}")

    summary = {}
    for name in [*(name for name, _ in model_variants), *args.baselines]:
        selected = [record for record in records if record["model"] == name]
        summary[name] = {
            metric: {"mean": float(np.mean([r[metric] for r in selected])),
                     "std": float(np.std([r[metric] for r in selected]))}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
                           "log_loss", "fit_seconds", "predict_seconds", "n_layers")
        }

    config = {key: value for key, value in vars(args).items() if key != "output"}
    payload = {
        "provenance": provenance(config),
        "data_sha256": {"X": array_sha256(X), "y": array_sha256(y)},
        "n_samples": len(y),
        "n_features": X.shape[1],
        "splits": splits,
        "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
