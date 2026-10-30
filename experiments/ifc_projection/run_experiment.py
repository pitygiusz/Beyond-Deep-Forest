"""Four-way Deep Forest, projection, IFC, and combined comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / ".build" / "deep_forest_py310"))

from deepforest import CascadeForestClassifier
from random_projected_deep_forest import RandomProjectedCascadeForestClassifier


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
    interaction_counts = []
    if getattr(model, "use_interactions", False):
        interaction_counts = [
            len(model[layer].interaction_generator_.interactions_)
            for layer in range(model.n_layers_)
        ]
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "roc_auc": float(auc),
        "log_loss": float(log_loss(y_test, probability)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_layers": int(model.n_layers_),
        "interaction_counts_by_layer": interaction_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("breast_cancer", "digits"), default="breast_cancer")
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
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results" / "latest.json")
    args = parser.parse_args()

    if args.folds < 2 or args.n_estimators < 1 or args.n_trees < 1 or args.max_layers < 1:
        parser.error("folds must be at least 2; forest counts, trees and layers must be positive")

    data = load_breast_cancer() if args.dataset == "breast_cancer" else load_digits()
    X, y = data.data, data.target
    records = []
    forest_args = dict(
        n_estimators=args.n_estimators,
        n_trees=args.n_trees,
        max_layers=args.max_layers,
        n_jobs=2,
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
        use_interactions=True,
    )
    model_names = ("standard", "random_projection", "ifc", "ifc_random_projection")

    for seed in args.seeds:
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(folds.split(X, y), start=1):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            forest_seed = seed * 1000 + fold
            models = {
                "standard": CascadeForestClassifier(**forest_args, random_state=forest_seed),
                "random_projection": RandomProjectedCascadeForestClassifier(
                    **forest_args,
                    random_state=forest_seed,
                    **projection_args,
                ),
                "ifc": RandomProjectedCascadeForestClassifier(
                    **forest_args,
                    random_state=forest_seed,
                    **projection_args,
                    **interaction_args,
                    use_projection=False,
                ),
                "ifc_random_projection": RandomProjectedCascadeForestClassifier(
                    **forest_args,
                    random_state=forest_seed,
                    **projection_args,
                    **interaction_args,
                    use_projection=True,
                ),
            }
            for name, model in models.items():
                metrics = evaluate(model, X_train, y_train, X_test, y_test)
                records.append({"seed": seed, "fold": fold, "model": name, **metrics})
                print(f"seed={seed} fold={fold} {name}: accuracy={metrics['accuracy']:.4f} layers={metrics['n_layers']}")

    summary = {}
    for name in model_names:
        selected = [record for record in records if record["model"] == name]
        summary[name] = {
            metric: {"mean": float(np.mean([r[metric] for r in selected])),
                     "std": float(np.std([r[metric] for r in selected]))}
            for metric in ("accuracy", "macro_f1", "roc_auc", "log_loss", "fit_seconds", "predict_seconds", "n_layers")
        }

    payload = {
        "configuration": {key: value for key, value in vars(args).items() if key != "output"},
        "n_samples": len(y),
        "n_features": X.shape[1],
        "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
