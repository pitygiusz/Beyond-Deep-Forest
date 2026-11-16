"""Reproducible tabular comparison with nested projection-width selection.

The outer test split is used only for the final score. A separate slice of the
outer training split chooses a projection width; each cascade then uses its
own inner validation split to choose depth. Results and provenance are saved
beside one another, and an incompatible CSV cannot be resumed silently.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from experiments.ifc_projection import DeepForestVariantClassifier
from experiments.reproducibility import array_sha256, provenance
from prepare_data import DATASETS, get_train_test


ROOT = Path(__file__).resolve().parent.parent
DATASET_ORDER = [
    "orl", "arcene", "yeast", "madelon", "tuandromd", "spambase", "satimage",
    "musk_v2", "support2", "letter", "bank_marketing", "adult", "kick",
    "imdb", "mnist",
]
BASELINES = ("random_forest", "extra_trees", "hist_gradient_boosting")
MODES = (*DeepForestVariantClassifier.VARIANTS, *BASELINES)
FIELDS = (
    "run_id", "dataset", "mode", "seed", "n_train", "n_test", "n_features",
    "projection_width", "accuracy", "balanced_accuracy", "f1", "precision",
    "recall", "roc_auc", "log_loss", "recall_by_class", "tuning_seconds",
    "fit_seconds", "predict_seconds", "n_layers", "chosen_depth",
)


def make_model(mode, seed, args, projection_width=None):
    if mode in DeepForestVariantClassifier.VARIANTS:
        return DeepForestVariantClassifier(
            variant=mode, n_estimators=args.n_estimators, n_trees=args.n_trees,
            max_layers=args.max_layers, projection_width=(
                args.projection_width if projection_width is None else projection_width
            ),
            interaction_limit=args.interaction_limit,
            interaction_folds=args.interaction_folds,
            validation_fraction=args.validation_fraction,
            random_state=seed, n_jobs=args.n_jobs,
        )
    total_trees = 2 * args.n_estimators * args.n_trees
    if mode == "random_forest":
        estimator = RandomForestClassifier(n_estimators=total_trees, n_jobs=args.n_jobs,
                                           random_state=seed)
        grid = {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}
    elif mode == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=total_trees, n_jobs=args.n_jobs,
                                         random_state=seed)
        grid = {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}
    elif mode == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                   random_state=seed)
        grid = {"max_leaf_nodes": [15, 31], "min_samples_leaf": [10, 30]}
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return GridSearchCV(
        estimator, grid, scoring="balanced_accuracy", refit=True, n_jobs=1,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
    )


def choose_projection_width(mode, X_train, y_train, seed, args):
    if mode not in ("projection", "combined") or not args.projection_width_candidates:
        return args.projection_width, {}, 0.0
    build_idx, tune_idx = train_test_split(
        np.arange(len(y_train)), test_size=args.width_validation_fraction,
        stratify=y_train, random_state=seed + 10_000,
    )
    started = perf_counter()
    scores = {}
    for width in sorted(set(args.projection_width_candidates)):
        candidate = make_model(mode, seed, args, projection_width=width)
        candidate.fit(X_train[build_idx], y_train[build_idx])
        scores[str(width)] = float(balanced_accuracy_score(
            y_train[tune_idx], candidate.predict(X_train[tune_idx])
        ))
    best_width = max(sorted(set(args.projection_width_candidates)),
                     key=lambda width: (scores[str(width)], -width))
    return best_width, scores, perf_counter() - started


def evaluate(model, X_train, X_test, y_train, y_test):
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)
    predict_seconds = perf_counter() - started
    try:
        auc = (roc_auc_score(y_test, probability[:, 1]) if probability.shape[1] == 2
               else roc_auc_score(y_test, probability, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "precision": precision_score(y_test, predicted, average="macro", zero_division=0),
        "recall": recall_score(y_test, predicted, average="macro", zero_division=0),
        "roc_auc": auc,
        "log_loss": log_loss(y_test, probability, labels=model.classes_),
        "recall_by_class": json.dumps(recall_score(
            y_test, predicted, average=None, labels=model.classes_, zero_division=0
        ).tolist()),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_layers": int(getattr(model, "n_layers_", 0)),
        "chosen_depth": getattr(model, "chosen_depth_", None),
    }


def save_metadata(path, metadata):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASET_ORDER)
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=["bare", "projection", "ifc", "combined", "hist_gradient_boosting"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--projection-width-candidates", nargs="*", type=int,
                        default=[0, 4, 16])
    parser.add_argument("--width-validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--interaction-limit", type=int, default=6)
    parser.add_argument("--interaction-folds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "results" / f"tabular_repro_{datetime.now():%Y%m%d_%H%M%S}.csv"
    if (args.n_estimators < 1 or args.n_trees < 1 or args.max_layers < 1
            or args.projection_width < 0 or any(width < 0 for width in args.projection_width_candidates)
            or args.interaction_folds < 2 or not 0 < args.test_size < 1
            or not 0 <= args.validation_fraction < 1
            or not 0 < args.width_validation_fraction < 1):
        parser.error("invalid forest counts, widths, folds, or split fractions")

    config = {key: value for key, value in vars(args).items() if key != "output"}
    details = provenance(config)
    output = args.output
    sidecar = output.with_suffix(".metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not sidecar.exists():
            raise ValueError(f"Cannot resume {output}: provenance sidecar is missing")
        metadata = json.loads(sidecar.read_text())
        if metadata["run_id"] != details["run_id"]:
            raise ValueError("Cannot resume a result file created with different code or settings")
        with output.open(newline="") as stream:
            done = {(row["dataset"], row["mode"], int(row["seed"]))
                    for row in csv.DictReader(stream)}
    else:
        metadata = {**details, "splits": {}, "model_runs": {}}
        done = set()
        save_metadata(sidecar, metadata)

    for dataset in args.datasets:
        for seed in args.seeds:
            key = f"{dataset}:{seed}"
            X_train, X_test, y_train, y_test, train_idx, test_idx = get_train_test(
                dataset, test_size=args.test_size, seed=seed, return_indices=True
            )
            split = {
                "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
                "X_train_sha256": array_sha256(X_train),
                "X_test_sha256": array_sha256(X_test),
                "y_train_sha256": array_sha256(y_train),
                "y_test_sha256": array_sha256(y_test),
            }
            if key in metadata["splits"] and metadata["splits"][key] != split:
                raise ValueError(f"Prepared data or split changed for {key}; refusing to mix results")
            metadata["splits"][key] = split
            save_metadata(sidecar, metadata)
            for mode in args.modes:
                if (dataset, mode, seed) in done:
                    continue
                width, tuning_scores, tuning_seconds = choose_projection_width(
                    mode, X_train, y_train, seed, args
                )
                model = make_model(mode, seed, args, projection_width=width)
                run_key = f"{dataset}:{seed}:{mode}"
                metrics = evaluate(model, X_train, X_test, y_train, y_test)
                row = {
                    "run_id": details["run_id"], "dataset": dataset, "mode": mode,
                    "seed": seed, "n_train": len(y_train), "n_test": len(y_test),
                    "n_features": X_train.shape[1],
                    "projection_width": width if mode in ("projection", "combined") else "",
                    "tuning_seconds": tuning_seconds, **metrics,
                }
                model_details = (
                    {"search_grid": model.param_grid, "best_parameters": model.best_params_,
                     "best_inner_score": float(model.best_score_)}
                    if mode in BASELINES else {"parameters": model.get_params(deep=False)}
                )
                metadata["model_runs"][run_key] = {
                    **model_details,
                    "projection_width_scores": tuning_scores,
                    "depth_validation_scores": getattr(model, "depth_validation_scores_", []),
                }
                save_metadata(sidecar, metadata)
                with output.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                done.add((dataset, mode, seed))
                print(f"{dataset:15s} {mode:23s} seed={seed} "
                      f"bal_acc={metrics['balanced_accuracy']:.4f} "
                      f"width={row['projection_width']} layers={metrics['n_layers']}", flush=True)
    print(f"Results: {output}\nProvenance: {sidecar}")


if __name__ == "__main__":
    main()
