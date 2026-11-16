"""Two-phase tabular tuning and final evaluation.

Tune one model per dataset using stratified cross-validation on the training
split, then evaluate tuned and default models against the same held-out test
split. Preprocessing is fitted within each CV fold.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from experiments.ifc_projection import DeepForestVariantClassifier
from experiments.reproducibility import array_sha256, provenance
from prepare_data import DATASETS, TABULAR_DATASETS, get_raw_train_test, preprocess_splits


ROOT = Path(__file__).resolve().parent.parent
ALL_DATASETS = tuple(DATASETS)
TABULAR_DATASET_ORDER = tuple(
    dataset for dataset in ALL_DATASETS if dataset in TABULAR_DATASETS
)
MODES = (*DeepForestVariantClassifier.VARIANTS,
         "random_forest", "extra_trees", "hist_gradient_boosting")
FIELDS = (
    "run_id", "dataset", "mode", "parameter_set", "seed", "n_train", "n_test", "n_features",
    "projection_width", "accuracy", "balanced_accuracy", "f1", "precision",
    "recall", "roc_auc", "log_loss", "recall_by_class", "tuning_seconds",
    "fit_seconds", "predict_seconds", "n_layers", "chosen_depth",
)


def make_model(mode, seed, args, model_params=None):
    model_params = model_params or {}
    if mode in DeepForestVariantClassifier.VARIANTS:
        parameters = {
            "variant": mode,
            "n_estimators": args.n_estimators,
            "n_trees": args.n_trees,
            "max_layers": args.max_layers,
            "projection_width": args.projection_width,
            "interaction_limit": args.interaction_limit,
            "interaction_folds": args.interaction_folds,
            "validation_fraction": args.validation_fraction,
            "random_state": seed,
            "n_jobs": args.n_jobs,
        }
        parameters.update(model_params)
        return DeepForestVariantClassifier(**parameters)
    total_trees = 2 * args.n_estimators * args.n_trees
    if mode == "random_forest":
        parameters = {"n_estimators": total_trees, "n_jobs": args.n_jobs,
                      "random_state": seed}
        parameters.update(model_params)
        return RandomForestClassifier(**parameters)
    elif mode == "extra_trees":
        parameters = {"n_estimators": total_trees, "n_jobs": args.n_jobs,
                      "random_state": seed}
        parameters.update(model_params)
        return ExtraTreesClassifier(**parameters)
    elif mode == "hist_gradient_boosting":
        parameters = {"max_iter": 200, "learning_rate": 0.05, "random_state": seed}
        parameters.update(model_params)
        return HistGradientBoostingClassifier(**parameters)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def current_model_params(mode, args):
    if mode in DeepForestVariantClassifier.VARIANTS:
        parameters = {
            "n_estimators": args.n_estimators,
            "n_trees": args.n_trees,
            "max_layers": args.max_layers,
            "max_depth": None,
            "min_samples_leaf": 1,
        }
        if mode in ("projection", "combined"):
            parameters.update({
                "projection_width": args.projection_width,
                "projection_depth": 2,
                "negative_slope": 0.1,
            })
        if mode in ("ifc", "combined", "utility_ifc"):
            parameters.update({
                "interaction_limit": args.interaction_limit,
                "interaction_bootstraps": 3,
                "interaction_trees": 10,
                "rit_trees": 20,
                "rit_depth": 3,
                "stability_threshold": 0.5,
                "interaction_folds": args.interaction_folds,
            })
        if mode == "utility_ifc":
            parameters.update({
                "utility_feature_budget": 16,
                "utility_cv_folds": 3,
                "utility_min_gain": 0.005,
                "utility_cost_penalty": 0.001,
                "utility_max_groups": 4,
            })
        return parameters
    if mode in ("random_forest", "extra_trees"):
        return {
            "n_estimators": 2 * args.n_estimators * args.n_trees,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
        }
    if mode == "hist_gradient_boosting":
        return {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        }
    raise ValueError(f"Unknown mode: {mode}")


def suggest_model_params(trial, mode, args):
    if mode in DeepForestVariantClassifier.VARIANTS:
        parameters = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 1, max(4, 2 * args.n_estimators)
            ),
            "n_trees": trial.suggest_int(
                "n_trees", min(10, args.n_trees), max(200, 2 * args.n_trees)
            ),
            "max_layers": trial.suggest_int("max_layers", 1, max(10, 2 * args.max_layers)),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        if mode in ("projection", "combined"):
            widths = sorted(set(args.projection_width_candidates) | {args.projection_width})
            parameters.update({
                "projection_width": trial.suggest_categorical("projection_width", widths),
                "projection_depth": trial.suggest_int("projection_depth", 1, 4),
                "negative_slope": trial.suggest_float(
                    "negative_slope", 0.01, 0.5, log=True
                ),
            })
        if mode in ("ifc", "combined", "utility_ifc"):
            parameters.update({
                "interaction_limit": trial.suggest_int(
                    "interaction_limit", 1, max(12, 2 * args.interaction_limit)
                ),
                "interaction_bootstraps": trial.suggest_int(
                    "interaction_bootstraps", 1, 6
                ),
                "interaction_trees": trial.suggest_int("interaction_trees", 1, 40),
                "rit_trees": trial.suggest_int("rit_trees", 1, 60),
                "rit_depth": trial.suggest_int("rit_depth", 1, 6),
                "stability_threshold": trial.suggest_float(
                    "stability_threshold", 0.2, 0.8
                ),
                "interaction_folds": trial.suggest_int(
                    "interaction_folds", 2, max(5, 2 * args.interaction_folds)
                ),
            })
        if mode == "utility_ifc":
            parameters.update({
                "utility_feature_budget": trial.suggest_int(
                    "utility_feature_budget", 1, 32
                ),
                "utility_cv_folds": trial.suggest_int("utility_cv_folds", 2, 5),
                "utility_min_gain": trial.suggest_float(
                    "utility_min_gain", 0.0001, 0.05, log=True
                ),
                "utility_cost_penalty": trial.suggest_float(
                    "utility_cost_penalty", 0.0001, 0.05, log=True
                ),
                "utility_max_groups": trial.suggest_int("utility_max_groups", 1, 8),
            })
        return parameters
    if mode in ("random_forest", "extra_trees"):
        current_trees = 2 * args.n_estimators * args.n_trees
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", max(1, current_trees // 2),
                max(50, 2 * current_trees),
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.75]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
        }
    if mode == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_categorical(
                "l2_regularization", [0.0, 0.01, 0.1, 1.0, 10.0]
            ),
        }
    raise ValueError(f"Unknown mode: {mode}")


def tune_hyperparameters(dataset, mode, X_train, y_train, seed, args):
    sampler = (
        RandomSampler(seed=seed)
        if args.tuning_method == "random"
        else TPESampler(seed=seed, n_startup_trials=min(5, args.tuning_trials))
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(current_model_params(mode, args))
    folds = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=seed
    )

    def objective(trial):
        parameters = suggest_model_params(trial, mode, args)
        scores = []
        for fold, (fit_idx, validation_idx) in enumerate(folds.split(X_train, y_train)):
            X_fit, X_validation = preprocess_splits(
                dataset, X_train.iloc[fit_idx], X_train.iloc[validation_idx]
            )
            candidate = make_model(mode, seed + fold, args, parameters)
            candidate.fit(X_fit, y_train[fit_idx])
            scores.append(balanced_accuracy_score(
                y_train[validation_idx], candidate.predict(X_validation)
            ))
        return float(np.mean(scores))

    started = perf_counter()
    study.optimize(objective, n_trials=args.tuning_trials, n_jobs=1)
    trials = [
        {"number": trial.number, "value": trial.value, "params": trial.params}
        for trial in study.trials
    ]
    return study.best_params, float(study.best_value), trials, perf_counter() - started


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
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def dataframe_sha256(frame):
    return array_sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy())


def split_manifest(X_train, X_test, y_train, y_test, train_idx, test_idx):
    X_all = pd.concat(
        (X_train.reset_index(drop=True), X_test.reset_index(drop=True)),
        ignore_index=True,
    )
    X_all = X_all.iloc[np.argsort(np.concatenate((train_idx, test_idx)))].reset_index(drop=True)
    y_all = np.empty(len(y_train) + len(y_test), dtype=np.asarray(y_train).dtype)
    y_all[train_idx] = y_train
    y_all[test_idx] = y_test
    return {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "dataset_X_sha256": dataframe_sha256(X_all),
        "dataset_y_sha256": array_sha256(y_all),
        "X_train_sha256": dataframe_sha256(X_train),
        "X_test_sha256": dataframe_sha256(X_test),
        "y_train_sha256": array_sha256(y_train),
        "y_test_sha256": array_sha256(y_test),
    }


def model_config(args):
    return {
        "n_estimators": args.n_estimators,
        "n_trees": args.n_trees,
        "max_layers": args.max_layers,
        "n_jobs": args.n_jobs,
        "projection_width": args.projection_width,
        "projection_width_candidates": args.projection_width_candidates,
        "interaction_limit": args.interaction_limit,
        "interaction_folds": args.interaction_folds,
        "validation_fraction": args.validation_fraction,
    }


def tuning_config(args):
    return {
        "datasets": args.datasets,
        "modes": args.modes,
        "test_size": args.test_size,
        "tuning_seed": args.tuning_seed,
        "cv_folds": args.cv_folds,
        "tuning_method": args.tuning_method,
        "tuning_trials": args.tuning_trials,
        "model": model_config(args),
    }


def open_tuning_artifact(path, args):
    config = tuning_config(args)
    if path.exists():
        artifact = json.loads(path.read_text())
        if artifact.get("config") != config:
            raise ValueError(f"Tuning file {path} was created with different settings")
        expected = provenance({"phase": "tune", "config": config})
        if artifact.get("run_id") != expected["run_id"]:
            raise ValueError(f"Tuning file {path} has incompatible code or dependencies")
        return artifact

    details = provenance({"phase": "tune", "config": config})
    artifact = {**details, "config": config, "splits": {}, "models": {}}
    save_metadata(path, artifact)
    return artifact


def run_tuning_phase(args):
    path = args.tuning_file
    if path is None:
        path = ROOT / "results" / (
            f"tabular_tuning_{args.tuning_method}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
    artifact = open_tuning_artifact(path, args)

    for dataset_index, dataset in enumerate(args.datasets):
        X_train, X_test, y_train, y_test, train_idx, test_idx = get_raw_train_test(
            dataset, test_size=args.test_size, seed=args.tuning_seed
        )
        split = split_manifest(
            X_train, X_test, y_train, y_test, train_idx, test_idx
        )
        previous_split = artifact["splits"].get(dataset)
        if previous_split is not None and previous_split != split:
            raise ValueError(f"Dataset or split changed for {dataset}; refusing to resume")
        artifact["splits"][dataset] = split
        save_metadata(path, artifact)

        study_seed = args.tuning_seed + dataset_index * 1000
        for mode in args.modes:
            key = f"{dataset}:{mode}"
            if key in artifact["models"]:
                continue
            best_params, best_score, trials, tuning_seconds = tune_hyperparameters(
                dataset, mode, X_train, y_train, study_seed, args
            )
            artifact["models"][key] = {
                "dataset": dataset,
                "mode": mode,
                "seed": study_seed,
                "best_parameters": best_params,
                "best_cv_balanced_accuracy": best_score,
                "tuning_seconds": tuning_seconds,
                "trials": trials,
            }
            save_metadata(path, artifact)
            print(f"{dataset:15s} {mode:23s} "
                  f"cv_bal_acc={best_score:.4f} trials={len(trials)}", flush=True)

    print(f"Tuning results: {path}")


def load_compatible_tuning(path, args):
    if path is None or not path.is_file():
        raise ValueError("Final phase requires an existing --tuning-file")
    artifact = json.loads(path.read_text())
    config = artifact.get("config", {})
    expected = provenance({"phase": "tune", "config": config})
    if artifact.get("run_id") != expected["run_id"]:
        raise ValueError("Tuning file was created with incompatible code or dependencies")
    if config.get("model") != model_config(args):
        raise ValueError("Tuning file uses different model defaults or validation settings")
    if config.get("test_size") != args.test_size:
        raise ValueError("Tuning and final phases must use the same test split fraction")
    if config.get("tuning_seed") != args.tuning_seed:
        raise ValueError("Tuning and final phases must use the same tuning seed")
    if not set(args.datasets).issubset(config.get("datasets", [])):
        raise ValueError("Tuning file does not contain every requested dataset")
    if not set(args.modes).issubset(config.get("modes", [])):
        raise ValueError("Tuning file does not contain every requested model")
    for dataset in args.datasets:
        for mode in args.modes:
            if f"{dataset}:{mode}" not in artifact.get("models", {}):
                raise ValueError(f"Tuning file has no result for {dataset}:{mode}")
    return artifact


def run_final_phase(args):
    tuning = load_compatible_tuning(args.tuning_file, args)
    output = args.output or ROOT / "results" / (
        f"tabular_final_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    sidecar = output.with_suffix(".metadata.json")
    final_config = {
        "phase": "final",
        "tuning_run_id": tuning["run_id"],
        "datasets": args.datasets,
        "modes": args.modes,
        "seeds": args.seeds,
        "test_size": args.test_size,
        "tuning_seed": args.tuning_seed,
        "model": model_config(args),
    }
    details = provenance(final_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not sidecar.exists():
            raise ValueError(f"Cannot resume {output}: provenance sidecar is missing")
        metadata = json.loads(sidecar.read_text())
        if metadata["run_id"] != details["run_id"]:
            raise ValueError("Cannot resume a final CSV created with different settings")
        with output.open(newline="") as stream:
            done = {
                (row["dataset"], row["mode"], row["parameter_set"], int(row["seed"]))
                for row in csv.DictReader(stream)
            }
    else:
        metadata = {**details, "splits": {}, "model_runs": {}}
        done = set()
        save_metadata(sidecar, metadata)

    for dataset in args.datasets:
        for seed in args.seeds:
            X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = (
                get_raw_train_test(dataset, test_size=args.test_size, seed=seed)
            )
            split = split_manifest(
                X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx
            )
            tuned_split = tuning["splits"][dataset]
            if (split["dataset_X_sha256"] != tuned_split["dataset_X_sha256"]
                    or split["dataset_y_sha256"] != tuned_split["dataset_y_sha256"]):
                raise ValueError(f"Prepared data changed for {dataset}")
            if seed == args.tuning_seed and split != tuned_split:
                raise ValueError(f"Tuning and final splits differ for {dataset}")
            split_key = f"{dataset}:{seed}"
            if split_key in metadata["splits"] and metadata["splits"][split_key] != split:
                raise ValueError(f"Prepared data or split changed for {split_key}")
            metadata["splits"][split_key] = split
            X_train, X_test = preprocess_splits(dataset, X_train_raw, X_test_raw)
            save_metadata(sidecar, metadata)

            for mode in args.modes:
                tuned_run = tuning["models"][f"{dataset}:{mode}"]
                parameter_sets = (
                    ("default", {}),
                    ("tuned", tuned_run["best_parameters"]),
                )
                for parameter_set, parameters in parameter_sets:
                    run_key = (dataset, mode, parameter_set, seed)
                    if run_key in done:
                        continue
                    model = make_model(mode, seed, args, parameters)
                    metrics = evaluate(model, X_train, X_test, y_train, y_test)
                    width = parameters.get("projection_width", args.projection_width)
                    row = {
                        "run_id": details["run_id"],
                        "dataset": dataset,
                        "mode": mode,
                        "parameter_set": parameter_set,
                        "seed": seed,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "n_features": X_train.shape[1],
                        "projection_width": (
                            width if mode in ("projection", "combined") else ""
                        ),
                        "tuning_seconds": (
                            tuned_run["tuning_seconds"] if parameter_set == "tuned" else 0
                        ),
                        **metrics,
                    }
                    metadata["model_runs"][":".join(map(str, run_key))] = {
                        "parameters": model.get_params(deep=False),
                        "best_cv_balanced_accuracy": (
                            tuned_run["best_cv_balanced_accuracy"]
                            if parameter_set == "tuned" else None
                        ),
                        "chosen_depth": metrics["chosen_depth"],
                    }
                    save_metadata(sidecar, metadata)
                    with output.open("a", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS)
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                    done.add(run_key)
                    print(f"{dataset:15s} {mode:23s} {parameter_set:7s} "
                          f"seed={seed} bal_acc={metrics['balanced_accuracy']:.4f}",
                          flush=True)

    print(f"Final results: {output}\nProvenance: {sidecar}")


def main(dataset_scope=ALL_DATASETS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "final"), required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=dataset_scope, default=list(dataset_scope)
    )
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tuning-seed", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--tuning-method", choices=("random", "bayesian"),
                        default="bayesian")
    parser.add_argument("--tuning-trials", type=int, default=20)
    parser.add_argument("--tuning-file", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--projection-width", type=int, default=16)
    parser.add_argument("--projection-width-candidates", nargs="*", type=int,
                        default=[0, 4, 16])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--interaction-limit", type=int, default=6)
    parser.add_argument("--interaction-folds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if (args.n_estimators < 1 or args.n_trees < 1 or args.max_layers < 1
            or args.projection_width < 0 or any(width < 0 for width in args.projection_width_candidates)
            or args.interaction_limit < 1 or args.interaction_folds < 2
            or args.cv_folds < 2 or not 0 < args.test_size < 1
            or args.tuning_trials < 1
            or not 0 <= args.validation_fraction < 1):
        parser.error("invalid forest counts, widths, folds, or split fractions")

    if args.phase == "tune":
        run_tuning_phase(args)
    else:
        run_final_phase(args)


if __name__ == "__main__":
    main()
