"""Random hyperparameter search and test evaluation for every dataset and model.

    uv run -m experiments.run --trials 10 --model-seeds 0 1 2 3 4

Each dataset gets one stratified train/test split. For each model:

1. Score the default settings and trials - 1 random configurations by
   balanced accuracy in stratified CV on the training part. Preprocessing is
   fitted inside each fold; the test part is not used.
2. Fit the defaults and the best configuration on the whole training part,
   once per model seed, and score each fit on the test part.

Results are written to a run folder:
    settings.json  the settings of the run
    search.csv     one row per candidate configuration with its CV score
    test.csv       one row per dataset, model, default/tuned, and model seed,
                   with fit and test-prediction times

Repeat the command with --output <run folder> to resume an interrupted run;
dataset/model pairs that are already in test.csv are skipped.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from experiments.models import MODELS, build_model, candidates
from experiments.prepare_data import DATASETS, TABULAR_DATASETS, get_raw_train_test, preprocess_splits


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def fit_fold(model_name, params, seed, X_fit, y_fit, X_val, y_val):
    """Balanced accuracy and seconds of one CV fit; runs in a worker process."""
    start = perf_counter()
    model = build_model(model_name, params, seed)
    model.fit(X_fit, y_fit)
    return balanced_accuracy_score(y_val, model.predict(X_val)), perf_counter() - start


def score_on_test(model, X_test, y_test):
    start = perf_counter()
    predicted = model.predict(X_test)
    predict_seconds = perf_counter() - start
    probability = model.predict_proba(X_test)
    try:
        if probability.shape[1] == 2:
            auc = roc_auc_score(y_test, probability[:, 1])
        else:
            auc = roc_auc_score(y_test, probability, multi_class="ovr", labels=model.classes_)
    except ValueError:  # a class is missing from the test part
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
            y_test, predicted, average=None, labels=model.classes_, zero_division=0).tolist()),
        "predict_seconds": predict_seconds,
    }


def append_rows(path, rows):
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def finished_pairs(path):
    if not path.exists():
        return set()
    with path.open(newline="") as file:
        return {(row["dataset"], row["model"]) for row in csv.DictReader(file)}


def search(dataset, model_name, X_train, y_train, args):
    """CV score of every candidate; all candidate/fold fits run in args.n_jobs processes.

    Preprocessing is fitted on each training fold; cv_seconds sums the fold fits.
    """
    folds = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.seed)
    fold_data = [
        (*preprocess_splits(dataset, X_train.iloc[fit_idx], X_train.iloc[val_idx]),
         y_train[fit_idx], y_train[val_idx])
        for fit_idx, val_idx in folds.split(X_train, y_train)
    ]
    params_list = candidates(model_name, args.trials, args.seed)
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_fold)(model_name, params, args.seed + fold, X_fit, y_fit, X_val, y_val)
        for params in params_list
        for fold, (X_fit, X_val, y_fit, y_val) in enumerate(fold_data)
    )
    rows = []
    for number, params in enumerate(params_list):
        scores, seconds = zip(*results[number * len(fold_data):(number + 1) * len(fold_data)])
        rows.append({
            "dataset": dataset,
            "model": model_name,
            "candidate": number,  # 0 = default settings
            "cv_balanced_accuracy": float(np.mean(scores)),
            "cv_seconds": sum(seconds),
            "params": json.dumps(params),
        })
    return rows


def evaluate(dataset, model_name, param_set, candidate, X_train, X_test, y_train, y_test, args):
    rows = []
    params = json.loads(candidate["params"])
    for model_seed in args.model_seeds:
        model = build_model(model_name, params, model_seed)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        rows.append({
            "dataset": dataset,
            "model": model_name,
            "params": param_set,
            "model_seed": model_seed,
            "candidate": candidate["candidate"],
            "cv_balanced_accuracy": candidate["cv_balanced_accuracy"],
            **score_on_test(model, X_test, y_test),
            "fit_seconds": fit_seconds,
            "chosen_depth": getattr(model, "chosen_depth_", None),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(TABULAR_DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--trials", type=int, default=10, help="candidates per search, including the defaults")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0, help="seed of the train/test split, CV folds, and sampling")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel processes for the search fits; each model uses one core")
    parser.add_argument("--output", type=Path, help="run folder; reuse one to resume")
    args = parser.parse_args()

    output = args.output or RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items() if key not in ("output", "n_jobs")}
    settings_file = output / "settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        raise SystemExit(f"{output} was created with other settings:\n{settings_file.read_text()}")
    settings_file.write_text(json.dumps(settings, indent=2))

    done = finished_pairs(output / "test.csv")
    for dataset in args.datasets:
        if all((dataset, model_name) in done for model_name in args.models):
            continue
        X_train_raw, X_test_raw, y_train, y_test = get_raw_train_test(dataset, args.test_size, args.seed)
        X_train, X_test = preprocess_splits(dataset, X_train_raw, X_test_raw)

        for model_name in args.models:
            if (dataset, model_name) in done:
                continue
            search_rows = search(dataset, model_name, X_train_raw, y_train, args)
            default = search_rows[0]
            best = max(search_rows, key=lambda row: row["cv_balanced_accuracy"])
            default_rows = evaluate(dataset, model_name, "default", default,
                                    X_train, X_test, y_train, y_test, args)
            if best is default:  # same fits, no need to repeat them
                tuned_rows = [row | {"params": "tuned"} for row in default_rows]
            else:
                tuned_rows = evaluate(dataset, model_name, "tuned", best,
                                      X_train, X_test, y_train, y_test, args)
            test_rows = default_rows + tuned_rows
            append_rows(output / "search.csv", search_rows)
            append_rows(output / "test.csv", test_rows)

            test_score = {param_set: np.mean([row["balanced_accuracy"] for row in test_rows
                                              if row["params"] == param_set])
                          for param_set in ("default", "tuned")}
            print(f"{dataset}/{model_name:30s} "
                  f"cv: default={default['cv_balanced_accuracy']:.4f} best={best['cv_balanced_accuracy']:.4f}  "
                  f"test: default={test_score['default']:.4f} tuned={test_score['tuned']:.4f}", flush=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
