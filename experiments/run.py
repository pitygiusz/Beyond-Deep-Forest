"""Run the four Deep Forest variants on the tabular datasets.

One stratified 70/30 train/test split per seed. Results are appended to a CSV
(one row per dataset x mode x seed); rows already in the CSV are skipped, so an
interrupted run can be resumed by running the same command again.

Usage (from the repository root):
    uv run -m experiments.run
    uv run -m experiments.run --datasets spambase letter --modes bare ifc --seeds 0 1
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from experiments.ifc_projection import DeepForestVariantClassifier
from prepare_data import DATASETS, get_train_test

ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", category=DataDimensionalityWarning)

DATASET_ORDER = ["orl", "arcene", "yeast", "madelon", "tuandromd", "spambase", "satimage", "musk_v2",
                 "support2", "letter", "bank_marketing", "adult", "kick", "imdb", "mnist"]
MODES = list(DeepForestVariantClassifier.VARIANTS)  # bare, projection, ifc, combined


def run_one(dataset, mode, seed, args):
    X_train, X_test, y_train, y_test = get_train_test(dataset, test_size=args.test_size, seed=seed)
    model = DeepForestVariantClassifier(
        variant=mode, n_estimators=args.n_estimators, n_trees=args.n_trees,
        max_layers=args.max_layers, random_state=seed, n_jobs=args.n_jobs,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - start

    return {
        "dataset": dataset, "mode": mode, "seed": seed,
        "n_train": len(y_train), "n_test": len(y_test), "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "n_layers": model.n_layers_,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=2, help="RF/ExtraTrees pairs per layer")
    parser.add_argument("--n-trees", type=int, default=50, help="trees per forest")
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel workers per forest (-1 = all cores; has a large startup cost)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: results/tabular_<timestamp>.csv)")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "results" / f"tabular_{datetime.now():%Y%m%d_%H%M%S}.csv"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        old = pd.read_csv(args.output)
        done = set(zip(old["dataset"], old["mode"], old["seed"]))

    for dataset in args.datasets:
        for seed in args.seeds:
            for mode in args.modes:
                if (dataset, mode, seed) in done:
                    continue
                row = run_one(dataset, mode, seed, args)
                pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                           header=not args.output.exists())
                print(f"{dataset:15s} {mode:10s} seed={seed}  acc={row['accuracy']:.4f}  "
                      f"bal_acc={row['balanced_accuracy']:.4f}  f1={row['f1']:.4f}  "
                      f"layers={row['n_layers']}  fit={row['fit_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    assert set(DATASET_ORDER) <= set(DATASETS), "DATASET_ORDER has a name missing from prepare_data.DATASETS"
    main()
