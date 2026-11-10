"""Run dataset-adapted MLP classifiers on the prepared repository datasets.

Usage (from the repository root):
    .venv\Scripts\python.exe -m experiments.MLP.run_experiment
    .venv\Scripts\python.exe -m experiments.MLP.run_experiment --datasets yeast imdb mnist --seeds 0 1
"""

import argparse
from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

from experiments.MLP.data import get_mlp_train_test
from experiments.MLP.models import dataset_family, hidden_layer_sizes, make_mlp
from prepare_data import DATASETS

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATASET_ORDER = ["orl", "arcene", "yeast", "madelon", "tuandromd", "spambase", "satimage", "musk_v2",
                 "support2", "letter", "bank_marketing", "adult", "kick", "imdb", "mnist"]
DATASET_CHOICES = list(DATASETS)
INPUT_MODES = ("standardized", "raw")

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_one(dataset, seed, input_mode, args):
    X_train, X_test, y_train, y_test = get_mlp_train_test(
        dataset, test_size=args.test_size, seed=seed, input_mode=input_mode)
    model = make_mlp(
        dataset,
        n_layers=args.n_layers,
        random_state=seed,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - start

    return {
        "dataset": dataset,
        "family": dataset_family(dataset),
        "input_mode": input_mode,
        "seed": seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
        "n_layers": args.n_layers,
        "hidden_layer_sizes": str(hidden_layer_sizes(dataset, args.n_layers)),
        "max_iter": args.max_iter,
        "alpha": args.alpha,
        "learning_rate_init": args.learning_rate_init,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": model.n_iter_,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=DATASET_ORDER)
    parser.add_argument("--input-modes", nargs="+", choices=INPUT_MODES, default=list(INPUT_MODES),
                        help="standardized matches the forest preprocessing; raw is an unscaled MLP ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-layers", type=int, default=10,
                        help="hidden-layer count (default matches the forest max_layers setting)")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results" / "mlp_benchmark.csv")
    args = parser.parse_args()
    if not 1 <= args.n_layers <= 10 or args.max_iter < 1 or args.alpha < 0 or args.learning_rate_init <= 0:
        parser.error("n-layers must be 1..10, max-iter positive, alpha non-negative, and learning-rate-init positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        old = pd.read_csv(args.output)
        run_key = ("dataset", "input_mode", "seed", "hidden_layer_sizes", "max_iter", "alpha",
                   "learning_rate_init")
        if not set(run_key).issubset(old.columns):
            parser.error(f"{args.output} has an older result schema; choose a new --output path")
        done = set(old[list(run_key)].itertuples(index=False, name=None))

    for dataset in args.datasets:
        for seed in args.seeds:
            architecture = str(hidden_layer_sizes(dataset, args.n_layers))
            for input_mode in args.input_modes:
                key = (dataset, input_mode, seed, architecture, args.max_iter, args.alpha,
                       args.learning_rate_init)
                if key in done:
                    continue
                row = run_one(dataset, seed, input_mode, args)
                pd.DataFrame([row]).to_csv(args.output, mode="a", index=False,
                                           header=not args.output.exists())
                done.add(key)
                print(f"{dataset:15s} {input_mode:12s} seed={seed}  "
                      f"acc={row['accuracy']:.4f}  bal_acc={row['balanced_accuracy']:.4f}  "
                      f"f1={row['f1']:.4f}  layers={row['n_layers']}  fit={row['fit_seconds']:.1f}s",
                      flush=True)


if __name__ == "__main__":
    main()