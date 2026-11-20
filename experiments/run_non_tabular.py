"""One fit of each Deep Forest variant on the non-tabular datasets.

    uv run -m experiments.run_non_tabular

Every variant uses default_params with 4 RF/ExtraTrees pairs per layer and
500 trees per forest. No search; one fit per dataset and variant.

Splits (stratified, seed --seed):
    mnist  5,000 train / 10,000 test images
    orl    80/20 split (8 / 2 images per person), then 5 of the 8 training
           images per person are drawn at random: 200 train / 80 test
    imdb   5,000 train / 5,000 test reviews

Results are written to <run folder>/test.csv, one row per dataset and variant,
with test metrics, fit time, and number of cascade layers. Repeat the command
with --output <run folder> to resume; finished pairs are skipped.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.models import DEEP_FORESTS, build_model, default_params
from experiments.prepare_data import load_dataset, preprocess_splits
from experiments.run import RESULTS_DIR, append_rows, finished_pairs, score_on_test

N_ESTIMATORS = 4
N_TREES = 500
ORL_TRAIN_PER_PERSON = 5
SPLIT_SIZES = {"mnist": (10_000, 10_000), "imdb": (5_000, 5_000)}


def train_test(dataset, seed):
    X, y = load_dataset(dataset, subset=False)
    if dataset == "orl":
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        rng = np.random.default_rng(seed)
        keep = np.concatenate(
            [
                rng.choice(np.flatnonzero(y_pool == person), ORL_TRAIN_PER_PERSON, replace=False)
                for person in np.unique(y_pool)
            ]
        )
        return X_pool.iloc[keep], X_test, y_pool[keep], y_test
    train_size, test_size = SPLIT_SIZES[dataset]
    return train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "orl", "imdb"],
                        default=["mnist", "orl", "imdb"])
    parser.add_argument("--models", nargs="+", choices=DEEP_FORESTS, default=list(DEEP_FORESTS))
    parser.add_argument("--seed", type=int, default=0, help="seed of the split and the model")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="threads per model fit; more than 1 is slower with deepforest")
    parser.add_argument("--output", type=Path, help="run folder; reuse one to resume")
    args = parser.parse_args()

    output = args.output or RESULTS_DIR / f"non_tabular_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(
        {"datasets": args.datasets, "models": args.models, "seed": args.seed,
         "n_estimators": N_ESTIMATORS, "n_trees": N_TREES}, indent=2))

    done = finished_pairs(output / "test.csv")
    for dataset in args.datasets:
        if all((dataset, model_name) in done for model_name in args.models):
            continue
        X_train_raw, X_test_raw, y_train, y_test = train_test(dataset, args.seed)
        X_train, X_test = preprocess_splits(dataset, X_train_raw, X_test_raw)

        for model_name in args.models:
            if (dataset, model_name) in done:
                continue
            params = default_params(model_name) | {"n_estimators": N_ESTIMATORS, "n_trees": N_TREES}
            model = build_model(model_name, params, args.seed, n_jobs=args.n_jobs)
            start = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - start
            row = {
                "dataset": dataset,
                "model": model_name,
                "seed": args.seed,
                **score_on_test(model, X_test, y_test),
                "fit_seconds": fit_seconds,
                "n_layers": model.n_layers_,
                "chosen_depth": model.chosen_depth_,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": X_train.shape[1],
                "params": json.dumps(params),
            }
            append_rows(output / "test.csv", [row])
            print(f"{dataset}/{model_name:12s} balanced_accuracy={row['balanced_accuracy']:.4f} "
                  f"layers={row['n_layers']} fit={fit_seconds:.0f}s", flush=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
