import sys
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from experiments.run import main as run_all_main
from experiments.run_tabular import main as run_tabular_main
from experiments.run_tabular_repro import (
    ALL_DATASETS,
    TABULAR_DATASET_ORDER,
    current_model_params,
    split_manifest,
    suggest_model_params,
    tune_hyperparameters,
)
from prepare_data import DATASETS, get_raw_train_test, preprocess_splits


def _args(**overrides):
    values = {
        "n_estimators": 2,
        "n_trees": 50,
        "max_layers": 5,
        "projection_width": 16,
        "projection_width_candidates": [0, 4, 16],
        "interaction_limit": 6,
        "interaction_folds": 3,
        "validation_fraction": 0.2,
        "tuning_validation_fraction": 0.2,
        "n_jobs": 1,
        "tuning_method": "random",
        "tuning_trials": 2,
        "cv_folds": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_entrypoints_default_to_their_dataset_scopes(monkeypatch):
    for entrypoint, expected in (
        (run_tabular_main, TABULAR_DATASET_ORDER),
        (run_all_main, ALL_DATASETS),
    ):
        monkeypatch.setattr(sys, "argv", ["runner", "--phase", "tune"])
        with patch("experiments.run_tabular_repro.run_tuning_phase") as run:
            entrypoint()
        assert run.call_args.args[0].datasets == list(expected)

    assert set(ALL_DATASETS) == set(DATASETS)
    assert set(TABULAR_DATASET_ORDER).isdisjoint({"imdb", "mnist", "cifar10", "orl"})


def test_tabular_entrypoint_rejects_non_tabular_dataset(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "--phase", "tune", "--datasets", "imdb"]
    )
    with patch("experiments.run_tabular_repro.run_tuning_phase"):
        with pytest.raises(SystemExit):
            run_tabular_main()


@pytest.mark.parametrize(
    "mode",
    [
        "bare", "projection", "ifc", "combined", "utility_ifc",
        "random_forest", "extra_trees", "hist_gradient_boosting",
    ],
)
def test_search_space_accepts_current_configuration(mode):
    args = _args()
    current = current_model_params(mode, args)
    suggested = suggest_model_params(optuna.trial.FixedTrial(current), mode, args)

    assert suggested == current


def test_search_space_includes_small_configured_forest_values():
    args = _args(n_estimators=1, n_trees=5, max_layers=1)
    current = current_model_params("bare", args)

    assert suggest_model_params(optuna.trial.FixedTrial(current), "bare", args) == current


@pytest.mark.parametrize("method", ["random", "bayesian"])
def test_hyperparameter_search_runs_and_returns_finite_trial_scores(method):
    X, y = make_classification(
        n_samples=80, n_features=8, n_informative=5, random_state=12
    )
    args = _args(n_estimators=1, n_trees=10, tuning_method=method, tuning_trials=2)

    best_params, best_score, trials, _ = tune_hyperparameters(
        "yeast", "random_forest", pd.DataFrame(X), y, seed=7, args=args
    )

    assert len(trials) == 2
    assert 0 <= best_score <= 1
    assert best_params["n_estimators"] >= 1
    assert all(np.isfinite(trial["value"]) for trial in trials)


def test_raw_train_test_split_is_stratified_and_disjoint():
    X = pd.DataFrame({"value": np.arange(100)})
    y = np.repeat([0, 1], 50)
    with patch("prepare_data.load_dataset", return_value=(X, y)):
        X_train, X_test, y_train, y_test, train_idx, test_idx = get_raw_train_test(
            "synthetic", test_size=0.2, seed=4
        )

    assert len(X_test) == 20
    assert set(train_idx).isdisjoint(test_idx)
    assert np.bincount(y_train).tolist() == [40, 40]
    assert np.bincount(y_test).tolist() == [10, 10]
    with patch("prepare_data.load_dataset", return_value=(X, y)):
        other_split = get_raw_train_test("synthetic", test_size=0.2, seed=5)
    other_manifest = split_manifest(*other_split)
    manifest = split_manifest(X_train, X_test, y_train, y_test, train_idx, test_idx)
    assert other_manifest["dataset_X_sha256"] == manifest["dataset_X_sha256"]
    assert other_manifest["dataset_y_sha256"] == manifest["dataset_y_sha256"]


def test_preprocessing_is_fitted_only_on_each_training_fold():
    X_train = pd.DataFrame({"value": [1.0, 2.0], "category": ["a", "a"]})
    X_validation = pd.DataFrame({"value": [1000.0], "category": ["unseen"]})

    train_features, validation_features = preprocess_splits(
        "yeast", X_train, X_validation
    )

    assert train_features[:, 0].mean() == pytest.approx(0.0)
    assert validation_features[0, 0] == pytest.approx(1997.0)
