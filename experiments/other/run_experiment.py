"""
Comprehensive Experiment: Benchmarking Original Deep Forest vs. Outlier-Pruned Deep Forest.

This script executes a systematic evaluation comparing:
1. Original Deep Forest (gcForest): All M forests output class probability vectors concatenated
   with raw features into the augmented vector.
2. Outlier-Pruned Deep Forest: At each cascade layer, the 2 forests whose probability predictions
   diverge the most from the ensemble consensus are identified and pruned from the augmented vector.

Evaluations:
- Primary Dataset: Breast Cancer Wisconsin (569 samples, 30 features, binary)
- Secondary Dataset: Digits (1797 samples, 64 features, 10 classes) - tests multi-class vector pruning
- Configs tested:
  * 8 base forests per layer (4 RF + 4 ERF), pruning 2 outliers -> 6 kept (n_estimators=4)
  * 4 base forests per layer (2 RF + 2 ERF), pruning 2 outliers -> 2 kept (n_estimators=2, matching deep_forest_architecture_v2.pdf)
- Validation: Stratified 5-Fold Cross-Validation with multiple metrics:
  * Accuracy, Balanced Accuracy, Macro F1, ROC-AUC, Log-Loss
  * Fit time, Inference time, Optimal cascade layers
  * Detailed layer-by-layer outlier diagnostics
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from deepforest import CascadeForestClassifier
from outlier_pruned_deep_forest import OutlierPrunedCascadeForestClassifier


def evaluate_fold(model, X_train, y_train, X_test, y_test, is_multiclass=False):
    """Fit model and return all evaluation metrics."""
    t_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t_start

    t_infer_start = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    infer_time = time.time() - t_infer_start

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC calculation
    try:
        if is_multiclass:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1])
    except Exception:
        auc = np.nan

    # Log-Loss calculation
    try:
        loss = log_loss(y_test, y_proba)
    except Exception:
        loss = np.nan

    n_layers = getattr(model, "n_layers_", 1)

    # Pruning diagnostics if available
    pruning_info = None
    if hasattr(model, "get_pruning_report"):
        try:
            pruning_info = model.get_pruning_report()
        except Exception:
            pruning_info = None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "log_loss": loss,
        "fit_time": fit_time,
        "infer_time": infer_time,
        "n_layers": n_layers,
        "pruning_report": pruning_info,
    }


def run_cv_experiment(dataset_name="breast_cancer", n_estimators=4, n_trees=50, n_splits=5, random_state=42):
    """Run 5-Fold Cross Validation comparing Original vs Outlier-Pruned Deep Forest."""
    print("=" * 80)
    print(f"Running Experiment on Dataset: {dataset_name.upper()} | n_estimators={n_estimators} ({n_estimators*2} total forests) | n_trees={n_trees}")
    print("=" * 80)

    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        is_multiclass = False
    elif dataset_name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        is_multiclass = True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {
        "original": [],
        "outlier_pruned": [],
    }
    all_pruning_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # 1. Original Deep Forest
        print("Training Original Deep Forest...")
        orig_model = CascadeForestClassifier(
            n_estimators=n_estimators,
            n_trees=n_trees,
            random_state=random_state + fold,
            verbose=0,
            n_jobs=2,
        )
        orig_metrics = evaluate_fold(orig_model, X_train, y_train, X_test, y_test, is_multiclass)
        results["original"].append(orig_metrics)
        print(f"  [Original] Acc: {orig_metrics['accuracy']:.4f} | F1: {orig_metrics['macro_f1']:.4f} | Layers: {orig_metrics['n_layers']} | Time: {orig_metrics['fit_time']:.2f}s")

        # 2. Outlier-Pruned Deep Forest (removes 2 outlier forests from augmented vector)
        print("Training Outlier-Pruned Deep Forest (removing 2 outlier forests)...")
        pruned_model = OutlierPrunedCascadeForestClassifier(
            n_estimators=n_estimators,
            n_trees=n_trees,
            n_outliers_removed=2,
            divergence_metric="euclidean",
            random_state=random_state + fold,
            verbose=0,
            n_jobs=2,
        )
        pruned_metrics = evaluate_fold(pruned_model, X_train, y_train, X_test, y_test, is_multiclass)
        results["outlier_pruned"].append(pruned_metrics)
        all_pruning_reports.append(pruned_metrics["pruning_report"])
        print(f"  [Outlier-Pruned] Acc: {pruned_metrics['accuracy']:.4f} | F1: {pruned_metrics['macro_f1']:.4f} | Layers: {pruned_metrics['n_layers']} | Time: {pruned_metrics['fit_time']:.2f}s")

    # Aggregate results into DataFrame
    summary = {}
    for model_key in ["original", "outlier_pruned"]:
        metrics_list = results[model_key]
        summary[model_key] = {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in metrics_list])),
            "accuracy_std": float(np.std([m["accuracy"] for m in metrics_list])),
            "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in metrics_list])),
            "balanced_accuracy_std": float(np.std([m["balanced_accuracy"] for m in metrics_list])),
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics_list])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics_list])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in metrics_list])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in metrics_list])),
            "log_loss_mean": float(np.mean([m["log_loss"] for m in metrics_list])),
            "log_loss_std": float(np.std([m["log_loss"] for m in metrics_list])),
            "fit_time_mean": float(np.mean([m["fit_time"] for m in metrics_list])),
            "fit_time_std": float(np.std([m["fit_time"] for m in metrics_list])),
            "infer_time_mean": float(np.mean([m["infer_time"] for m in metrics_list])),
            "infer_time_std": float(np.std([m["infer_time"] for m in metrics_list])),
            "n_layers_mean": float(np.mean([m["n_layers"] for m in metrics_list])),
            "raw_folds": [
                {k: v for k, v in m.items() if k != "pruning_report"}
                for m in metrics_list
            ],
        }

    # Statistical Significance: Paired t-test and Wilcoxon signed-rank test
    orig_accs = [m["accuracy"] for m in results["original"]]
    pruned_accs = [m["accuracy"] for m in results["outlier_pruned"]]
    orig_losses = [m["log_loss"] for m in results["original"]]
    pruned_losses = [m["log_loss"] for m in results["outlier_pruned"]]

    try:
        t_stat, p_val_ttest = stats.ttest_rel(pruned_accs, orig_accs)
    except Exception:
        t_stat, p_val_ttest = np.nan, np.nan

    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(pruned_accs, orig_accs)
    except Exception:
        w_stat, p_val_wilcoxon = np.nan, np.nan

    stats_summary = {
        "ttest_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "ttest_pval": float(p_val_ttest) if not np.isnan(p_val_ttest) else None,
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pval": float(p_val_wilcoxon) if not np.isnan(p_val_wilcoxon) else None,
    }

    experiment_output = {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "n_estimators": n_estimators,
        "total_forests_per_layer": 2 * n_estimators,
        "n_outliers_removed": 2,
        "kept_forests_per_layer": 2 * n_estimators - 2,
        "n_trees_per_forest": n_trees,
        "n_splits": n_splits,
        "summary": summary,
        "statistical_tests": stats_summary,
        "sample_pruning_report": all_pruning_reports[0],
    }

    return experiment_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="experiment_results.json")
    args = parser.parse_args()

    all_experiments = []

    # Experiment 1: Breast Cancer Wisconsin with 8 forests per layer (4 RF + 4 ERF), 2 outliers removed -> 6 kept
    exp1 = run_cv_experiment(
        dataset_name="breast_cancer",
        n_estimators=4,
        n_trees=args.n_trees,
        n_splits=5,
        random_state=42,
    )
    all_experiments.append(exp1)

    # Experiment 2: Breast Cancer Wisconsin with 4 forests per layer (2 RF + 2 ERF, as in PDF diagram), 2 outliers removed -> 2 kept
    exp2 = run_cv_experiment(
        dataset_name="breast_cancer",
        n_estimators=2,
        n_trees=args.n_trees,
        n_splits=5,
        random_state=42,
    )
    all_experiments.append(exp2)

    # Experiment 3: Digits Dataset (10 classes, multi-class class-vector augmentation) with 8 forests per layer
    exp3 = run_cv_experiment(
        dataset_name="digits",
        n_estimators=4,
        n_trees=args.n_trees,
        n_splits=5,
        random_state=42,
    )
    all_experiments.append(exp3)

    out_path = os.path.join(SCRIPT_DIR, args.output_file)
    with open(out_path, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All experiments completed! Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
