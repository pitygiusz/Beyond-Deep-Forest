"""
Comprehensive Experiment: Standard Deep Forest vs. Transformer-Augmented Deep Forest.

This experiment compares:
1. Standard Deep Forest (gcForest): Classic Prediction-based Feature Concatenation (PFC),
   where raw features and forest probability distributions are naively concatenated.
2. Transformer-Augmented Deep Forest (gcForest + ITAM): Replaces naive concatenation with an
   Inter-Layer Transformer Attention Module (ITAM) that performs Multi-Head Self-Attention
   between raw features and forest predictions to dynamically contextualize information before
   passing it to subsequent cascade layers.

Evaluation Protocol:
- Stratified 5-Fold Cross-Validation on:
  * Breast Cancer Wisconsin (Binary classification, 569 samples, 30 features)
  * Digits (Multiclass classification, 1797 samples, 64 features, 10 classes)
- Metrics:
  * Accuracy, Balanced Accuracy, Macro F1, ROC-AUC, Log-Loss
  * Training time, Inference time, Optimal cascade layers
  * Statistical hypothesis testing (paired t-test and Wilcoxon signed-rank test)
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
from transformer_deep_forest import TransformerCascadeForestClassifier


def evaluate_model_fold(model, X_train, y_train, X_test, y_test, is_multiclass=False):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    infer_time = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    try:
        if is_multiclass:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1])
    except Exception:
        auc = np.nan

    try:
        loss = log_loss(y_test, y_proba)
    except Exception:
        loss = np.nan

    n_layers = getattr(model, "n_layers_", 1)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "log_loss": loss,
        "fit_time": fit_time,
        "infer_time": infer_time,
        "n_layers": n_layers,
    }


def run_benchmark(dataset_name="breast_cancer", n_estimators=4, n_trees=50, n_splits=5, random_state=42):
    print("=" * 80)
    print(f"EXPERIMENT: Standard Deep Forest vs. Transformer Deep Forest")
    print(f"Dataset: {dataset_name.upper()} | n_estimators={n_estimators} ({n_estimators*2} forests/layer) | n_trees={n_trees}")
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
        "standard_deep_forest": [],
        "transformer_deep_forest": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # 1. Standard Deep Forest
        print("Training Standard Deep Forest (gcForest)...")
        std_model = CascadeForestClassifier(
            n_estimators=n_estimators,
            n_trees=n_trees,
            random_state=random_state + fold,
            verbose=0,
            n_jobs=2,
        )
        std_metrics = evaluate_model_fold(std_model, X_train, y_train, X_test, y_test, is_multiclass)
        results["standard_deep_forest"].append(std_metrics)
        print(f"  [Standard]    Acc: {std_metrics['accuracy']:.4f} | F1: {std_metrics['macro_f1']:.4f} | Layers: {std_metrics['n_layers']} | Loss: {std_metrics['log_loss']:.4f} | Time: {std_metrics['fit_time']:.2f}s")

        # 2. Transformer-Augmented Deep Forest
        print("Training Transformer-Augmented Deep Forest (gcForest + ITAM)...")
        trans_model = TransformerCascadeForestClassifier(
            n_estimators=n_estimators,
            n_trees=n_trees,
            d_token=16,
            n_heads=2,
            transformer_epochs=15,
            random_state=random_state + fold,
            verbose=0,
            n_jobs=2,
        )
        trans_metrics = evaluate_model_fold(trans_model, X_train, y_train, X_test, y_test, is_multiclass)
        results["transformer_deep_forest"].append(trans_metrics)
        print(f"  [Transformer] Acc: {trans_metrics['accuracy']:.4f} | F1: {trans_metrics['macro_f1']:.4f} | Layers: {trans_metrics['n_layers']} | Loss: {trans_metrics['log_loss']:.4f} | Time: {trans_metrics['fit_time']:.2f}s")

    # Aggregate metrics
    summary = {}
    for model_key in ["standard_deep_forest", "transformer_deep_forest"]:
        m_list = results[model_key]
        summary[model_key] = {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in m_list])),
            "accuracy_std": float(np.std([m["accuracy"] for m in m_list])),
            "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in m_list])),
            "balanced_accuracy_std": float(np.std([m["balanced_accuracy"] for m in m_list])),
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in m_list])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in m_list])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in m_list])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in m_list])),
            "log_loss_mean": float(np.mean([m["log_loss"] for m in m_list])),
            "log_loss_std": float(np.std([m["log_loss"] for m in m_list])),
            "fit_time_mean": float(np.mean([m["fit_time"] for m in m_list])),
            "fit_time_std": float(np.std([m["fit_time"] for m in m_list])),
            "infer_time_mean": float(np.mean([m["infer_time"] for m in m_list])),
            "infer_time_std": float(np.std([m["infer_time"] for m in m_list])),
            "n_layers_mean": float(np.mean([m["n_layers"] for m in m_list])),
            "raw_folds": m_list,
        }

    # Hypothesis testing
    std_accs = [m["accuracy"] for m in results["standard_deep_forest"]]
    trans_accs = [m["accuracy"] for m in results["transformer_deep_forest"]]

    try:
        t_stat, p_val_ttest = stats.ttest_rel(trans_accs, std_accs)
    except Exception:
        t_stat, p_val_ttest = np.nan, np.nan

    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(trans_accs, std_accs)
    except Exception:
        w_stat, p_val_wilcoxon = np.nan, np.nan

    stats_summary = {
        "ttest_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "ttest_pval": float(p_val_ttest) if not np.isnan(p_val_ttest) else None,
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pval": float(p_val_wilcoxon) if not np.isnan(p_val_wilcoxon) else None,
    }

    return {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "n_estimators": n_estimators,
        "n_trees": n_trees,
        "n_splits": n_splits,
        "summary": summary,
        "statistical_tests": stats_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="transformer_experiment_results.json")
    args = parser.parse_args()

    experiments = []

    # Experiment 1: Breast Cancer (Primary benchmark from plan)
    exp1 = run_benchmark(
        dataset_name="breast_cancer",
        n_estimators=4,
        n_trees=args.n_trees,
        n_splits=5,
        random_state=42,
    )
    experiments.append(exp1)

    # Experiment 2: Digits (10-class multi-class tabular benchmark)
    exp2 = run_benchmark(
        dataset_name="digits",
        n_estimators=4,
        n_trees=args.n_trees,
        n_splits=5,
        random_state=42,
    )
    experiments.append(exp2)

    out_path = os.path.join(SCRIPT_DIR, args.output_file)
    with open(out_path, "w") as f:
        json.dump(experiments, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All Transformer experiments completed! Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
