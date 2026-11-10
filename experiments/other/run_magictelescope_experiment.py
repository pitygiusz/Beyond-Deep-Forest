"""
MagicTelescope Benchmark:
Evaluating (1) Standard Deep Forest, (2) Outlier-Pruned Deep Forest, and (3) Transformer-Augmented Deep Forest.

Dataset Details:
- Name: MagicTelescope (OpenML ID: 1120 / name='MagicTelescope')
- Samples: 19,020
- Features: 10 continuous features (fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist)
- Classes: 2 (gamma rays 'g' vs hadron rays 'h')

Experimental Suite:
1. Standard Deep Forest (gcForest; Zhou & Feng 2019): Baseline PFC
2. Outlier-Pruned Deep Forest: Removes the 2 outlier forests (highest consensus divergence) from the augmented vector
3. Transformer-Augmented Deep Forest: Inter-Layer Transformer Attention Module (ITAM) with Multi-Head Self-Attention
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from deepforest import CascadeForestClassifier
from outlier_pruned_deep_forest import OutlierPrunedCascadeForestClassifier
from transformer_deep_forest import TransformerCascadeForestClassifier


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n--- Training {name} ---")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # Ensure numerical normalization for probability sum
    y_proba = y_proba / np.sum(y_proba, axis=1, keepdims=True)
    infer_time = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba[:, 1])
    loss = log_loss(y_test, y_proba)
    n_layers = getattr(model, "n_layers_", 1)

    print(f"[{name}] Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | F1: {macro_f1:.4f} | AUC: {auc:.4f} | Loss: {loss:.4f} | Layers: {n_layers} | Fit Time: {fit_time:.2f}s")

    pruning_info = None
    if hasattr(model, "get_pruning_report"):
        try:
            pruning_info = model.get_pruning_report()
        except Exception:
            pruning_info = None

    return {
        "name": name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "roc_auc": float(auc),
        "log_loss": float(loss),
        "fit_time": float(fit_time),
        "infer_time": float(infer_time),
        "n_layers": int(n_layers),
        "pruning_report": pruning_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", type=int, default=35)
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="magictelescope_experiment_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("LOADING MAGICTELESCOPE DATASET FROM OPENML...")
    data = fetch_openml(name="MagicTelescope", version=1, as_frame=False)
    X = data.data.astype(np.float64)
    y = LabelEncoder().fit_transform(data.target)
    print(f"Dataset Loaded: Shape={X.shape}, Classes={np.unique(y)}, Class balance={np.bincount(y)}")
    print(f"Hyperparameters: n_estimators={args.n_estimators} ({args.n_estimators*2} forests/layer), n_trees={args.n_trees}")
    print("=" * 80)

    # 1. Primary Holdout Evaluation (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nHoldout Split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    holdout_results = {}

    # Model 1: Standard Deep Forest
    std_df = CascadeForestClassifier(
        n_estimators=args.n_estimators,
        n_trees=args.n_trees,
        random_state=42,
        verbose=1,
        n_jobs=2,
    )
    holdout_results["standard_deep_forest"] = evaluate_model(
        "Standard Deep Forest", std_df, X_train, y_train, X_test, y_test
    )

    # Model 2: Outlier-Pruned Deep Forest
    outlier_df = OutlierPrunedCascadeForestClassifier(
        n_estimators=args.n_estimators,
        n_trees=args.n_trees,
        n_outliers_removed=2,
        divergence_metric="euclidean",
        random_state=42,
        verbose=1,
        n_jobs=2,
    )
    holdout_results["outlier_pruned_deep_forest"] = evaluate_model(
        "Outlier-Pruned Deep Forest", outlier_df, X_train, y_train, X_test, y_test
    )

    # Model 3: Transformer-Augmented Deep Forest
    trans_df = TransformerCascadeForestClassifier(
        n_estimators=args.n_estimators,
        n_trees=args.n_trees,
        d_token=16,
        n_heads=2,
        transformer_epochs=12,
        random_state=42,
        verbose=1,
        n_jobs=2,
    )
    holdout_results["transformer_deep_forest"] = evaluate_model(
        "Transformer-Augmented Deep Forest", trans_df, X_train, y_train, X_test, y_test
    )

    # 2. 3-Fold Cross-Validation for Statistical Reliability
    print("\n" + "=" * 80)
    print("RUNNING 3-FOLD STRATIFIED CROSS-VALIDATION ON MAGICTELESCOPE...")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_records = {
        "standard_deep_forest": [],
        "outlier_pruned_deep_forest": [],
        "transformer_deep_forest": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n>>> Cross-Validation Fold {fold + 1}/3 <<<")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Standard
        m_std = CascadeForestClassifier(
            n_estimators=args.n_estimators, n_trees=args.n_trees, random_state=42 + fold, verbose=0, n_jobs=2
        )
        res_std = evaluate_model("Standard (CV)", m_std, X_tr, y_tr, X_te, y_te)
        cv_records["standard_deep_forest"].append(res_std)

        # Outlier-Pruned
        m_out = OutlierPrunedCascadeForestClassifier(
            n_estimators=args.n_estimators, n_trees=args.n_trees, n_outliers_removed=2, random_state=42 + fold, verbose=0, n_jobs=2
        )
        res_out = evaluate_model("Outlier-Pruned (CV)", m_out, X_tr, y_tr, X_te, y_te)
        cv_records["outlier_pruned_deep_forest"].append(res_out)

        # Transformer
        m_trans = TransformerCascadeForestClassifier(
            n_estimators=args.n_estimators, n_trees=args.n_trees, d_token=16, n_heads=2, transformer_epochs=12, random_state=42 + fold, verbose=0, n_jobs=2
        )
        res_trans = evaluate_model("Transformer (CV)", m_trans, X_tr, y_tr, X_te, y_te)
        cv_records["transformer_deep_forest"].append(res_trans)

    cv_summary = {}
    for k in cv_records:
        accs = [r["accuracy"] for r in cv_records[k]]
        f1s = [r["macro_f1"] for r in cv_records[k]]
        aucs = [r["roc_auc"] for r in cv_records[k]]
        losses = [r["log_loss"] for r in cv_records[k]]
        layers = [r["n_layers"] for r in cv_records[k]]
        times = [r["fit_time"] for r in cv_records[k]]
        cv_summary[k] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "log_loss_mean": float(np.mean(losses)),
            "log_loss_std": float(np.std(losses)),
            "n_layers_mean": float(np.mean(layers)),
            "fit_time_mean": float(np.mean(times)),
            "raw_folds": cv_records[k],
        }

    final_payload = {
        "dataset": "MagicTelescope",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": 2,
        "n_estimators": args.n_estimators,
        "n_trees": args.n_trees,
        "holdout_results": holdout_results,
        "cv_summary": cv_summary,
    }

    out_path = os.path.join(SCRIPT_DIR, args.output_file)
    with open(out_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"MagicTelescope Experiments Completed! Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
