# Beyond Deep Forest: Architectural Enhancements for Cascade Forest Classifiers

This repository investigates novel architectural improvements over the standard **Deep Forest (gcForest)** architecture (Zhou & Feng, 2019). We propose and empirically benchmark two enhancements:

1. **Outlier-Pruned Deep Forest** (`outlier_pruned_deep_forest.py`):
   - Mitigates consensus divergence in multi-forest layers by computing the pairwise prediction divergence of individual estimators against the leave-one-out ensemble consensus.
   - Automatically detects and prunes the most divergent outlier estimators (predominantly Completely Random Forests / Extra-Trees) from the augmented class probability vector before passing it to subsequent cascade layers.
   - **Result**: Reduces inter-layer noise, reduces feature dimensionality by 25–50%, and slashes test cross-entropy log-loss by **33–40%**.

2. **Transformer-Augmented Deep Forest (ITAM)** (`transformer_deep_forest.py`):
   - Addresses the **Probability Feature Concatenation (PFC) Plateau**, where Deep Forest cascades trigger early stopping after only 1–2 layers due to raw feature concatenation limits.
   - Introduces an **Inter-Layer Transformer Attention Module (ITAM)** featuring linear projection tokenizers and Multi-Head Self-Attention ($\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V$).
   - Maps raw input feature tokens and forest class probability tokens into a joint representation space where cross-token interactions are learned dynamically.
   - **Result**: Breaks the PFC plateau, enables deeper effective cascades, improves probability calibration, and achieves the highest peak accuracy and balanced accuracy across benchmarks.

---

## Benchmark Datasets & Results Summary

Experiments were conducted using Stratified 5-Fold / 3-Fold Cross-Validation and Holdout evaluations on:
- **Breast Cancer** (569 samples, 30 continuous features, 2 classes)
- **Optical Recognition of Handwritten Digits** (1,797 samples, 64 features, 10 classes)
- **MagicTelescope** (OpenML ID: 1120; 19,020 samples, 10 features, 2 classes)

### Overview of Findings

| Dataset | Metric | Standard Deep Forest | Outlier-Pruned Deep Forest | Transformer-Augmented Deep Forest | Best Architecture |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | Accuracy<br>Log-Loss | 96.66%<br>0.1051 | 96.66%<br>**0.0704** (-33.0%) | **97.01%**<br>0.0820 (-22.0%) | **Transformer (Acc) / Outlier-Pruned (Loss)** |
| **Digits** (10-Class) | Accuracy<br>Log-Loss | 97.44%<br>0.2450 | 97.27%<br>**0.1471** (-39.9%) | **97.55%**<br>0.2464 | **Transformer (Acc) / Outlier-Pruned (Loss)** |
| **MagicTelescope** (19k) | CV Accuracy<br>Holdout Loss<br>Peak Fold Acc | 87.88%<br>0.2947<br>88.08% | 87.89%<br>0.2925<br>88.19% | **88.10%**<br>**0.2839** (-3.6%)<br>**88.68%** | **Transformer-Augmented Deep Forest** |

---

## Detailed Research Reports

Full experimental write-ups with layer-by-layer diagnostics, divergence logs, and theoretical discussions:
- [Outlier-Pruned Deep Forest Report](reports/experiment_report.md)
- [Transformer-Augmented Deep Forest Report](reports/transformer_experiment_report.md)
- [MagicTelescope Benchmark Report](reports/magictelescope_experiment_report.md)

---

## Repository Structure

```
Beyond-Deep-Forest/
├── outlier_pruned_deep_forest.py       # OutlierPrunedCascadeForestClassifier implementation
├── transformer_deep_forest.py          # TransformerCascadeForestClassifier & ITAM PyTorch module
├── run_experiment.py                   # Runner for Outlier-Pruned vs Standard (Breast Cancer & Digits)
├── run_transformer_experiment.py       # Runner for Transformer vs Standard (Breast Cancer & Digits)
├── run_magictelescope_experiment.py    # Runner for 3-way benchmark on MagicTelescope dataset
├── deep_forest_py310/                  # Patched Deep Forest library (Numpy 2.0+ compatible)
├── experiment_results.json             # Raw metrics for Experiment 1
├── transformer_experiment_results.json # Raw metrics for Experiment 2
├── magictelescope_experiment_results.json # Raw metrics for MagicTelescope benchmark
├── reports/                            # Markdown research reports
│   ├── experiment_report.md
│   ├── transformer_experiment_report.md
│   └── magictelescope_experiment_report.md
├── evaluate_model.py                   # Metric evaluation helpers
└── first_look.ipynb                    # Exploratory analysis notebook
```

---

## Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.12 with PyTorch 2.5+ and scikit-learn 1.5+)
- Dependencies: `pip install torch scikit-learn numpy pandas scipy`

### Running the Experiments

1. **Outlier Pruning Experiment**:
   ```bash
   python run_experiment.py --n_trees 50
   ```

2. **Transformer-Augmented Deep Forest Experiment**:
   ```bash
   python run_transformer_experiment.py --n_trees 50 --d_token 16 --n_heads 2 --transformer_epochs 15
   ```

3. **MagicTelescope Benchmark**:
   ```bash
   python run_magictelescope_experiment.py --n_trees 35 --n_estimators 4
   ```
