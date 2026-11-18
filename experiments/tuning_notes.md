# Hyperparameter search with a 10-trial budget

## Problem

The plan is 10 Optuna trials per dataset and model. The search spaces in
`experiments/models.py` are large compared to that budget:

| Model | Searched parameters |
|---|---|
| `bare` | 5 |
| `projection` | 8 |
| `ifc` | 12 |
| `combined` | 15 |
| `utility_ifc` | 17 |
| `random_forest`, `extra_trees` | 4 |
| `hist_gradient_boosting` | 5 |

## Findings

1. **Bayesian and random search behave about the same at 10 trials.**
   Trial 1 is the default settings and trials 2–5 are random. Only trials 6–10 are chosen
   by bayesian search, from a model fitted on 5 points in up to 15 dimensions. Switching
   to random search therefore changes little.

2. **The noise is larger than the expected gains.** On `yeast`, changing only
   the model seed moved default `bare` from 0.477 to 0.564 test balanced
   accuracy. With 10 trials in 15 dimensions, the best trial is often just the
   luckiest one.

3. **Unequal search spaces bias the variant comparison.** Every variant gets
   the same 10 trials, so the variants with more parameters (`combined`,
   `utility_ifc`) are searched much more thinly than `bare`.

## Recommendation: tune 2–3 parameters per model, fix the rest at defaults

| Model | Tune | Why |
|---|---|---|
| all Deep Forests | `max_depth`, `min_samples_leaf` | Control overfitting, most important on small datasets |
| `projection` | + `projection_width` (0/4/16) | Defines the variant; 0 lets tuning switch projections off |
| `ifc` | + `interaction_limit`, `stability_threshold` | Number of interaction features and how strictly they are filtered |
| `combined` | + `projection_width`, `interaction_limit` | The core setting of each part |
| `random_forest`, `extra_trees` | `max_features`, `min_samples_leaf` | Usual high-impact settings |
| `hist_gradient_boosting` | `learning_rate`, `max_leaf_nodes` | Usual high-impact settings |

Fix at defaults:

- `n_trees`, `n_estimators`: more trees rarely hurt and only cost time; fixing
  them keeps runtimes comparable.
- `max_layers`: the cascade already picks its depth on a validation slice,
  so this is only an upper limit.
- Detailed IFC and projection settings (`rit_*`, `interaction_bootstraps`,
  `interaction_trees`, `interaction_folds`, `projection_depth`,
  `negative_slope`): second-order effects that 10 trials cannot resolve.

With 2–4 dimensions, 10 trials cover the space reasonably well, and the
choice of sampler matters even less.

## Further tips

- Optional: Optuna's `QMCSampler` (quasi-random) spreads a small number of
  trials more evenly than pure random search.
- Keep the default settings as trial 1 (as now), so the tuned CV score is
  never below the default one.
- Stay at 3 CV folds: more folds give less noisy scores, but each trial costs
  proportionally more.
