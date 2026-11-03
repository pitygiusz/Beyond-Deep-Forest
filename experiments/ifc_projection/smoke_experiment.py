"""Small sklearn-style example for one or all Deep Forest variants.

Run from the repository root with:
    python -m experiments.ifc_projection.smoke_experiment
"""

from __future__ import annotations

import argparse

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import DeepForestVariantClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=(*DeepForestVariantClassifier.VARIANTS, "all"), default="combined"
    )
    args = parser.parse_args()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    variants = DeepForestVariantClassifier.VARIANTS if args.variant == "all" else (args.variant,)

    for variant in variants:
        estimator = DeepForestVariantClassifier(
            variant=variant,
            n_estimators=2,
            n_trees=30,
            max_layers=2,
            projection_width=8,
            interaction_limit=4,
            interaction_bootstraps=2,
            interaction_trees=5,
            rit_trees=8,
            random_state=42,
            n_jobs=2,
        )
        model = clone(estimator).fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        print(
            f"{variant}: accuracy={accuracy_score(y_test, predicted):.4f}, "
            f"layers={model.n_layers_}, probabilities={probabilities.shape}, "
            f"interactions={model.interaction_counts_by_layer_}"
        )


if __name__ == "__main__":
    main()
