"""Small, dependency-light provenance helpers for saved experiment results."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    "prepare_data.py",
    "pyproject.toml",
    "uv.lock",
    "experiments/run_tabular.py",
    "experiments/run.py",
    "experiments/run_tabular_repro.py",
    "experiments/ifc_projection/run_experiment.py",
    "experiments/reproducibility.py",
    "experiments/ifc_projection/models.py",
    "experiments/ifc_projection/random_projected_deep_forest.py",
    "experiments/ifc_projection/interaction_features.py",
    "experiments/ifc_projection/utility_feedback.py",
)


def array_sha256(array) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def provenance(config: dict, source_files=SOURCE_FILES) -> dict:
    packages = ("numpy", "scipy", "scikit-learn", "deep-forest-py310",
                "pandas", "openml", "ucimlrepo", "xgboost", "optuna")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    details = {
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": sources,
    }
    details["run_id"] = hashlib.sha256(
        json.dumps(details, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return details
