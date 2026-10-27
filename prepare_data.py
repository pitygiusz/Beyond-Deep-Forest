"""Load, clean and split the tabular datasets.

Usage:  X_train, X_val, X_test, y_train, y_val, y_test = get_split("adult")
"""
from pathlib import Path

import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("raw_data")

# name -> (source, id or path, target column)
DATASETS = {
    # binary
    "spambase": ("openml", 44, "target"),
    "musk_v2": ("openml", 1116, "target"),
    "madelon": ("openml", 1485, "target"),
    "kick": ("openml", 41162, "target"),
    "arcene": ("openml", 1458, "target"),
    "bank_marketing": ("openml", 1461, "target"),
    "adult": ("openml", 1590, "target"),
    "support2": ("uci", 880, "hospdead"),
    # TUNADROMD (UCI id 813) is not available via ucimlrepo or OpenML - downloaded manually
    "tuandromd": ("csv", RAW_DIR / "tuandromd.csv", "Label"),
    # multiclass
    "satimage": ("openml", 182, "target"),
    "letter": ("openml", 6, "target"),
    "yeast": ("openml", 181, "target"),
}

# columns that would leak the target (other outcome columns of the same study)
LEAKY_COLUMNS = {
    "support2": ["death", "sfdm2"],
}


def load_raw(source, key):
    if source == "openml":
        ds = openml.datasets.get_dataset(key, download_data=True)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        return X.assign(target=y)
    if source == "uci":
        ds = fetch_ucirepo(id=key)
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    return pd.read_csv(key)


def split_target(df, target, drop=()):
    """Drop rows without a label and encode the target as 0..k-1.

    Binary targets get minority class = 1; multiclass targets are numbered in sorted label order.
    """
    df = df.dropna(subset=[target])
    classes = df[target].value_counts()
    if len(classes) == 2:
        y = (df[target] == classes.index[-1]).astype(int)
    else:
        y = df[target].astype("category").cat.codes
    X = df.drop(columns=[target, *drop], errors="ignore")
    return X, y.to_numpy()


def clean_features(X, max_unique_ratio=0.5):
    """Drop constant columns and ID-like categorical columns."""
    X = X.copy()
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    categorical = X.select_dtypes(exclude="number").columns
    id_like = [c for c in categorical if X[c].nunique() > max_unique_ratio * len(X)]
    return X.drop(columns=constant + id_like)


def load_dataset(name):
    """Return cleaned features X (DataFrame) and integer labels y."""
    source, key, target = DATASETS[name]
    X, y = split_target(load_raw(source, key), target, drop=LEAKY_COLUMNS.get(name, ()))
    return clean_features(X), y


def _as_str_with_missing(X):
    """Cast categoricals to str, with missing values as their own category."""
    return pd.DataFrame(X).astype(object).fillna("missing").astype(str)


def make_preprocessor(X, max_categories=20):
    """Unfitted transformer: median impute + scale numerics, one-hot categoricals (missing = own category)."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("to_str", FunctionTransformer(_as_str_with_missing, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=max_categories,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def get_split(name, val_size=0.2, test_size=0.2, seed=0):
    """Stratified train/val/test split; the preprocessor is fitted on train only.

    Returns X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.
    """
    X, y = load_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=seed)

    prep = make_preprocessor(X_train)
    X_train = prep.fit_transform(X_train)
    return X_train, prep.transform(X_val), prep.transform(X_test), y_train, y_val, y_test


if __name__ == "__main__":
    for name in DATASETS:
        X_train, X_val, X_test, y_train, y_val, y_test = get_split(name)
        print(f"{name}: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}, "
              f"{len(set(y_train))} classes")
