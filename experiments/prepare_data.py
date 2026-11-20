import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW_DIR = Path(__file__).resolve().parent.parent / "raw_data"
CACHE_DIR = RAW_DIR / "cache"

IMDB_PARQUET = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/{}-00000-of-00001.parquet"
TFIDF_FEATURES = 5000

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
    "tuandromd": ("csv", RAW_DIR / "tuandromd.csv", "Label"),
    "imdb": ("imdb", None, "target"),  # text
    "hill_valley": ("openml", 1479, "target"),
    "amazon_employee_access": ("openml", 4135, "target"),
    "christine": ("openml", 41142, "target"),
    "kr_vs_kp": ("openml", 3, "target"),
    "ozone_level_8hr": ("openml", 1487, "target"),
    "steel_plates_fault": ("openml", 1504, "target"),
    "kc1": ("openml", 1067, "target"),
    # multiclass
    "mnist": ("openml", 554, "target"),
    "cifar10": ("openml", 40927, "target"),
    "orl": ("olivetti", None, "target"),  # sklearn
    "satimage": ("openml", 182, "target"),
    "letter": ("openml", 6, "target"),
    "yeast": ("openml", 181, "target"),
}


NON_TABULAR_DATASETS = frozenset({"imdb", "mnist", "cifar10", "orl"})
TABULAR_DATASETS = tuple(name for name in DATASETS if name not in NON_TABULAR_DATASETS)
TEXT_DATASETS = {"imdb"}

# datasets too large to use whole: keep a stratified random subset of this many rows
SUBSET_ROWS = {"mnist": 5000, "cifar10": 3000, "imdb": 3000}

# columns that would leak the target (other outcome columns of the same study)
LEAKY_COLUMNS = {
    "support2": ["death", "sfdm2"],
}


def _load_olivetti():
    """ORL faces: 400 images of 40 people, 64x64 pixels, cached by scikit-learn in raw_data/sklearn/."""
    X, y = fetch_olivetti_faces(return_X_y=True, data_home=str(RAW_DIR / "sklearn"))
    return pd.DataFrame(X, columns=[f"px{i}" for i in range(X.shape[1])]).assign(
        target=y
    )


def _load_imdb():
    """IMDB reviews (train + test, 50,000 total), downloaded once as Parquet; 1 = positive review."""
    frames = [
        pd.read_parquet(IMDB_PARQUET.format(split)) for split in ("train", "test")
    ]
    df = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"text": df["text"], "target": df["label"]})


def load_raw(source, key):
    if source == "openml":
        import openml

        ds = openml.datasets.get_dataset(key, download_data=True)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        return X.assign(target=y)
    if source == "uci":
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=key)
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    if source == "olivetti":
        return _load_olivetti()
    if source == "imdb":
        return _load_imdb()
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


def load_dataset(name, subset=True):
    """Return cleaned features X (DataFrame) and integer labels y (cached in raw_data/cache/).

    subset=False skips the SUBSET_ROWS cut and returns every row.
    """
    cache = CACHE_DIR / f"{name}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            X, y = pickle.load(f)
    else:
        source, key, target = DATASETS[name]
        X, y = split_target(
            load_raw(source, key), target, drop=LEAKY_COLUMNS.get(name, ())
        )
        if name not in TEXT_DATASETS:
            X = clean_features(X)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((X, y), f)
    if subset and name in SUBSET_ROWS and len(y) > SUBSET_ROWS[name]:
        X, _, y, _ = train_test_split(
            X, y, train_size=SUBSET_ROWS[name], stratify=y, random_state=0
        )
    return X, y


def _as_str_with_missing(X):
    """Cast categoricals to str, with missing values as their own category."""
    return pd.DataFrame(X).astype(object).fillna("missing").astype(str)


def make_preprocessor(X, max_categories=20):
    """Unfitted transformer: median impute + scale numerics, one-hot categoricals (missing = own category)."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            (
                "to_str",
                FunctionTransformer(
                    _as_str_with_missing, feature_names_out="one-to-one"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=max_categories,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def get_raw_train_test(name, test_size=0.2, seed=0):
    """Stratified train/test split of the cleaned, not yet preprocessed dataset.

    Returns X_train, X_test (DataFrames), y_train, y_test (arrays).
    """
    X, y = load_dataset(name)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def preprocess_splits(name, X_train, *X_others):
    """Fit preprocessing on one training split and transform one or more held-out splits."""
    if name in TEXT_DATASETS:
        tfidf = TfidfVectorizer(max_features=TFIDF_FEATURES, dtype=np.float32)
        X_train_transformed = tfidf.fit_transform(X_train["text"]).toarray()
        return (
            X_train_transformed,
            *(tfidf.transform(X["text"]).toarray() for X in X_others),
        )

    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    return (
        X_train_transformed,
        *(preprocessor.transform(X) for X in X_others),
    )
