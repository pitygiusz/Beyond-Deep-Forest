"""Prepare dataset inputs for the MLP without scaling numeric features."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from prepare_data import TFIDF_FEATURES, TEXT_DATASETS, get_train_test, load_dataset


def _as_str_with_missing(X):
    return pd.DataFrame(X).astype(object).fillna("missing").astype(str)


def make_preprocessor(X):
    """Impute numeric inputs without scaling; encode categories for MLP compatibility."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("to_str", FunctionTransformer(_as_str_with_missing, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20,
                                     sparse_output=False)),
        ]), categorical),
    ])


def get_mlp_train_test(dataset, test_size=0.3, seed=0, input_mode="raw"):
    """Return a deterministic split with either the forest or raw-numeric preprocessing."""
    if input_mode == "standardized":
        return get_train_test(dataset, test_size=test_size, seed=seed)
    if input_mode != "raw":
        raise ValueError("input_mode must be 'standardized' or 'raw'")

    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    if dataset in TEXT_DATASETS:
        vectorizer = TfidfVectorizer(max_features=TFIDF_FEATURES, dtype=np.float32)
        train_features = vectorizer.fit_transform(X_train["text"]).toarray()
        test_features = vectorizer.transform(X_test["text"]).toarray()
        return train_features, test_features, y_train, y_test

    preprocessor = make_preprocessor(X_train)
    return (preprocessor.fit_transform(X_train), preprocessor.transform(X_test),
            y_train, y_test)