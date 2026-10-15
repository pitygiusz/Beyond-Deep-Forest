"""
Implementation of Transformer-Augmented Deep Forest.

Architecture:
In standard Deep Forest (gcForest), cascade levels pass class probability vectors
from M forests via flat concatenation (Prediction-based Feature Concatenation - PFC).

In Transformer-Augmented Deep Forest:
Between cascade levels, an Inter-Layer Transformer Attention Module (ITAM) is introduced:
1. Raw feature tokens and forest prediction tokens are embedded into a shared latent space.
2. Multi-Head Self-Attention (MHSA) computes cross-attention weights between raw features
   and forest probability outputs.
3. The attention-contextualized representations are projected back into calibrated,
   context-aware class distributions for the subsequent cascade layer.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import is_classifier, ClassifierMixin
from sklearn.metrics import accuracy_score

# Ensure local deepforest package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "deep_forest_py310")))

from deepforest import CascadeForestClassifier
from deepforest._layer import ClassificationCascadeLayer, _build_estimator
from deepforest import _utils


class InterLayerTransformer(nn.Module):
    """
    Lightweight Inter-Layer Transformer that performs Multi-Head Self-Attention
    across raw features and forest probability distributions.
    """

    def __init__(self, n_raw_features, n_forests, n_classes, d_token=16, n_heads=2, dropout=0.05):
        super().__init__()
        self.n_raw_features = n_raw_features
        self.n_forests = n_forests
        self.n_classes = n_classes
        self.d_token = d_token

        # Feature tokenizer: project raw scalar features to d_token
        self.feature_proj = nn.Linear(1, d_token)

        # Forest tokenizer: project class probability vector (n_classes) to d_token
        self.forest_proj = nn.Linear(n_classes, d_token)

        # Positional/identity embeddings for tokens
        self.type_embed = nn.Embedding(2, d_token)  # 0 for raw features, 1 for forests

        # Transformer Encoder Layer with Multi-Head Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Projection head: project contextualized forest tokens back to class probability logits
        self.out_head = nn.Linear(d_token, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_raw, x_forests):
        """
        x_raw: (batch_size, n_raw_features)
        x_forests: (batch_size, n_forests, n_classes)
        Returns:
            contextualized_probas: (batch_size, n_forests * n_classes)
            logits_for_loss: (batch_size, n_classes)
        """
        B = x_raw.size(0)

        # 1. Project tokens
        raw_tokens = self.feature_proj(x_raw.unsqueeze(-1))  # (B, n_raw_features, d_token)
        forest_tokens = self.forest_proj(x_forests)  # (B, n_forests, d_token)

        # 2. Add type embeddings
        raw_tokens = raw_tokens + self.type_embed(torch.zeros(self.n_raw_features, dtype=torch.long, device=x_raw.device))
        forest_tokens = forest_tokens + self.type_embed(torch.ones(self.n_forests, dtype=torch.long, device=x_forests.device))

        # 3. Concatenate tokens along sequence dimension
        all_tokens = torch.cat([raw_tokens, forest_tokens], dim=1)  # (B, n_raw_features + n_forests, d_token)

        # 4. Multi-Head Self-Attention across features and forests
        transformed = self.transformer(all_tokens)

        # 5. Extract contextualized forest tokens
        ctx_forest_tokens = transformed[:, self.n_raw_features :, :]  # (B, n_forests, d_token)

        # 6. Project to refined class distributions
        logits = self.out_head(ctx_forest_tokens)  # (B, n_forests, n_classes)
        probas = self.softmax(logits)  # (B, n_forests, n_classes)

        # Flatten for cascade augmentation: (B, n_forests * n_classes)
        flat_probas = probas.reshape(B, self.n_forests * self.n_classes)

        # Mean logits across forests for training loss
        mean_logits = torch.mean(logits, dim=1)  # (B, n_classes)

        return flat_probas, mean_logits


class TransformerClassificationCascadeLayer(ClassificationCascadeLayer):
    """
    Cascade forest layer equipped with an Inter-Layer Transformer Attention Module (ITAM).
    """

    def __init__(
        self,
        layer_idx,
        n_outputs,
        criterion,
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        partial_mode=False,
        buffer=None,
        n_jobs=None,
        random_state=None,
        verbose=1,
        n_raw_features=None,
        d_token=16,
        n_heads=2,
        transformer_epochs=15,
        lr=1e-3,
    ):
        super().__init__(
            layer_idx=layer_idx,
            n_outputs=n_outputs,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            backend=backend,
            partial_mode=partial_mode,
            buffer=buffer,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_raw_features = n_raw_features
        self.d_token = d_token
        self.n_heads = n_heads
        self.transformer_epochs = transformer_epochs
        self.lr = lr
        self.transformer_ = None
        self.device = torch.device("cpu")

    def fit_transform(self, X, y, sample_weight=None):
        self._validate_params()
        n_samples, self.n_features = X.shape

        if self.n_raw_features is None:
            self.n_raw_features = self.n_features

        X_aug = []
        oob_decision_function = np.zeros((n_samples, self.n_outputs))

        # 1. Fit Random Forests (self.n_estimators // 2)
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "rf",
                self._make_estimator(estimator_idx, "rf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "rf")
            self.estimators_.update({key: _estimator})

        # 2. Fit Extra-Trees (self.n_estimators // 2)
        for estimator_idx in range(self.n_estimators // 2):
            X_aug_, _estimator = _build_estimator(
                X,
                y,
                self.layer_idx,
                estimator_idx,
                "erf",
                self._make_estimator(estimator_idx, "erf"),
                oob_decision_function,
                self.partial_mode,
                self.buffer,
                self.verbose,
                sample_weight,
            )
            X_aug.append(X_aug_)
            key = "{}-{}-{}".format(self.layer_idx, estimator_idx, "erf")
            self.estimators_.update({key: _estimator})

        # Shape: (n_samples, n_forests, n_outputs)
        X_forests_np = np.stack(X_aug, axis=1).astype(np.float32)
        X_raw_np = X[:, : self.n_raw_features].astype(np.float32)

        # 3. Build & Train Inter-Layer Transformer Attention Module
        self.transformer_ = InterLayerTransformer(
            n_raw_features=self.n_raw_features,
            n_forests=self.n_estimators,
            n_classes=self.n_outputs,
            d_token=self.d_token,
            n_heads=self.n_heads,
        ).to(self.device)

        # Normalize raw features for stable neural training
        self.raw_mean_ = np.mean(X_raw_np, axis=0, keepdims=True)
        self.raw_std_ = np.std(X_raw_np, axis=0, keepdims=True) + 1e-6
        X_raw_norm = (X_raw_np - self.raw_mean_) / self.raw_std_

        x_raw_tensor = torch.tensor(X_raw_norm, dtype=torch.float32, device=self.device)
        x_forests_tensor = torch.tensor(X_forests_np, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        optimizer = torch.optim.AdamW(self.transformer_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        batch_size = min(128, n_samples)
        indices = np.arange(n_samples)

        self.transformer_.train()
        for epoch in range(self.transformer_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                b_idx = indices[start : start + batch_size]
                optimizer.zero_grad()
                _, logits = self.transformer_(x_raw_tensor[b_idx], x_forests_tensor[b_idx])
                loss = criterion(logits, y_tensor[b_idx])
                loss.backward()
                optimizer.step()

        # 4. Generate Attention-Contextualized Augmented Features
        self.transformer_.eval()
        with torch.no_grad():
            flat_probas, _ = self.transformer_(x_raw_tensor, x_forests_tensor)
            X_aug_transformed = flat_probas.cpu().numpy().astype(np.float64)

        # 5. Set OOB decision function from contextualized probabilities
        reshaped = X_aug_transformed.reshape(n_samples, self.n_estimators, self.n_outputs)
        self.oob_decision_function_ = np.mean(reshaped, axis=1)
        y_pred = np.argmax(self.oob_decision_function_, axis=1)
        self.val_performance_ = accuracy_score(y, y_pred, sample_weight=sample_weight)

        return X_aug_transformed

    def predict_full(self, X):
        """
        Evaluate all base estimators and transform predictions through the fitted Transformer.
        """
        n_samples, _ = X.shape
        pred_list = []
        for idx, (key, estimator) in enumerate(self.estimators_.items()):
            if self.verbose > 1:
                msg = "{} - Evaluating estimator = {:<5} in layer = {}"
                key_short = key.split("-")[-1] + "_" + str(key.split("-")[-2])
                print(msg.format(_utils.ctime(), key_short, self.layer_idx))
            if self.partial_mode:
                estimator = self.buffer.load_estimator(estimator)
            pred_list.append(estimator.predict(X))

        X_forests_np = np.stack(pred_list, axis=1).astype(np.float32)
        X_raw_np = X[:, : self.n_raw_features].astype(np.float32)
        X_raw_norm = (X_raw_np - self.raw_mean_) / self.raw_std_

        x_raw_tensor = torch.tensor(X_raw_norm, dtype=torch.float32, device=self.device)
        x_forests_tensor = torch.tensor(X_forests_np, dtype=torch.float32, device=self.device)

        self.transformer_.eval()
        with torch.no_grad():
            flat_probas, _ = self.transformer_(x_raw_tensor, x_forests_tensor)
            return flat_probas.cpu().numpy().astype(np.float64)

    def transform(self, X):
        return self.predict_full(X)


class TransformerCascadeForestClassifier(CascadeForestClassifier):
    """
    Deep Forest Classifier augmented with an Inter-Layer Transformer Attention Module (ITAM).
    Replaces naive Prediction-based Feature Concatenation (PFC) with Multi-Head Self-Attention
    between raw features and forest probability predictions.
    """

    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=4,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs=None,
        backend="custom",
        n_tolerant_rounds=2,
        delta=0.0,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
        d_token=16,
        n_heads=2,
        transformer_epochs=15,
        lr=1e-3,
    ):
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.d_token = d_token
        self.n_heads = n_heads
        self.transformer_epochs = transformer_epochs
        self.lr = lr
        self.n_raw_features_ = None

    def fit(self, X, y, sample_weight=None):
        self.n_raw_features_ = X.shape[1]
        return super().fit(X, y, sample_weight=sample_weight)

    def _make_layer(self, **layer_args):
        return TransformerClassificationCascadeLayer(
            **layer_args,
            n_raw_features=self.n_raw_features_,
            d_token=self.d_token,
            n_heads=self.n_heads,
            transformer_epochs=self.transformer_epochs,
            lr=self.lr,
        )
