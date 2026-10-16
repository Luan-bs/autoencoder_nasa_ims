"""
src/models/anomaly_detector.py

Unsupervised Anomaly Detection Baselines (PCA & Isolation Forest) with
unified interface for scoring and feature-level explainability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class BaseAnomalyDetector(ABC):
    """Unified interface for all anomaly detectors (baselines and autoencoders)."""

    name: str

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseAnomalyDetector":
        """Fit detector exclusively on healthy training data."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return 1D array of anomaly scores (higher value = more anomalous).
        """

    @abstractmethod
    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """
        Return 2D array (n_samples, n_features) of per-feature anomaly contributions
        for explainability.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return model-specific configuration and learned parameters."""

    def save(self, path: str | Path) -> None:
        """Persist fitted model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "BaseAnomalyDetector":
        """Load persisted model from disk."""
        return joblib.load(Path(path))


class PCADetector(BaseAnomalyDetector):
    """
    Linear Reconstruction Baseline using Principal Component Analysis (PCA).

    Retains enough principal components to explain `n_components` (default 95%)
    of healthy training variance. Anomaly score is the Mean Squared Reconstruction Error (MSE).
    """

    name = "pca"

    def __init__(self, n_components: float | int = 0.95, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed
        self.model = PCA(n_components=n_components, svd_solver="full", random_state=seed)

    def fit(self, X_train: np.ndarray) -> "PCADetector":
        self.model.fit(X_train)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self.model.transform(X)
        return self.model.inverse_transform(Z)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.reconstruct(X)
        return (X - X_hat) ** 2

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_type": "PCA",
            "variance_target": self.n_components,
            "n_components_selected": int(self.model.n_components_),
            "explained_variance_ratio_sum": round(float(np.sum(self.model.explained_variance_ratio_)), 4),
            "seed": self.seed,
        }


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Tree-based Isolation Baseline using Isolation Forest.

    Isolates observations by randomly selecting a feature and a split value.
    Anomalies require fewer splits (shorter path length in trees).
    Anomaly score is `-model.score_samples(X)` (strictly positive; higher = more anomalous).
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X_train)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # sklearn score_samples is in [-1, 0]; negate so higher = more anomalous
        return -self.model.score_samples(X)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute marginal anomaly score contribution per feature via single-feature
        ablation to the healthy mean (0 in scaled space).
        For each feature j, measures how much replacing X[:, j] with 0 (healthy mean)
        reduces the Isolation Forest anomaly score, plus squared z-score magnitude.
        """
        base_scores = self.score_samples(X)
        n_samples, n_features = X.shape
        contribs = np.zeros((n_samples, n_features), dtype=np.float64)

        for j in range(n_features):
            X_ablated = X.copy()
            X_ablated[:, j] = 0.0
            ablated_scores = self.score_samples(X_ablated)
            # Positive drop when feature j is reset to healthy baseline
            contribs[:, j] = np.maximum(0.0, base_scores - ablated_scores)

        # Fallback to z-score energy if tree score difference is flat
        row_sums = contribs.sum(axis=1, keepdims=True)
        zero_mask = (row_sums[:, 0] < 1e-9)
        if np.any(zero_mask):
            contribs[zero_mask] = X[zero_mask] ** 2

        return contribs

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_type": "IsolationForest",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "seed": self.seed,
        }


def build_detector(
    model_name: str,
    seed: int = 42,
    model_cfg: dict[str, Any] | None = None,
    train_cfg: dict[str, Any] | None = None,
) -> BaseAnomalyDetector:
    """Factory function for all baselines (PCA, IsolationForest) and Autoencoders (MLP, Conv1D, LSTM)."""
    key = model_name.lower().strip()
    m_cfg = model_cfg or {}
    t_cfg = train_cfg or {}

    if key == "pca":
        return PCADetector(
            n_components=m_cfg.get("n_components", 0.95),
            seed=seed,
        )
    elif key in ("isolation_forest", "iforest", "if"):
        return IsolationForestDetector(
            n_estimators=int(m_cfg.get("n_estimators", 200)),
            contamination=m_cfg.get("contamination", "auto"),
            seed=seed,
        )
    elif key in ("mlp", "conv1d", "lstm"):
        from src.models.autoencoder import AutoencoderDetector

        return AutoencoderDetector(
            arch=key,
            model_cfg=m_cfg,
            train_cfg=t_cfg,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: 'pca', 'isolation_forest', 'mlp', 'conv1d', 'lstm'."
        )


