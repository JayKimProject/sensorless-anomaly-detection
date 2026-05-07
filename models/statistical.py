"""
Sensorless Anomaly Detection — Statistical Detectors

Classical, lightweight detectors that work well with small datasets
and require no GPU:

  • ZScoreDetector      — univariate z-score per feature
  • MahalanobisDetector — multivariate distance (covariance-aware)
  • IQRDetector         — robust interquartile-range fence
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseDetector(ABC):
    """Minimal interface shared by all detectors."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score (higher = more anomalous) for each row."""
        ...

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        scores = self.score(X)
        thr = threshold if threshold is not None else self.threshold_
        return (scores > thr).astype(int)

    def fit_threshold(self, X_val: np.ndarray, contamination: float = 0.05) -> float:
        """
        Set threshold from a validation set so that `contamination` fraction
        of samples are flagged as anomalies.
        """
        scores = self.score(X_val)
        self.threshold_ = float(np.quantile(scores, 1 - contamination))
        return self.threshold_


# ------------------------------------------------------------------
# Z-Score Detector
# ------------------------------------------------------------------

class ZScoreDetector(BaseDetector):
    """
    Per-feature z-score; anomaly score = max |z| across features.

    Advantages : interpretable, O(F) memory, instant inference
    Limitations: assumes feature independence, Gaussian distribution
    """

    def __init__(self, n_sigma: float = 3.0):
        self.n_sigma = n_sigma
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.threshold_: float = n_sigma

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)   # worst-case feature deviation


# ------------------------------------------------------------------
# Mahalanobis Detector
# ------------------------------------------------------------------

class MahalanobisDetector(BaseDetector):
    """
    Mahalanobis distance from the training-set centroid.

    D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

    Advantages : accounts for feature correlations
    Limitations: requires N >> F; singular covariance causes issues
    """

    def __init__(self, reg: float = 1e-4):
        self.reg = reg
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X.T)
        # regularisation to avoid singular matrix
        cov += self.reg * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean_
        # efficient batch computation: diag(diff @ cov_inv @ diff.T)
        left = diff @ self.cov_inv_
        d_sq = np.einsum("ij,ij->i", left, diff)
        return np.sqrt(np.maximum(d_sq, 0))


# ------------------------------------------------------------------
# IQR Detector
# ------------------------------------------------------------------

class IQRDetector(BaseDetector):
    """
    Robust detector using per-feature IQR fences.

    anomaly score = max over features of max(0, |x - median| - k·IQR/2)

    Advantages : robust to outliers in training data
    Limitations: feature independence assumption
    """

    def __init__(self, k: float = 1.5):
        self.k = k
        self.q1_: Optional[np.ndarray] = None
        self.q3_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> "IQRDetector":
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.median_ = np.median(X, axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        iqr = self.q3_ - self.q1_ + 1e-12
        fence = self.k * iqr / 2
        deviation = np.maximum(0, np.abs(X - self.median_) - fence)
        return deviation.max(axis=1)


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generator import MotorCurrentGenerator
    from features.extractor import FeatureExtractor

    gen = MotorCurrentGenerator()
    X_raw, y = gen.generate_dataset(n_normal=300, n_fault_each=30)
    Xf = FeatureExtractor().extract_batch(X_raw)

    # train only on normal
    X_train = Xf[y == 0]
    X_test  = Xf

    for Detector in (ZScoreDetector, MahalanobisDetector, IQRDetector):
        det = Detector().fit(X_train)
        det.fit_threshold(X_train, contamination=0.05)
        preds = det.predict(X_test)
        tp = ((preds == 1) & (y != 0)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y != 0)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        print(f"{Detector.__name__:25s}  precision={prec:.2f}  recall={rec:.2f}")
