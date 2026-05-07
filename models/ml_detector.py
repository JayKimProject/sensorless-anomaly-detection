"""
Sensorless Anomaly Detection — Classical ML Detectors

Scikit-learn based semi-supervised detectors trained on normal data only:

  • IsolationForestDetector — ensemble of random-cut trees
  • OneClassSVMDetector     — kernel-based novelty detection
  • LOFDetector             — local density-based outlier factor
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional


class IsolationForestDetector:
    """
    Isolation Forest: anomalies are isolated faster by random splits.

    Best for    : high-dimensional feature spaces, large datasets
    Limitation  : insensitive to local density structure
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (negated decision function)."""
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal."""
        Xs = self.scaler.transform(X)
        raw = self.model.predict(Xs)   # sklearn uses -1 / +1
        return ((raw == -1)).astype(int)


class OneClassSVMDetector:
    """
    One-Class SVM: learns a hypersphere boundary around normal data.

    Best for    : compact, well-clustered normal class
    Limitation  : does not scale to large N; sensitive to kernel/nu
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.05,
                 gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return ((self.model.predict(Xs) == -1)).astype(int)


class LOFDetector:
    """
    Local Outlier Factor: compares local density to k-nearest neighbours.

    Best for    : datasets with varying density clusters
    Limitation  : transductive — must re-fit to classify new points
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.scaler = StandardScaler()
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self._X_train = self.scaler.fit_transform(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        LOF must see training data at prediction time (transductive).
        We combine train and test, then return labels for test rows.
        """
        Xs_test = self.scaler.transform(X)
        combined = np.vstack([self._X_train, Xs_test])

        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        labels_all = lof.fit_predict(combined)
        labels_test = labels_all[len(self._X_train):]
        return (labels_test == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs_test = self.scaler.transform(X)
        combined = np.vstack([self._X_train, Xs_test])

        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        lof.fit_predict(combined)
        # negative_outlier_factor_: lower (more negative) = more anomalous
        nof_test = -lof.negative_outlier_factor_[len(self._X_train):]
        return nof_test


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generator import MotorCurrentGenerator
    from features.extractor import FeatureExtractor
    from sklearn.metrics import classification_report

    gen = MotorCurrentGenerator()
    X_raw, y = gen.generate_dataset(n_normal=400, n_fault_each=50)
    Xf = FeatureExtractor().extract_batch(X_raw)

    X_train = Xf[y == 0]
    X_test, y_test = Xf, (y != 0).astype(int)

    for Cls in (IsolationForestDetector, OneClassSVMDetector, LOFDetector):
        det = Cls().fit(X_train)
        preds = det.predict(X_test)
        print(f"\n{'='*50}")
        print(f" {Cls.__name__}")
        print(classification_report(y_test, preds,
                                    target_names=["normal", "anomaly"]))
