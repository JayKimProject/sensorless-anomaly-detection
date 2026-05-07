"""
Sensorless Anomaly Detection — Dense Autoencoder

Architecture
------------
Encoder : F → 64 → 32 → latent_dim
Decoder : latent_dim → 32 → 64 → F

Anomaly score = MSE reconstruction error.
Normal samples reconstruct well; anomalies have high error.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List


# ------------------------------------------------------------------
# Network definition
# ------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# ------------------------------------------------------------------
# Detector wrapper
# ------------------------------------------------------------------

class AutoencoderDetector:
    """
    Train a dense autoencoder on normal-class features, then use
    reconstruction error as an anomaly score at inference time.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DenseAutoencoder] = None
        self.threshold_: float = 0.0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / (self._std + 1e-12)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, verbose: bool = True) -> "AutoencoderDetector":
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        Xn = self._normalise(X)

        input_dim = Xn.shape[1]
        self.model = DenseAutoencoder(input_dim, self.latent_dim).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(self._to_tensor(Xn))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        history: List[float] = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            avg_loss = epoch_loss / len(Xn)
            history.append(avg_loss)
            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:4d}/{self.epochs}  loss={avg_loss:.6f}")

        self.history_ = history
        self.fit_threshold(X, contamination=0.05)
        return self

    def fit_threshold(self, X: np.ndarray, contamination: float = 0.05) -> float:
        scores = self.score(X)
        self.threshold_ = float(np.quantile(scores, 1 - contamination))
        return self.threshold_

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error."""
        assert self.model is not None, "Call fit() first."
        Xn = self._normalise(X)
        t = self._to_tensor(Xn)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(t)
        errors = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


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

    print("Training AutoencoderDetector …")
    det = AutoencoderDetector(epochs=80, latent_dim=6).fit(X_train)
    preds = det.predict(X_test)
    print(classification_report(y_test, preds, target_names=["normal", "anomaly"]))
