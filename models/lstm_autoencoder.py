"""
Sensorless Anomaly Detection — LSTM Autoencoder

Operates directly on raw time-series windows (no hand-crafted features).

Architecture
------------
Input        : (batch, seq_len, 1)
Encoder LSTM : seq → hidden state h_enc
Decoder LSTM : h_enc repeated → reconstruction
Output       : (batch, seq_len, 1)

Anomaly score = mean squared error over the reconstructed window.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # last layer's hidden state → (batch, hidden)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int,
                 seq_len: int, num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # repeat latent vector across time dimension
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.fc(out)   # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.encoder = LSTMEncoder(1, hidden_size, num_layers)
        self.decoder = LSTMDecoder(hidden_size, 1, seq_len, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ------------------------------------------------------------------
# Detector wrapper
# ------------------------------------------------------------------

class LSTMAutoencoderDetector:
    """
    Anomaly detector based on LSTM autoencoder reconstruction error.

    Accepts raw 1-D signals and internally windows them.
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 5e-4,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMAutoencoder] = None
        self.threshold_: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _segment(self, X: np.ndarray) -> np.ndarray:
        """
        Slide windows over each signal in X (N, L) → windows (W, window_size).
        Also returns mapping: which original signal each window came from.
        """
        windows = []
        mapping = []
        for i, x in enumerate(X):
            L = len(x)
            start = 0
            while start + self.window_size <= L:
                windows.append(x[start: start + self.window_size])
                mapping.append(i)
                start += self.stride
        return np.array(windows, dtype=np.float32), np.array(mapping)

    def _norm_stats(self, X: np.ndarray):
        flat = X.reshape(-1, self.window_size)
        self._mean = flat.mean()
        self._std  = flat.std() + 1e-12

    def _normalize(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self._mean) / self._std

    def _to_tensor(self, windows: np.ndarray) -> torch.Tensor:
        # (N, window_size) → (N, window_size, 1)
        t = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, verbose: bool = True) -> "LSTMAutoencoderDetector":
        windows, _ = self._segment(X)
        self._norm_stats(windows)
        wn = self._normalize(windows)

        self.model = LSTMAutoencoder(
            self.window_size, self.hidden_size, self.num_layers
        ).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(self._to_tensor(wn))
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            avg = epoch_loss / len(wn)
            history.append(avg)
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        self.history_ = history
        self.fit_threshold(X, contamination=0.05)
        return self

    def fit_threshold(self, X: np.ndarray, contamination: float = 0.05) -> float:
        scores = self.score(X)
        self.threshold_ = float(np.quantile(scores, 1 - contamination))
        return self.threshold_

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-signal reconstruction error (averaged over windows)."""
        assert self.model is not None, "Call fit() first."
        windows, mapping = self._segment(X)
        wn = self._normalize(windows)
        t = self._to_tensor(wn)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(t)
        errors = ((recon - t) ** 2).mean(dim=(1, 2)).cpu().numpy()

        # aggregate: mean error per original signal
        n_signals = len(X)
        signal_scores = np.zeros(n_signals)
        counts = np.zeros(n_signals)
        for w_idx, sig_idx in enumerate(mapping):
            signal_scores[sig_idx] += errors[w_idx]
            counts[sig_idx] += 1
        return signal_scores / (counts + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generator import MotorCurrentGenerator
    from sklearn.metrics import classification_report

    gen = MotorCurrentGenerator()
    X_raw, y = gen.generate_dataset(n_normal=200, n_fault_each=30)

    X_train = X_raw[y == 0]
    X_test, y_test = X_raw, (y != 0).astype(int)

    print("Training LSTMAutoencoderDetector …")
    det = LSTMAutoencoderDetector(epochs=30, hidden_size=32).fit(X_train)
    preds = det.predict(X_test)
    print(classification_report(y_test, preds, target_names=["normal", "anomaly"]))
