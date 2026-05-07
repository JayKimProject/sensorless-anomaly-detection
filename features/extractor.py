"""
Sensorless Anomaly Detection — Feature Extractor

Converts raw current signals into a compact feature vector covering:
  • Time-domain statistics
  • Frequency-domain (FFT) statistics
  • Spectral entropy
  • Motor-current signature analysis (MCSA) harmonics
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExtractorConfig:
    fs: float = 10_000
    f_supply: float = 50.0
    harmonic_orders: List[int] = None   # which harmonics to extract

    def __post_init__(self):
        if self.harmonic_orders is None:
            self.harmonic_orders = [1, 3, 5, 7, 9]  # odd harmonics


class FeatureExtractor:
    """
    Transform a 1-D current signal into a fixed-length feature vector.

    Feature groups
    --------------
    Time domain (8)    : mean, std, rms, peak, crest factor,
                         kurtosis, skewness, shape factor
    Frequency domain (5): spectral centroid, spectral spread,
                          spectral skewness, spectral kurtosis,
                          spectral entropy
    MCSA harmonics (k) : amplitude at each harmonic of f_supply
    """

    def __init__(self, cfg: ExtractorConfig = ExtractorConfig()):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------

    def _time_features(self, x: np.ndarray) -> Dict[str, float]:
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        return {
            "td_mean":         float(np.mean(x)),
            "td_std":          float(np.std(x)),
            "td_rms":          float(rms),
            "td_peak":         float(peak),
            "td_crest_factor": float(peak / (rms + 1e-12)),
            "td_kurtosis":     float(kurtosis(x)),
            "td_skewness":     float(skew(x)),
            "td_shape_factor": float(rms / (np.mean(np.abs(x)) + 1e-12)),
        }

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------

    def _freq_features(
        self, x: np.ndarray
    ) -> Dict[str, float]:
        N = len(x)
        magnitudes = np.abs(rfft(x)) / N
        freqs = rfftfreq(N, d=1.0 / self.cfg.fs)

        # avoid DC
        magnitudes[0] = 0.0

        total_power = np.sum(magnitudes ** 2) + 1e-12
        prob = magnitudes ** 2 / total_power

        centroid = float(np.sum(freqs * prob))
        spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * prob)))

        # spectral skewness & kurtosis (distribution moments over freq axis)
        norm = spread + 1e-12
        sp_skew = float(np.sum(((freqs - centroid) / norm) ** 3 * prob))
        sp_kurt = float(np.sum(((freqs - centroid) / norm) ** 4 * prob))

        # spectral entropy
        entropy = float(-np.sum(prob * np.log(prob + 1e-12)))

        return {
            "fd_centroid":       centroid,
            "fd_spread":         spread,
            "fd_skewness":       sp_skew,
            "fd_kurtosis":       sp_kurt,
            "fd_entropy":        entropy,
        }

    # ------------------------------------------------------------------
    # MCSA (Motor Current Signature Analysis) harmonics
    # ------------------------------------------------------------------

    def _mcsa_features(self, x: np.ndarray) -> Dict[str, float]:
        N = len(x)
        magnitudes = np.abs(rfft(x)) / N
        freqs = rfftfreq(N, d=1.0 / self.cfg.fs)
        freq_resolution = self.cfg.fs / N

        features = {}
        for order in self.cfg.harmonic_orders:
            target = order * self.cfg.f_supply
            # find closest FFT bin
            bin_idx = int(round(target / freq_resolution))
            bin_idx = min(bin_idx, len(magnitudes) - 1)
            features[f"mcsa_h{order}"] = float(magnitudes[bin_idx])

        return features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, x: np.ndarray) -> np.ndarray:
        """Return a 1-D feature vector for a single signal."""
        feats: Dict[str, float] = {}
        feats.update(self._time_features(x))
        feats.update(self._freq_features(x))
        feats.update(self._mcsa_features(x))
        return np.array(list(feats.values()), dtype=np.float32)

    def extract_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised extraction for a signal matrix (N, L) → (N, F)."""
        return np.stack([self.extract(x) for x in X])

    @property
    def feature_names(self) -> List[str]:
        dummy = np.zeros(int(self.cfg.fs))
        keys: Dict[str, float] = {}
        keys.update(self._time_features(dummy))
        keys.update(self._freq_features(dummy))
        keys.update(self._mcsa_features(dummy))
        return list(keys.keys())


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generator import MotorCurrentGenerator

    gen = MotorCurrentGenerator()
    X, y = gen.generate_dataset(n_normal=10, n_fault_each=3)

    ext = FeatureExtractor()
    Xf = ext.extract_batch(X)

    print(f"Raw signal shape    : {X.shape}")
    print(f"Feature matrix shape: {Xf.shape}")
    print(f"Feature names ({len(ext.feature_names)}): {ext.feature_names}")
