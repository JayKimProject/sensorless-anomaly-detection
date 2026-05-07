"""
Sensorless Anomaly Detection — Signal Generator

Simulates motor phase-current signals for normal operation and
three fault classes: bearing fault, rotor eccentricity, winding fault.
No dedicated physical sensor is required; signals are derived from
readily available electrical measurements (current/voltage).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SignalConfig:
    fs: float = 10_000       # sampling rate (Hz)
    duration: float = 1.0    # seconds per sample
    f_supply: float = 50.0   # supply frequency (Hz)
    f_rotor: float = 47.5    # rotor frequency (Hz)
    amplitude: float = 5.0   # nominal current amplitude (A)
    noise_std: float = 0.05  # Gaussian noise level
    random_seed: int = 42


@dataclass
class FaultParams:
    bearing_defect_freq: float = 105.0   # BPFO (Hz)
    eccentricity_freq: float = 2.5       # sideband spacing (Hz)
    winding_harmonic: int = 3            # dominant harmonic order
    fault_amplitude: float = 0.15       # fault injection strength


class MotorCurrentGenerator:
    """
    Generates three-phase motor current signals using analytical models.

    Normal signal  : I(t) = A·sin(2π·f_s·t) + noise
    Bearing fault  : adds amplitude modulation at bearing defect freq
    Eccentricity   : adds sidebands around supply frequency
    Winding fault  : adds odd harmonic distortion
    """

    FAULT_LABELS = {
        "normal": 0,
        "bearing": 1,
        "eccentricity": 2,
        "winding": 3,
    }

    def __init__(self, cfg: SignalConfig = SignalConfig(),
                 fault: FaultParams = FaultParams()):
        self.cfg = cfg
        self.fault = fault
        self.rng = np.random.default_rng(cfg.random_seed)
        self.n_samples = int(cfg.fs * cfg.duration)
        self.t = np.linspace(0, cfg.duration, self.n_samples, endpoint=False)

    # ------------------------------------------------------------------
    # Core builders
    # ------------------------------------------------------------------

    def _base_signal(self, phase_deg: float = 0.0) -> np.ndarray:
        phase = np.deg2rad(phase_deg)
        return self.cfg.amplitude * np.sin(
            2 * np.pi * self.cfg.f_supply * self.t + phase
        )

    def _noise(self) -> np.ndarray:
        return self.rng.normal(0, self.cfg.noise_std, self.n_samples)

    def _bearing_fault(self) -> np.ndarray:
        mod = 1 + self.fault.fault_amplitude * np.cos(
            2 * np.pi * self.fault.bearing_defect_freq * self.t
        )
        return self._base_signal() * mod + self._noise()

    def _eccentricity_fault(self) -> np.ndarray:
        f_sb1 = self.cfg.f_supply + self.fault.eccentricity_freq
        f_sb2 = self.cfg.f_supply - self.fault.eccentricity_freq
        sideband = self.fault.fault_amplitude * (
            np.sin(2 * np.pi * f_sb1 * self.t)
            + np.sin(2 * np.pi * f_sb2 * self.t)
        )
        return self._base_signal() + sideband + self._noise()

    def _winding_fault(self) -> np.ndarray:
        h = self.fault.winding_harmonic
        harmonic = self.fault.fault_amplitude * np.sin(
            2 * np.pi * h * self.cfg.f_supply * self.t
        )
        return self._base_signal() + harmonic + self._noise()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, fault_type: str = "normal") -> np.ndarray:
        """Return a single 1-D current signal for the requested fault type."""
        builders = {
            "normal": lambda: self._base_signal() + self._noise(),
            "bearing": self._bearing_fault,
            "eccentricity": self._eccentricity_fault,
            "winding": self._winding_fault,
        }
        if fault_type not in builders:
            raise ValueError(f"Unknown fault type '{fault_type}'. "
                             f"Choose from {list(builders)}")
        return builders[fault_type]()

    def generate_dataset(
        self,
        n_normal: int = 500,
        n_fault_each: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a labelled dataset.

        Returns
        -------
        X : (N, n_samples)  raw current signals
        y : (N,)            integer labels (0=normal, 1-3=fault classes)
        """
        signals, labels = [], []

        for _ in range(n_normal):
            signals.append(self.generate("normal"))
            labels.append(self.FAULT_LABELS["normal"])

        for fault_type, label in self.FAULT_LABELS.items():
            if fault_type == "normal":
                continue
            for _ in range(n_fault_each):
                signals.append(self.generate(fault_type))
                labels.append(label)

        X = np.stack(signals)
        y = np.array(labels)

        # shuffle
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    gen = MotorCurrentGenerator()
    X, y = gen.generate_dataset()
    print(f"Dataset shape : {X.shape}")
    print(f"Label counts  : { {k: int((y == v).sum()) for k, v in gen.FAULT_LABELS.items()} }")
