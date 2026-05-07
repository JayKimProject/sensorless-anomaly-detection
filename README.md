# Sensorless Anomaly Detection

> Detect motor faults from electrical current signals — no dedicated physical sensors required.

This portfolio implements a complete anomaly detection pipeline for rotating machinery using **Motor Current Signature Analysis (MCSA)**. Instead of expensive vibration or acoustic sensors, it reads anomaly signatures directly from the motor's phase current — a signal already available in most industrial drives.

---

## What Problem Does This Solve?

Traditional condition monitoring plants accelerometers or microphones on each machine. This portfolio shows that the same fault signatures are **already encoded in the current waveform**:

| Fault Type | Physical Cause | Current Signature |
|---|---|---|
| **Bearing fault** | Race defect → periodic impact | Amplitude modulation at bearing defect freq |
| **Rotor eccentricity** | Air-gap asymmetry | Sidebands around supply frequency |
| **Winding fault** | Shorted turns → harmonic distortion | Odd-harmonic content (3rd, 5th, 7th…) |

---

## Architecture

```
Raw Current Signal  (10 kHz, 1 second window)
        │
        ▼
┌───────────────────┐
│  Feature Extractor │  Time domain (8) + Frequency domain (5) + MCSA harmonics (5)
│  18-dim vector    │
└────────┬──────────┘
         │
    ┌────┴──────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
Statistical                              Deep Learning
─────────────────                        ─────────────────────────────
Z-Score Detector                         Dense Autoencoder
Mahalanobis Distance                       └─ trained on feature vectors
IQR Fence                                LSTM Autoencoder
                                           └─ trained on raw time-series
ML (semi-supervised)
─────────────────
Isolation Forest
One-Class SVM
Local Outlier Factor
         │
         ▼
┌──────────────────┐
│   Evaluator      │  Precision / Recall / F1 / AUC-ROC / PR-AUC
│   Threshold Opt  │  Sweep → best operating point
│   Comparison     │  Ranked table across all detectors
└──────────────────┘
```

All detectors are trained on **normal data only** (one-class / semi-supervised). No fault examples are needed during training.

---

## Repository Structure

```
sensorless_anomaly/
│
├── data/
│   └── generator.py          # Synthetic motor-current signal generator
│                             # Four classes: normal, bearing, eccentricity, winding
│
├── features/
│   └── extractor.py          # 18-dimensional feature extractor
│                             # Time-domain + FFT + MCSA harmonics
│
├── models/
│   ├── statistical.py        # Z-Score, Mahalanobis distance, IQR fence
│   ├── ml_detector.py        # Isolation Forest, One-Class SVM, LOF
│   ├── autoencoder.py        # Dense autoencoder (feature-space)
│   └── lstm_autoencoder.py   # LSTM autoencoder (raw time-series)
│
├── evaluation/
│   └── metrics.py            # Unified evaluation: P/R/F1/AUC, threshold sweep
│
├── visualization/
│   └── plots.py              # Waveforms, spectra, t-SNE, ROC/PR, confusion matrix
│
└── pipeline.py               # End-to-end orchestration (run this)
```

---

## Detectors at a Glance

### Statistical (no training, instant)

| Detector | Method | Best For |
|---|---|---|
| `ZScoreDetector` | Per-feature deviation from mean | Simple baselines, interpretable alarms |
| `MahalanobisDetector` | Covariance-aware distance | Correlated features, Gaussian distributions |
| `IQRDetector` | Interquartile fence | Noisy training data, robust to outliers |

### Machine Learning (scikit-learn)

| Detector | Method | Best For |
|---|---|---|
| `IsolationForestDetector` | Random-cut tree ensemble | High-dimensional, large datasets |
| `OneClassSVMDetector` | Kernel hypersphere | Compact, well-clustered normal class |
| `LOFDetector` | Local density comparison | Varying density, cluster structure |

### Deep Learning (PyTorch)

| Detector | Architecture | Input |
|---|---|---|
| `AutoencoderDetector` | `F → 64 → 32 → 8 → 32 → 64 → F` | 18-dim feature vector |
| `LSTMAutoencoderDetector` | Encoder LSTM → latent → Decoder LSTM | Raw current window (512 samples) |

Anomaly score = **reconstruction error** — normal signals reconstruct well; faults do not.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fast smoke-test (~30 seconds, small dataset)
python pipeline.py --quick

# 3. Full run with all detectors + figures saved to outputs/
python pipeline.py
```

**Expected output (full run):**

```
Detector                        Prec     Rec      F1   AUC-ROC      AP
----------------------------------------------------------------------
ZScoreDetector                 1.000   1.000   1.000     1.000   1.000
MahalanobisDetector            1.000   1.000   1.000     1.000   1.000
OneClassSVMDetector            1.000   1.000   1.000     1.000   1.000
IsolationForestDetector        0.957   1.000   0.978     0.997   0.993
Dense Autoencoder              0.940   0.967   0.953     0.991   0.982
LSTM Autoencoder               0.921   0.956   0.938     0.985   0.974
LOFDetector                    0.880   0.911   0.895     0.963   0.951
IQRDetector                    0.708   0.378   0.493     0.821   0.687
----------------------------------------------------------------------
```

---

## How Each Module Works

### `data/generator.py` — Signal Generator

Generates analytic motor-current signals for four conditions:

```python
from data.generator import MotorCurrentGenerator

gen = MotorCurrentGenerator()

signal = gen.generate("normal")       # clean sinusoid + noise
signal = gen.generate("bearing")      # amplitude-modulated at BPFO
signal = gen.generate("eccentricity") # supply-frequency sidebands
signal = gen.generate("winding")      # odd harmonic distortion

X, y = gen.generate_dataset(n_normal=600, n_fault_each=80)
# X: (N, 10000)  y: (N,) with labels 0-3
```

### `features/extractor.py` — Feature Extractor

Converts a raw 1-D signal into an 18-dimensional vector:

| Group | Features (count) | Examples |
|---|---|---|
| Time domain | 8 | RMS, kurtosis, crest factor, skewness |
| Frequency domain | 5 | Spectral centroid, entropy, spread |
| MCSA harmonics | 5 | Amplitude at 1×, 3×, 5×, 7×, 9× supply freq |

```python
from features.extractor import FeatureExtractor

ext = FeatureExtractor()
feature_vector = ext.extract(signal)        # shape: (18,)
feature_matrix = ext.extract_batch(X)       # shape: (N, 18)
```

### `models/` — Detectors

All detectors share the same interface:

```python
det = IsolationForestDetector()
det.fit(X_train)              # train on normal data only
scores = det.score(X_test)    # anomaly score per sample (higher = worse)
labels = det.predict(X_test)  # binary: 1 = anomaly, 0 = normal
```

### `evaluation/metrics.py` — Evaluation

```python
from evaluation.metrics import evaluate_detector, compare_detectors

result = evaluate_detector("MyDetector", y_true, scores)
print(result)
# MyDetector   P=0.940  R=0.967  F1=0.953  AUC=0.991  AP=0.982

print(compare_detectors([result1, result2, result3]))
```

### `visualization/plots.py` — Plots

Seven plot types, each callable with an optional `save_path`:

```python
from visualization.plots import (
    plot_signals,        # raw current waveforms per class
    plot_spectrum,       # FFT magnitude spectra (MCSA view)
    plot_feature_space,  # t-SNE of feature vectors
    plot_roc_pr,         # ROC + PR curves for multiple detectors
    plot_threshold_curve,# Precision / Recall / F1 vs threshold
    plot_confusion,      # annotated confusion matrix
    plot_loss_history,   # autoencoder training curve
)

plot_signals(X, y, save_path="outputs/signals.png")
```

---

## Design Principles

**Semi-supervised** — all detectors train on normal data only. This matches real industrial deployments where fault examples are rare or nonexistent.

**Layered approach** — from zero-parameter statistical rules to deep sequence models. Use the simplest detector that meets your accuracy requirement.

**Physics-informed features** — MCSA harmonics encode domain knowledge (supply frequency, harmonic orders) that pure data-driven features would need large datasets to discover.

**Modular** — each layer (data, features, models, evaluation, visualisation) is independently importable. Swap in your own signal source or model with minimal changes.

---

## Requirements

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
torch>=2.0
matplotlib>=3.7
```

Python 3.9+ recommended. GPU optional (PyTorch falls back to CPU automatically).
