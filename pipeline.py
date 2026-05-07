"""
Sensorless Anomaly Detection — End-to-End Pipeline

Runs the complete workflow:
  1. Generate synthetic motor-current dataset
  2. Extract hand-crafted features
  3. Train all detectors (statistical, ML, deep learning)
  4. Evaluate and compare
  5. Save visualisations to ./outputs/

Usage
-----
  python pipeline.py              # full run
  python pipeline.py --quick      # smaller dataset for fast smoke-test
  python pipeline.py --no-plots   # skip figure generation
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from data.generator import MotorCurrentGenerator, SignalConfig
from features.extractor import FeatureExtractor
from models.statistical import ZScoreDetector, MahalanobisDetector, IQRDetector
from models.ml_detector import IsolationForestDetector, OneClassSVMDetector, LOFDetector
from models.autoencoder import AutoencoderDetector
from models.lstm_autoencoder import LSTMAutoencoderDetector
from evaluation.metrics import evaluate_detector, compare_detectors, threshold_curve
from visualization.plots import (
    plot_signals, plot_spectrum, plot_feature_space,
    plot_roc_pr, plot_threshold_curve, plot_confusion, plot_loss_history,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_output_dir() -> Path:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    return out


def split_train_test(Xf, X_raw, y, train_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    normal_idx  = np.where(y == 0)[0]
    anomaly_idx = np.where(y != 0)[0]

    rng.shuffle(normal_idx)
    n_train = int(len(normal_idx) * train_ratio)
    train_idx = normal_idx[:n_train]
    test_idx  = np.concatenate([normal_idx[n_train:], anomaly_idx])

    return (
        Xf[train_idx],       # feature train (normal only)
        X_raw[train_idx],    # raw train (normal only)
        Xf[test_idx],        # feature test
        X_raw[test_idx],     # raw test
        (y[test_idx] != 0).astype(int),  # binary label (1=anomaly)
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(quick: bool = False, no_plots: bool = False):
    out = make_output_dir()

    # ── 1. Generate dataset ──────────────────────────────────────
    print("\n[1/5] Generating motor-current dataset …")
    n_normal     = 200 if quick else 600
    n_fault_each = 20  if quick else 80

    gen = MotorCurrentGenerator(SignalConfig(random_seed=42))
    X_raw, y = gen.generate_dataset(n_normal=n_normal, n_fault_each=n_fault_each)
    print(f"      Total samples : {len(y)}  "
          f"(normal={( y==0).sum()}  faults={(y!=0).sum()})")

    # ── 2. Extract features ──────────────────────────────────────
    print("\n[2/5] Extracting features …")
    ext = FeatureExtractor()
    Xf  = ext.extract_batch(X_raw)
    print(f"      Feature matrix : {Xf.shape}")

    (Xf_train, Xr_train,
     Xf_test,  Xr_test, y_test) = split_train_test(Xf, X_raw, y)

    # ── 3. Visualise data ────────────────────────────────────────
    if not no_plots:
        print("\n[3/5] Generating data visualisations …")
        plot_signals(X_raw, y, save_path=str(out / "signals.png"))
        plot_spectrum(X_raw, y, save_path=str(out / "spectrum.png"))
        plot_feature_space(Xf, y, save_path=str(out / "tsne.png"))
    else:
        print("\n[3/5] Skipping plots (--no-plots).")

    # ── 4. Train and evaluate detectors ─────────────────────────
    print("\n[4/5] Training detectors …")
    results = []
    all_scores = {}

    def run_detector(name, det, fit_data, score_data):
        t0 = time.time()
        det.fit(fit_data)
        elapsed = time.time() - t0

        if hasattr(det, "score"):
            scores = det.score(score_data)
        else:
            scores = det.predict(score_data).astype(float)

        result = evaluate_detector(name, y_test, scores)
        results.append(result)
        all_scores[name] = scores
        print(f"      {result}  [{elapsed:.1f}s]")

        if not no_plots:
            plot_confusion(result.confusion, name,
                           save_path=str(out / f"confusion_{name.replace(' ','_')}.png"))
        return result

    # Statistical
    run_detector("Z-Score",      ZScoreDetector().fit(Xf_train),
                 Xf_train, Xf_test)
    run_detector("Mahalanobis",  MahalanobisDetector().fit(Xf_train),
                 Xf_train, Xf_test)
    run_detector("IQR",          IQRDetector().fit(Xf_train),
                 Xf_train, Xf_test)

    # ML (re-fit inside run_detector via the fit() call)
    run_detector("Isolation Forest",
                 IsolationForestDetector(), Xf_train, Xf_test)
    run_detector("One-Class SVM",
                 OneClassSVMDetector(),     Xf_train, Xf_test)
    run_detector("LOF",
                 LOFDetector(),             Xf_train, Xf_test)

    # Dense autoencoder
    print("      [Dense AE] training …")
    ae_det = AutoencoderDetector(
        latent_dim=6, epochs=50 if quick else 120, verbose=False
    )
    run_detector("Dense Autoencoder", ae_det, Xf_train, Xf_test)
    if not no_plots:
        plot_loss_history(ae_det.history_, "Dense Autoencoder",
                          save_path=str(out / "loss_dense_ae.png"))

    # LSTM autoencoder (raw signals)
    print("      [LSTM AE] training …")
    lstm_det = LSTMAutoencoderDetector(
        epochs=20 if quick else 40, hidden_size=32, verbose=False
    )
    lstm_scores_train = None   # score() uses internal segments
    t0 = time.time()
    lstm_det.fit(Xr_train)
    elapsed = time.time() - t0
    lstm_scores = lstm_det.score(Xr_test)
    lstm_result = evaluate_detector("LSTM Autoencoder", y_test, lstm_scores)
    results.append(lstm_result)
    all_scores["LSTM Autoencoder"] = lstm_scores
    print(f"      {lstm_result}  [{elapsed:.1f}s]")
    if not no_plots:
        plot_loss_history(lstm_det.history_, "LSTM Autoencoder",
                          save_path=str(out / "loss_lstm_ae.png"))
        plot_confusion(lstm_result.confusion, "LSTM Autoencoder",
                       save_path=str(out / "confusion_LSTM_Autoencoder.png"))

    # ── 5. Summary ───────────────────────────────────────────────
    print("\n[5/5] Results summary\n")
    print(compare_detectors(results))

    if not no_plots:
        plot_roc_pr(y_test, all_scores,
                    save_path=str(out / "roc_pr.png"))

        # Threshold curve for best detector
        best = max(results, key=lambda r: r.f1)
        thr_curve = threshold_curve(y_test, all_scores[best.name])
        plot_threshold_curve(thr_curve, best.name,
                             save_path=str(out / "threshold_curve.png"))

        print(f"\nAll figures saved to {out}/")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensorless Anomaly Detection Pipeline"
    )
    parser.add_argument("--quick",    action="store_true",
                        help="Use a smaller dataset for rapid testing")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating visualisation figures")
    args = parser.parse_args()
    run(quick=args.quick, no_plots=args.no_plots)
