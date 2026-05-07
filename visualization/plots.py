"""
Sensorless Anomaly Detection — Visualisation Utilities

Provides publication-ready plots:
  • plot_signals        — raw current waveforms per class
  • plot_spectrum       — FFT magnitude spectra per class
  • plot_feature_space  — t-SNE of feature vectors
  • plot_roc_pr         — ROC and PR curves for multiple detectors
  • plot_threshold_curve— P/R/F1 vs threshold
  • plot_confusion      — annotated confusion matrix
  • plot_loss_history   — autoencoder training loss
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, List, Optional, Tuple


PALETTE = {
    "normal":       "#2196F3",
    "bearing":      "#F44336",
    "eccentricity": "#FF9800",
    "winding":      "#4CAF50",
}
FAULT_NAMES = ["normal", "bearing", "eccentricity", "winding"]


# ------------------------------------------------------------------
# 1. Raw signal waveforms
# ------------------------------------------------------------------

def plot_signals(
    X: np.ndarray,
    y: np.ndarray,
    fs: float = 10_000,
    n_cycles: float = 3,
    save_path: Optional[str] = None,
):
    """Plot one representative signal per class."""
    f_supply = 50.0
    n_show = int(fs * n_cycles / f_supply)
    t = np.arange(n_show) / fs * 1000  # ms

    n_classes = len(FAULT_NAMES)
    fig, axes = plt.subplots(1, n_classes, figsize=(14, 3), sharey=True)
    fig.suptitle("Motor Current Signals — One Representative per Class",
                 fontsize=13, fontweight="bold")

    for ax, (label_idx, name) in zip(axes, enumerate(FAULT_NAMES)):
        idxs = np.where(y == label_idx)[0]
        if len(idxs) == 0:
            ax.set_visible(False)
            continue
        sig = X[idxs[0], :n_show]
        ax.plot(t, sig, color=PALETTE[name], linewidth=0.8)
        ax.set_title(name.capitalize(), fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_xlim([0, t[-1]])

    axes[0].set_ylabel("Current (A)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 2. FFT magnitude spectra
# ------------------------------------------------------------------

def plot_spectrum(
    X: np.ndarray,
    y: np.ndarray,
    fs: float = 10_000,
    f_max: float = 500.0,
    save_path: Optional[str] = None,
):
    from scipy.fft import rfft, rfftfreq

    n_classes = len(FAULT_NAMES)
    fig, axes = plt.subplots(1, n_classes, figsize=(14, 3), sharey=False)
    fig.suptitle("Current Spectrum — Motor-Current Signature Analysis",
                 fontsize=13, fontweight="bold")

    for ax, (label_idx, name) in zip(axes, enumerate(FAULT_NAMES)):
        idxs = np.where(y == label_idx)[0]
        if len(idxs) == 0:
            ax.set_visible(False)
            continue

        # average magnitude over up to 10 samples
        sigs = X[idxs[:10]]
        N = sigs.shape[1]
        freqs = rfftfreq(N, d=1.0 / fs)
        mags = np.abs(rfft(sigs, axis=1)) / N
        avg_mag = mags.mean(axis=0)

        mask = freqs <= f_max
        ax.plot(freqs[mask], avg_mag[mask], color=PALETTE[name], linewidth=0.9)
        ax.set_title(name.capitalize(), fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_yscale("log")

    axes[0].set_ylabel("Magnitude (A)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 3. t-SNE feature space
# ------------------------------------------------------------------

def plot_feature_space(
    Xf: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(Xf)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_idx, name in enumerate(FAULT_NAMES):
        mask = y == label_idx
        if not mask.any():
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=PALETTE[name], label=name.capitalize(),
            alpha=0.7, s=20, edgecolors="none",
        )
    ax.set_title("t-SNE of Extracted Features", fontsize=13, fontweight="bold")
    ax.legend(markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 4. ROC + PR curves
# ------------------------------------------------------------------

def plot_roc_pr(
    y_true: np.ndarray,
    detector_scores: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ROC and Precision-Recall Curves", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(detector_scores)))
    for (name, scores), c in zip(detector_scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        from sklearn.metrics import auc as sk_auc, roc_auc_score
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc  = sk_auc(rec, prec)

        ax_roc.plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc:.2f})")
        ax_pr.plot(rec, prec, color=c, label=f"{name} (AP={pr_auc:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 5. Threshold operating curve
# ------------------------------------------------------------------

def plot_threshold_curve(
    curve: Dict[str, np.ndarray],
    detector_name: str = "",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["thresholds"], curve["precisions"], label="Precision", color="#2196F3")
    ax.plot(curve["thresholds"], curve["recalls"],    label="Recall",    color="#F44336")
    ax.plot(curve["thresholds"], curve["f1s"],        label="F1",        color="#4CAF50", linewidth=2)

    best_idx = np.argmax(curve["f1s"])
    ax.axvline(curve["thresholds"][best_idx], linestyle="--",
               color="#FF9800", label=f"Best thr={curve['thresholds'][best_idx]:.3f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"P / R / F1 vs Threshold — {detector_name}", fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 6. Confusion matrix
# ------------------------------------------------------------------

def plot_confusion(
    cm: np.ndarray,
    detector_name: str = "",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {detector_name}", fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# 7. Training loss history
# ------------------------------------------------------------------

def plot_loss_history(
    history: List[float],
    model_name: str = "Autoencoder",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss — {model_name}", fontweight="bold")
    ax.set_yscale("log")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
