"""
Sensorless Anomaly Detection — Evaluation Metrics

Provides a unified evaluation harness:
  • Binary classification metrics (precision, recall, F1, AUC-ROC)
  • Threshold sweep → optimal F1 operating point
  • Comparative report across multiple detectors
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


@dataclass
class EvalResult:
    name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray
    threshold: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"{self.name:<28s} "
            f"P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}  AUC={self.roc_auc:.3f}  "
            f"AP={self.pr_auc:.3f}"
        )


def evaluate_detector(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> EvalResult:
    """
    Compute all metrics for a single detector.

    Parameters
    ----------
    y_true    : binary ground-truth (1 = anomaly)
    scores    : continuous anomaly scores (higher = more anomalous)
    threshold : if None, optimise F1 over a sweep
    """
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, scores)

    preds = (scores > threshold).astype(int)

    return EvalResult(
        name=name,
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        confusion=confusion_matrix(y_true, preds),
        threshold=float(threshold),
    )


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_steps: int = 200,
) -> Tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_f1)."""
    lo, hi = scores.min(), scores.max()
    candidates = np.linspace(lo, hi, n_steps)
    best_f1, best_thr = 0.0, candidates[0]
    for thr in candidates:
        preds = (scores > thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def compare_detectors(results: List[EvalResult]) -> str:
    """Format a comparison table for all evaluated detectors."""
    header = (
        f"{'Detector':<28}  {'Prec':>6}  {'Rec':>6}  "
        f"{'F1':>6}  {'AUC-ROC':>8}  {'AP':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in sorted(results, key=lambda x: -x.f1):
        rows.append(
            f"{r.name:<28}  {r.precision:6.3f}  {r.recall:6.3f}  "
            f"{r.f1:6.3f}  {r.roc_auc:8.3f}  {r.pr_auc:6.3f}"
        )
    rows.append(sep)
    return "\n".join(rows)


def threshold_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_steps: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Return precision, recall, F1 as a function of threshold.
    Useful for plotting the operating-point curve.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))
    return {
        "thresholds":  thresholds,
        "precisions":  np.array(precisions),
        "recalls":     np.array(recalls),
        "f1s":         np.array(f1s),
    }


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 100 + [1] * 20)
    scores_good = np.where(y_true == 1,
                           rng.normal(0.8, 0.1, len(y_true)),
                           rng.normal(0.2, 0.1, len(y_true)))
    scores_bad  = rng.uniform(0, 1, len(y_true))

    results = [
        evaluate_detector("GoodDetector", y_true, scores_good),
        evaluate_detector("RandomDetector", y_true, scores_bad),
    ]
    print(compare_detectors(results))
    for r in results:
        print(f"\nConfusion matrix — {r.name}")
        print(r.confusion)
