"""
src/utils/metrics.py

Unsupervised PHM (Prognostics and Health Management) evaluation metrics,
threshold selection, RUL (Remaining Useful Life) at first sustained alarm,
and physical explainability summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def select_threshold(
    train_scores: np.ndarray,
    val_scores: np.ndarray | None = None,
    strategy: str = "percentile",
    percentile: float = 99.0,
) -> float:
    """
    Select anomaly threshold without ever looking at the test set.

    Parameters
    ----------
    train_scores : np.ndarray
        Anomaly / reconstruction error scores on the healthy training set.
    val_scores : np.ndarray | None
        Anomaly scores on the healthy validation set.
    strategy : str
        "percentile" (default: 99th percentile of train scores) or "val_percentile".
    percentile : float
        Percentile value in [0, 100].

    Returns
    -------
    float
        Threshold value.
    """
    if strategy == "percentile":
        return float(np.percentile(train_scores, percentile))
    elif strategy == "val_percentile":
        ref = val_scores if val_scores is not None else train_scores
        return float(np.percentile(ref, percentile))
    elif strategy == "val_max":
        ref = val_scores if val_scores is not None else train_scores
        return float(np.max(ref))
    else:
        raise ValueError(f"Unknown threshold strategy: '{strategy}'")


def find_sustained_alarm(
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sustained_window: int = 5,
    end_of_life_idx: int = 981,
    interval_minutes: float = 10.0,
) -> dict[str, Any]:
    """
    Find the first sustained alarm (N consecutive files above threshold)
    and compute Remaining Useful Life (RUL / Detection Lead Time) at that instant.

    Parameters
    ----------
    indices : np.ndarray
        File indices of the test split.
    scores : np.ndarray
        Anomaly scores corresponding to *indices*.
    threshold : float
        Anomaly threshold.
    sustained_window : int
        Number of consecutive files above threshold required to confirm an alarm.
    end_of_life_idx : int
        Final active snapshot index at failure (981 for 2nd_test).
    interval_minutes : float
        Nominal sampling interval in minutes (10.0 min).

    Returns
    -------
    dict with:
      - triggered (bool)
      - first_alarm_idx (int or None): index where the sustained sequence started
      - confirmed_alarm_idx (int or None): index where the N-th file confirmed the alarm
      - detection_lead_files (int): RUL in files from first_alarm_idx to end_of_life_idx
      - detection_lead_hours (float): RUL in hours from first_alarm_idx to end_of_life_idx
    """
    above = (scores > threshold).astype(int)
    n = len(above)

    for i in range(n - sustained_window + 1):
        if np.all(above[i : i + sustained_window] == 1):
            onset_idx = int(indices[i])
            confirm_idx = int(indices[i + sustained_window - 1])
            lead_files = max(0, int(end_of_life_idx - onset_idx))
            lead_hours = float(lead_files * interval_minutes / 60.0)
            return {
                "triggered": True,
                "first_alarm_idx": onset_idx,
                "confirmed_alarm_idx": confirm_idx,
                "detection_lead_files": lead_files,
                "detection_lead_hours": round(lead_hours, 2),
            }

    return {
        "triggered": False,
        "first_alarm_idx": None,
        "confirmed_alarm_idx": None,
        "detection_lead_files": 0,
        "detection_lead_hours": 0.0,
    }


def compute_phm_metrics(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    idx_test: np.ndarray,
    threshold: float,
    sustained_window: int = 5,
    end_of_life_idx: int = 981,
    interval_minutes: float = 10.0,
) -> dict[str, Any]:
    """
    Compute the full suite of unsupervised PHM anomaly detection metrics.
    """
    # 1. Detection rate on degradation zone (Recall) and False Alarm Rate on healthy validation
    recall = float(np.mean(test_scores > threshold))
    false_alarm_rate = float(np.mean(val_scores > threshold))
    train_exceed_rate = float(np.mean(train_scores > threshold))

    # 2. First sustained alarm & Remaining Useful Life (RUL)
    alarm_info = find_sustained_alarm(
        indices=idx_test,
        scores=test_scores,
        threshold=threshold,
        sustained_window=sustained_window,
        end_of_life_idx=end_of_life_idx,
        interval_minutes=interval_minutes,
    )

    # 3. Threshold-independent separation metrics
    val_mean = float(np.mean(val_scores))
    test_mean = float(np.mean(test_scores))
    separation_ratio = float(test_mean / (val_mean + 1e-12))

    # Spearman rank correlation of anomaly score with time during degradation
    corr, _ = spearmanr(idx_test, test_scores)
    spearman_monotonicity = float(0.0 if np.isnan(corr) else corr)

    # Mann-Whitney U / ROC-AUC between healthy validation (0) and degradation test (1)
    y_eval = np.concatenate([np.zeros(len(val_scores), dtype=int), np.ones(len(test_scores), dtype=int)])
    s_eval = np.concatenate([val_scores, test_scores])
    roc_auc = float(roc_auc_score(y_eval, s_eval))

    return {
        "recall": round(recall, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "train_exceed_rate": round(train_exceed_rate, 4),
        "first_alarm_idx": alarm_info["first_alarm_idx"],
        "confirmed_alarm_idx": alarm_info["confirmed_alarm_idx"],
        "detection_lead_files": alarm_info["detection_lead_files"],
        "detection_lead_hours": alarm_info["detection_lead_hours"],
        "separation_ratio": round(separation_ratio, 4),
        "spearman_monotonicity": round(spearman_monotonicity, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold_value": float(threshold),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "val_score_mean": val_mean,
        "val_score_std": float(np.std(val_scores)),
        "test_score_mean": test_mean,
        "test_score_max": float(np.max(test_scores)),
    }

