from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute standard binary metrics with AUPRC as the main imbalanced metric."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
    return metrics


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict[str, float]:
    """Compute precision and recall among the top-k risk scores."""
    if k <= 0:
        raise ValueError("k must be positive")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top_k = min(k, len(y_true))
    order = np.argsort(-y_score)[:top_k]
    positives = y_true.sum()
    true_positives_at_k = y_true[order].sum()

    return {
        f"precision_at_{k}": float(true_positives_at_k / top_k),
        f"recall_at_{k}": float(true_positives_at_k / positives) if positives else 0.0,
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return the threshold that maximizes F1 on a validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def mean_warning_time(
    scored_samples: pd.DataFrame,
    *,
    score_threshold: float,
    entity_col: str = "entity_id",
    score_col: str = "risk_score",
) -> float:
    """Average time between first warning and event among true positive entities, in hours."""
    required = {entity_col, "prediction_time", "event_time", "label", score_col}
    missing = required - set(scored_samples.columns)
    if missing:
        raise ValueError(f"scored_samples is missing columns: {sorted(missing)}")

    positives = scored_samples[
        (scored_samples["label"] == 1) & (scored_samples[score_col] >= score_threshold)
    ].copy()
    if positives.empty:
        return 0.0

    positives["prediction_time"] = pd.to_datetime(positives["prediction_time"], utc=True)
    positives["event_time"] = pd.to_datetime(positives["event_time"], utc=True)
    first_warnings = positives.sort_values("prediction_time").groupby(entity_col, as_index=False).first()
    warning_hours = (
        first_warnings["event_time"] - first_warnings["prediction_time"]
    ).dt.total_seconds() / 3600
    return float(warning_hours.clip(lower=0).mean())
