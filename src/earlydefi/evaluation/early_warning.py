from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def horizon_metrics(scores: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    required = {"model", "horizon_hours", "label", "risk_score", "threshold"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing columns: {sorted(missing)}")

    rows = []
    for (model, horizon), group in scores.groupby(["model", "horizon_hours"], sort=True):
        y_true = group["label"].astype(int).to_numpy()
        y_score = group["risk_score"].astype(float).to_numpy()
        threshold = float(group["threshold"].astype(float).median())
        y_pred = (y_score >= threshold).astype(int)
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        row = {
            "model": model,
            "horizon_hours": int(horizon),
            "n_samples": int(len(group)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "threshold": threshold,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auroc": _safe_auroc(y_true, y_score),
            "auprc": _safe_auprc(y_true, y_score),
        }
        row.update(_precision_recall_at_k(y_true, y_score, k=top_k))
        rows.append(row)
    return pd.DataFrame(rows)


def event_warning_metrics(scores: pd.DataFrame, *, top_k_values: list[int]) -> pd.DataFrame:
    required = {
        "model",
        "entity_id",
        "label",
        "risk_score",
        "threshold",
        "prediction_time",
        "event_time",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing columns: {sorted(missing)}")

    rows = []
    for model, group in scores.groupby("model", sort=True):
        group = group.copy()
        group["label"] = group["label"].astype(int)
        group["risk_score"] = group["risk_score"].astype(float)
        group["threshold"] = group["threshold"].astype(float)
        threshold = float(group["threshold"].median())

        entity_scores = (
            group.groupby("entity_id", as_index=False)
            .agg(label=("label", "max"), risk_score=("risk_score", "max"))
            .sort_values(["risk_score", "entity_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        positive_entities = int(entity_scores["label"].sum())
        normal_entities = int(len(entity_scores) - positive_entities)
        row = {
            "model": model,
            "threshold": threshold,
            "n_entities": int(len(entity_scores)),
            "positive_entities": positive_entities,
            "normal_entities": normal_entities,
        }
        for top_k in top_k_values:
            top = entity_scores.head(min(top_k, len(entity_scores)))
            true_positives = int(top["label"].sum())
            row[f"event_precision_at_{top_k}"] = float(true_positives / len(top)) if len(top) else 0.0
            row[f"event_recall_at_{top_k}"] = (
                float(true_positives / positive_entities) if positive_entities else np.nan
            )

        warning_rows = group[group["risk_score"] >= threshold].copy()
        warned = warning_rows.groupby("entity_id", as_index=False)["label"].max()
        warned_positive = int(warned["label"].sum()) if not warned.empty else 0
        warned_normal = int(len(warned) - warned_positive) if not warned.empty else 0
        row["threshold_warned_positive_entities"] = warned_positive
        row["threshold_warned_normal_entities"] = warned_normal
        row["threshold_event_recall"] = (
            float(warned_positive / positive_entities) if positive_entities else np.nan
        )
        row["threshold_event_precision"] = (
            float(warned_positive / len(warned)) if len(warned) else 0.0
        )

        warning_times = _warning_times_for_positive_entities(warning_rows)
        row["threshold_mean_warning_time_hours"] = (
            float(warning_times.mean()) if len(warning_times) else 0.0
        )
        row["threshold_median_warning_time_hours"] = (
            float(warning_times.median()) if len(warning_times) else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(np.asarray(y_true).sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, *, k: int) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = min(k, len(y_true))
    order = np.argsort(-y_score)[:top_k]
    positives = int(np.asarray(y_true).sum())
    true_positives = int(np.asarray(y_true)[order].sum())
    return {
        f"precision_at_{k}": float(true_positives / top_k) if top_k else 0.0,
        f"recall_at_{k}": float(true_positives / positives) if positives else np.nan,
    }


def _warning_times_for_positive_entities(warning_rows: pd.DataFrame) -> pd.Series:
    positives = warning_rows[warning_rows["label"].eq(1)].copy()
    if positives.empty:
        return pd.Series(dtype=float)
    positives["prediction_time"] = pd.to_datetime(positives["prediction_time"], utc=True, errors="coerce")
    positives["event_time"] = pd.to_datetime(positives["event_time"], utc=True, errors="coerce")
    positives = positives.dropna(subset=["prediction_time", "event_time"])
    if positives.empty:
        return pd.Series(dtype=float)
    first_warning = positives.sort_values("prediction_time").groupby("entity_id", as_index=False).first()
    warning_time = (
        first_warning["event_time"] - first_warning["prediction_time"]
    ).dt.total_seconds() / 3600
    return warning_time.clip(lower=0.0)
