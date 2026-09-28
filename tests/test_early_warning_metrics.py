from __future__ import annotations

import pandas as pd

from earlydefi.evaluation.early_warning import event_warning_metrics, horizon_metrics


def test_horizon_metrics_handles_single_class_horizon() -> None:
    scores = pd.DataFrame(
        {
            "model": ["m", "m", "m", "m"],
            "horizon_hours": [1, 1, 6, 6],
            "label": [1, 0, 0, 0],
            "risk_score": [0.9, 0.1, 0.2, 0.3],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
    )

    out = horizon_metrics(scores, top_k=1)

    h1 = out[out["horizon_hours"].eq(1)].iloc[0]
    h6 = out[out["horizon_hours"].eq(6)].iloc[0]
    assert h1["precision_at_1"] == 1.0
    assert h1["recall_at_1"] == 1.0
    assert pd.isna(h6["auroc"])
    assert pd.isna(h6["auprc"])


def test_event_warning_metrics_reports_entity_recall_and_warning_time() -> None:
    scores = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "entity_id": ["p1", "p1", "p2", "n1", "n1"],
            "label": [1, 1, 1, 0, 0],
            "risk_score": [0.7, 0.9, 0.4, 0.95, 0.2],
            "threshold": [0.8] * 5,
            "prediction_time": pd.to_datetime(
                [
                    "2024-01-01T18:00:00Z",
                    "2024-01-01T23:00:00Z",
                    "2024-01-02T23:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-03T01:00:00Z",
                ]
            ),
            "event_time": [
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                pd.NA,
                pd.NA,
            ],
        }
    )

    out = event_warning_metrics(scores, top_k_values=[1, 2])
    row = out.iloc[0]

    assert row["event_precision_at_1"] == 0.0
    assert row["event_recall_at_2"] == 0.5
    assert row["threshold_event_recall"] == 0.5
    assert row["threshold_event_precision"] == 0.5
    assert row["threshold_mean_warning_time_hours"] == 1.0
