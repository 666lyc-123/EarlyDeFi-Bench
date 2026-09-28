from __future__ import annotations

import pandas as pd


def latest_event_features(
    events: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    lookback_hours: int,
    event_types: list[str],
) -> pd.DataFrame:
    """Build simple event-count features within a lookback window before prediction time."""
    required_events = {"entity_id", "event_type", "observed_at"}
    required_samples = {"sample_id", "entity_id", "prediction_time"}
    missing_events = required_events - set(events.columns)
    missing_samples = required_samples - set(samples.columns)
    if missing_events:
        raise ValueError(f"events is missing columns: {sorted(missing_events)}")
    if missing_samples:
        raise ValueError(f"samples is missing columns: {sorted(missing_samples)}")

    events = events.copy()
    samples = samples.copy()
    events["observed_at"] = pd.to_datetime(events["observed_at"], utc=True)
    samples["prediction_time"] = pd.to_datetime(samples["prediction_time"], utc=True)

    rows = []
    lookback = pd.Timedelta(hours=lookback_hours)
    for sample in samples.itertuples(index=False):
        window_start = sample.prediction_time - lookback
        entity_events = events[
            (events["entity_id"] == sample.entity_id)
            & (events["observed_at"] <= sample.prediction_time)
            & (events["observed_at"] > window_start)
        ]
        for event_type in event_types:
            matched = entity_events[entity_events["event_type"] == event_type]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "feature_name": f"count_{event_type}_{lookback_hours}h",
                    "feature_value": float(len(matched)),
                    "feature_observed_at": matched["observed_at"].max()
                    if not matched.empty
                    else sample.prediction_time,
                }
            )
    return pd.DataFrame(rows)
