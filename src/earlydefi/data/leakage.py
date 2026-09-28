from __future__ import annotations

import pandas as pd


REQUIRED_SAMPLE_COLUMNS = {"sample_id", "prediction_time"}
REQUIRED_FEATURE_COLUMNS = {"sample_id", "feature_name", "feature_observed_at"}


def validate_feature_visibility(samples: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return feature rows that violate prediction-time visibility."""
    missing_samples = REQUIRED_SAMPLE_COLUMNS - set(samples.columns)
    missing_features = REQUIRED_FEATURE_COLUMNS - set(features.columns)
    if missing_samples:
        raise ValueError(f"samples is missing columns: {sorted(missing_samples)}")
    if missing_features:
        raise ValueError(f"features is missing columns: {sorted(missing_features)}")

    merged = features.merge(
        samples[["sample_id", "prediction_time"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if merged["prediction_time"].isna().any():
        missing = merged.loc[merged["prediction_time"].isna(), "sample_id"].unique()
        raise ValueError(f"features reference unknown sample_id values: {missing[:10]}")

    return merged.loc[merged["feature_observed_at"] > merged["prediction_time"]].copy()


def assert_no_feature_leakage(samples: pd.DataFrame, features: pd.DataFrame) -> None:
    """Raise when any feature uses information after prediction time."""
    violations = validate_feature_visibility(samples, features)
    if not violations.empty:
        preview = violations[
            ["sample_id", "feature_name", "feature_observed_at", "prediction_time"]
        ].head(10)
        raise ValueError(f"Found feature leakage:\n{preview.to_string(index=False)}")


def exclude_post_event_samples(samples: pd.DataFrame, event_time_col: str = "event_time") -> pd.DataFrame:
    """Remove rows where the target event already happened by prediction time."""
    required = {"prediction_time", event_time_col}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing columns: {sorted(missing)}")

    has_event = samples[event_time_col].notna()
    valid = ~has_event | (samples["prediction_time"] < samples[event_time_col])
    return samples.loc[valid].copy()
