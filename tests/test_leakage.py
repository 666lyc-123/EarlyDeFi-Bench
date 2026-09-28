from __future__ import annotations

import pandas as pd
import pytest

from earlydefi.data.leakage import assert_no_feature_leakage, validate_feature_visibility


def test_validate_feature_visibility_detects_future_feature() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prediction_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "feature_name": ["future_liquidity"],
            "feature_observed_at": [pd.Timestamp("2024-01-01T01:00:00Z")],
        }
    )

    violations = validate_feature_visibility(samples, features)

    assert len(violations) == 1


def test_assert_no_feature_leakage_accepts_past_feature() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prediction_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "feature_name": ["past_liquidity"],
            "feature_observed_at": [pd.Timestamp("2023-12-31T23:59:00Z")],
        }
    )

    assert_no_feature_leakage(samples, features)


def test_assert_no_feature_leakage_raises_on_future_feature() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prediction_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "feature_name": ["future_liquidity"],
            "feature_observed_at": [pd.Timestamp("2024-01-01T00:00:01Z")],
        }
    )

    with pytest.raises(ValueError, match="Found feature leakage"):
        assert_no_feature_leakage(samples, features)
