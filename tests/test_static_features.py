from __future__ import annotations

import pandas as pd

from earlydefi.data.static_features import append_opcode_features


def test_append_opcode_features_adds_static_rows(tmp_path) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "token_address": ["0xabc"],
            "prediction_time": [pd.Timestamp("2021-01-01T01:00:00Z")],
        }
    )
    features = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "feature_name": ["pool_age_hours"],
            "feature_value": [1.0],
            "feature_observed_at": [pd.Timestamp("2021-01-01T00:00:00Z")],
        }
    )
    opcodes = pd.DataFrame(
        {
            "token_address": ["0xABC"],
            "PUSH1": [7],
            "SSTORE": [2],
            "label": [1],
        }
    )
    samples_path = tmp_path / "samples.parquet"
    features_path = tmp_path / "features.parquet"
    opcodes_path = tmp_path / "opcodes.csv"
    output_path = tmp_path / "features_static.parquet"
    samples.to_parquet(samples_path, index=False)
    features.to_parquet(features_path, index=False)
    opcodes.to_csv(opcodes_path, index=False)

    out = append_opcode_features(samples_path, features_path, opcodes_path, output_path)

    names = set(out["feature_name"])
    assert "opcode__PUSH1" in names
    assert "opcode__SSTORE" in names
    assert "opcode_features_available" in names
    assert output_path.exists()
