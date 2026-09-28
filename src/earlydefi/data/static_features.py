from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_opcode_features(
    samples_path: str | Path,
    features_path: str | Path,
    opcodes_path: str | Path,
    output_path: str | Path,
    *,
    prefix: str = "opcode",
) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    features = pd.read_parquet(features_path)
    opcode_features = load_opcode_features(opcodes_path)

    observed_at = (
        features.groupby("sample_id", as_index=False)["feature_observed_at"]
        .min()
        .rename(columns={"feature_observed_at": "static_feature_observed_at"})
    )
    sample_tokens = samples[["sample_id", "token_address"]].merge(
        observed_at,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    sample_tokens["token_address"] = sample_tokens["token_address"].astype("string").str.lower()

    opcode_cols = [
        col
        for col in opcode_features.columns
        if col not in {"token_address", "label"}
    ]
    joined = sample_tokens.merge(
        opcode_features[["token_address", *opcode_cols]],
        on="token_address",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    joined["opcode_features_available"] = joined["_merge"].eq("both").astype(float)

    availability = joined[
        ["sample_id", "opcode_features_available", "static_feature_observed_at"]
    ].rename(
        columns={
            "opcode_features_available": "feature_value",
            "static_feature_observed_at": "feature_observed_at",
        }
    )
    availability["feature_name"] = f"{prefix}_features_available"
    availability = availability[["sample_id", "feature_name", "feature_value", "feature_observed_at"]]

    matched = joined[joined["_merge"].eq("both")].copy()
    opcode_rows = matched.melt(
        id_vars=["sample_id", "static_feature_observed_at"],
        value_vars=opcode_cols,
        var_name="feature_name",
        value_name="feature_value",
    )
    opcode_rows["feature_name"] = prefix + "__" + opcode_rows["feature_name"].astype(str)
    opcode_rows = opcode_rows.rename(
        columns={"static_feature_observed_at": "feature_observed_at"}
    )[["sample_id", "feature_name", "feature_value", "feature_observed_at"]]
    opcode_rows["feature_value"] = pd.to_numeric(opcode_rows["feature_value"], errors="coerce").fillna(0.0)

    out = pd.concat([features, availability, opcode_rows], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def load_opcode_features(opcodes_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(opcodes_path)
    if "token_address" not in raw.columns:
        raise ValueError("opcode feature file is missing token_address")
    out = raw.copy()
    out["token_address"] = out["token_address"].astype("string").str.lower()
    out = out.drop_duplicates("token_address", keep="last")
    return out
