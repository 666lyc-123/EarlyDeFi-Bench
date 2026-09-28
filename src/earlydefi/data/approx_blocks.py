from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BLOCK_SECONDS = {
    "ethereum": 12.0,
    "bsc": 3.0,
}


def estimate_prediction_blocks(
    samples_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    pairs = pd.read_parquet(pairs_path)
    out = estimate_prediction_blocks_frame(samples, pairs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def estimate_prediction_blocks_frame(samples: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    required_samples = {"sample_id", "chain", "pool_address", "prediction_time"}
    required_pairs = {"chain", "pool_address", "pool_creation_block", "pool_created_at"}
    missing_samples = required_samples - set(samples.columns)
    missing_pairs = required_pairs - set(pairs.columns)
    if missing_samples:
        raise ValueError(f"samples is missing columns: {sorted(missing_samples)}")
    if missing_pairs:
        raise ValueError(f"pairs is missing columns: {sorted(missing_pairs)}")

    samples = samples.copy()
    pairs = pairs.copy()
    samples["pool_address"] = samples["pool_address"].astype("string").str.lower()
    pairs["pool_address"] = pairs["pool_address"].astype("string").str.lower()
    pairs = pairs.drop_duplicates(["chain", "pool_address"], keep="last")

    merged = samples.merge(
        pairs[["chain", "pool_address", "pool_creation_block", "pool_created_at"]],
        on=["chain", "pool_address"],
        how="left",
        validate="many_to_one",
    )
    merged["prediction_time"] = pd.to_datetime(merged["prediction_time"], utc=True, errors="coerce")
    merged["pool_created_at"] = pd.to_datetime(merged["pool_created_at"], utc=True, errors="coerce")
    merged["block_seconds"] = merged["chain"].map(DEFAULT_BLOCK_SECONDS).fillna(12.0).astype(float)
    elapsed_seconds = (merged["prediction_time"] - merged["pool_created_at"]).dt.total_seconds().clip(lower=0)
    merged["prediction_block_estimate"] = (
        pd.to_numeric(merged["pool_creation_block"], errors="coerce")
        + np.floor(elapsed_seconds / merged["block_seconds"])
    )
    merged["prediction_block_estimate"] = merged["prediction_block_estimate"].round().astype("Int64")
    merged["status"] = "found"
    missing = merged["pool_creation_block"].isna() | merged["pool_created_at"].isna()
    merged.loc[missing, "prediction_block_estimate"] = pd.NA
    merged.loc[missing, "status"] = "missing_pool_creation"
    return merged[
        [
            "sample_id",
            "chain",
            "pool_address",
            "prediction_time",
            "prediction_block_estimate",
            "block_seconds",
            "status",
        ]
    ].copy()
