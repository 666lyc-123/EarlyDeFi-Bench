"""Rebuild the main long-format feature table offline from published caches.

Historical RPC collection used transient ``data/interim/*.jsonl`` caches that
are intentionally not part of the release. Instead, the per-family collection
outputs are published as parquet caches under ``data/caches/``:

- ``metadata_features_200_200_full.parquet`` — six metadata features per sample;
- ``rpc_event_count_features_200_200_full_24h_cached.parquet`` — 24h RPC
  pool-event count features per sample;
- ``pool_participant_graph_features_200_200_full.parquet`` — participant-graph
  features per sample (1h/6h/24h windows).

This script proves the released package is self-contained: the released main
feature table ``data/processed/pilot_features_200_200_full_matched_rpc_events_participant_graph.parquet``
is exactly the union of the three published caches, with no RPC access
required. It additionally re-derives the metadata feature values from the
released sample and benchmark-candidate tables as an independent cross-check.

The script exits non-zero if the rebuilt table does not match the released
table (same sample/feature coverage, same feature values, same observed-at
cutoffs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_FEATURES = (
    "has_pool_creation_time",
    "has_token_creation_time",
    "pool_age_hours",
    "pool_creation_block",
    "token_creation_block",
    "token_pool_delay_hours",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default="data/processed/pilot_samples_200_200_full_matched_stratified_temporal.parquet",
    )
    parser.add_argument(
        "--candidates",
        default="data/processed/benchmark_candidates_200_200_full.parquet",
    )
    parser.add_argument(
        "--metadata-cache",
        default="data/caches/metadata_features_200_200_full.parquet",
    )
    parser.add_argument(
        "--rpc-cache",
        default="data/caches/rpc_event_count_features_200_200_full_24h_cached.parquet",
    )
    parser.add_argument(
        "--graph-cache",
        default="data/caches/pool_participant_graph_features_200_200_full.parquet",
    )
    parser.add_argument(
        "--released",
        default="data/processed/pilot_features_200_200_full_matched_rpc_events_participant_graph.parquet",
    )
    parser.add_argument("--output-json", default="outputs/rebuild_verification.json")
    return parser.parse_args()


def load_published_caches(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for label, path in (
        ("metadata", args.metadata_cache),
        ("rpc", args.rpc_cache),
        ("graph", args.graph_cache),
    ):
        frame = pd.read_parquet(ROOT / path)
        missing = {"sample_id", "feature_name", "feature_value", "feature_observed_at"} - set(frame.columns)
        if missing:
            raise AssertionError(f"{label} cache {path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cross_check_metadata_values(samples: pd.DataFrame, candidates: pd.DataFrame, cache: pd.DataFrame) -> dict:
    """Independently recompute metadata feature values from released tables."""
    key = ["chain", "token_address", "pool_address"]
    merged = samples.merge(
        candidates[
            key + ["token_created_at", "pool_created_at", "token_creation_block", "pool_creation_block"]
        ],
        on=key,
        how="left",
        validate="many_to_one",
    )
    prediction_time = pd.to_datetime(merged["prediction_time"], utc=True)
    pool_created_at = pd.to_datetime(merged["pool_created_at"], utc=True)
    token_created_at = pd.to_datetime(merged["token_created_at"], utc=True)
    recomputed = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "has_pool_creation_time": merged["pool_created_at"].notna().astype(float),
            "has_token_creation_time": merged["token_created_at"].notna().astype(float),
            "pool_age_hours": (prediction_time - pool_created_at).dt.total_seconds() / 3600.0,
            "token_pool_delay_hours": (pool_created_at - token_created_at).dt.total_seconds() / 3600.0,
            "pool_creation_block": pd.to_numeric(merged["pool_creation_block"], errors="coerce"),
            "token_creation_block": pd.to_numeric(merged["token_creation_block"], errors="coerce"),
        }
    )
    wide = (
        cache[cache["feature_name"].isin(METADATA_FEATURES)]
        .pivot_table(index="sample_id", columns="feature_name", values="feature_value", aggfunc="first")
        .astype(float)
    )
    joined = wide.join(recomputed.set_index("sample_id").astype(float), rsuffix="__recomputed")
    max_abs_diff = 0.0
    for name in METADATA_FEATURES:
        diff = (joined[name] - joined[f"{name}__recomputed"]).abs().max()
        max_abs_diff = max(max_abs_diff, float(diff))
    return {"max_abs_diff": max_abs_diff, "tolerance": 1e-6, "pass": bool(max_abs_diff <= 1e-6)}


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(ROOT / args.samples)
    candidates = pd.read_parquet(ROOT / args.candidates)
    released = pd.read_parquet(ROOT / args.released)
    rebuilt = load_published_caches(args)

    key = ["sample_id", "feature_name"]
    rebuilt_indexed = rebuilt.set_index(key)
    released_indexed = released.set_index(key)

    missing = released_indexed.index.difference(rebuilt_indexed.index)
    extra = rebuilt_indexed.index.difference(released_indexed.index)
    common = released_indexed.index.intersection(rebuilt_indexed.index)

    released_values = pd.to_numeric(released_indexed.loc[common, "feature_value"], errors="coerce").astype(float)
    rebuilt_values = pd.to_numeric(rebuilt_indexed.loc[common, "feature_value"], errors="coerce").astype(float)
    value_mismatch = int(((released_values - rebuilt_values).abs() > 1e-6).sum())

    observed_released = pd.to_datetime(released_indexed.loc[common, "feature_observed_at"], utc=True)
    observed_rebuilt = pd.to_datetime(rebuilt_indexed.loc[common, "feature_observed_at"], utc=True)
    observed_mismatch = int((observed_released != observed_rebuilt).sum())

    metadata_check = cross_check_metadata_values(samples, candidates, rebuilt)

    prediction_time = samples[["sample_id", "prediction_time"]].copy()
    prediction_time["prediction_time"] = pd.to_datetime(prediction_time["prediction_time"], utc=True)
    per_sample_max = (
        pd.to_datetime(rebuilt["feature_observed_at"], utc=True)
        .groupby(rebuilt["sample_id"])
        .max()
        .rename("max_feature_observed_at")
        .reset_index()
    )
    joined = per_sample_max.merge(prediction_time, on="sample_id", how="left", validate="one_to_one")
    cutoff_violations = int(joined["max_feature_observed_at"].gt(joined["prediction_time"]).sum())

    report = {
        "released_rows": int(len(released)),
        "rebuilt_rows": int(len(rebuilt)),
        "samples": int(samples["sample_id"].nunique()),
        "feature_names_rebuilt": int(rebuilt["feature_name"].nunique()),
        "rows_missing_from_rebuild": int(len(missing)),
        "rows_extra_in_rebuild": int(len(extra)),
        "value_mismatches_above_1e-6": value_mismatch,
        "observed_at_mismatches": observed_mismatch,
        "metadata_value_cross_check": metadata_check,
        "cutoff_violations_max_feature_observed_after_prediction": cutoff_violations,
        "exact_match": bool(
            len(missing) == 0
            and len(extra) == 0
            and value_mismatch == 0
            and observed_mismatch == 0
            and metadata_check["pass"]
            and cutoff_violations == 0
        ),
        "inputs": {
            "samples": args.samples,
            "candidates": args.candidates,
            "metadata_cache": args.metadata_cache,
            "rpc_cache": args.rpc_cache,
            "graph_cache": args.graph_cache,
            "released": args.released,
        },
    }

    output = ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not report["exact_match"]:
        raise AssertionError("Rebuilt feature table does not match the released table.")
    print("PASS: released main feature table is exactly reproducible from published caches.")


if __name__ == "__main__":
    main()
