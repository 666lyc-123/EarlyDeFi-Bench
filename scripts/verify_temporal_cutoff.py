"""Temporal cutoff audit: verify max_included_time <= prediction_time.

This script closes the review request on temporal leakage with the workshop's
own released feature and trace files. For every one of the 1,056 main warning
samples it checks, with zero tolerance:

1. Feature-cutoff check: the maximum ``feature_observed_at`` over all 68
   features of a sample (i.e. the latest timestamp any included feature row
   claims to have observed) is not later than the sample's declared
   ``prediction_time``. This is the released-table form of the
   ``max_included_time <= prediction_time`` contract.
2. Pre-event check: positive samples are warned strictly before the
   reconstructed event time, and every sample is warned no earlier than pool
   creation.
3. Trace check: the case-study trace rows carry
   ``evidence_observed_no_later_than_prediction`` flags and
   ``max_evidence_observed_at_utc`` values; all of them must satisfy the
   cutoff against ``prediction_time_utc``.

Boundary note (kept explicit on purpose): historical RPC and
participant-graph collection indexed windows by *estimated prediction
blocks*. ``feature_observed_at`` is therefore the declared cutoff recorded at
collection time, and this audit verifies that declared cutoff against
``prediction_time`` — the same boundary the paper states. It does not claim
timestamp-level visibility of individual logs inside the window.

Outputs are archived to ``outputs/temporal_cutoff_verification.json`` and
``outputs/temporal_cutoff_verification.md``; the script exits non-zero if any
check finds a violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default="data/processed/pilot_samples_200_200_full_matched_stratified_temporal.parquet",
    )
    parser.add_argument(
        "--features",
        default="data/processed/pilot_features_200_200_full_matched_rpc_events_participant_graph.parquet",
    )
    parser.add_argument(
        "--candidates",
        default="data/processed/benchmark_candidates_200_200_full.parquet",
    )
    parser.add_argument("--trace", default="outputs/case_study_trace.csv")
    parser.add_argument("--output-json", default="outputs/temporal_cutoff_verification.json")
    parser.add_argument("--output-md", default="outputs/temporal_cutoff_verification.md")
    return parser.parse_args()


def feature_cutoff_check(samples: pd.DataFrame, features: pd.DataFrame) -> dict:
    prediction_time = samples[["sample_id", "prediction_time"]].copy()
    prediction_time["prediction_time"] = pd.to_datetime(prediction_time["prediction_time"], utc=True)

    rows = features[["sample_id", "feature_name", "feature_observed_at"]].copy()
    rows["feature_observed_at"] = pd.to_datetime(rows["feature_observed_at"], utc=True)
    joined = rows.merge(prediction_time, on="sample_id", how="left", validate="many_to_one")
    if joined["prediction_time"].isna().any():
        raise AssertionError("Feature rows reference samples missing from the sample table.")

    row_violations = joined[joined["feature_observed_at"].gt(joined["prediction_time"])]
    per_sample = (
        joined.groupby("sample_id")
        .agg(max_included_time=("feature_observed_at", "max"), prediction_time=("prediction_time", "first"))
        .reset_index()
    )
    per_sample["margin_seconds"] = (
        per_sample["prediction_time"] - per_sample["max_included_time"]
    ).dt.total_seconds()
    sample_violations = per_sample[per_sample["max_included_time"].gt(per_sample["prediction_time"])]

    return {
        "n_samples": int(per_sample["sample_id"].nunique()),
        "n_feature_rows": int(len(joined)),
        "n_feature_names": int(features["feature_name"].nunique()),
        "row_violations": int(len(row_violations)),
        "sample_violations": int(len(sample_violations)),
        "min_margin_seconds": float(per_sample["margin_seconds"].min()),
        "median_margin_seconds": float(per_sample["margin_seconds"].median()),
        "samples_with_zero_margin": int((per_sample["margin_seconds"] == 0).sum()),
        "pass": bool(len(row_violations) == 0 and len(sample_violations) == 0),
    }


def pre_event_check(samples: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    frame = samples.copy()
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], utc=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)

    positives = frame[frame["label"].eq(1)]
    not_before_event = int((positives["prediction_time"].ge(positives["event_time"])).sum())

    key = ["chain", "token_address", "pool_address"]
    merged = frame.merge(
        candidates[key + ["pool_created_at"]], on=key, how="left", validate="many_to_one"
    )
    merged["pool_created_at"] = pd.to_datetime(merged["pool_created_at"], utc=True)
    before_pool_creation = int(merged["prediction_time"].lt(merged["pool_created_at"]).sum())

    return {
        "n_positive_samples": int(len(positives)),
        "positive_warning_not_strictly_before_event": not_before_event,
        "n_samples_before_pool_creation": before_pool_creation,
        "pass": bool(not_before_event == 0 and before_pool_creation == 0),
    }


def trace_check(trace_path: Path) -> dict:
    if not trace_path.is_file():
        return {"skipped": True, "reason": f"{trace_path} not present", "pass": True}
    trace = pd.read_csv(trace_path)
    required = {"prediction_time_utc", "max_evidence_observed_at_utc", "evidence_observed_no_later_than_prediction"}
    missing = required - set(trace.columns)
    if missing:
        raise AssertionError(f"Case-study trace is missing columns: {sorted(missing)}")

    prediction = pd.to_datetime(trace["prediction_time_utc"], utc=True, errors="coerce")
    observed = pd.to_datetime(trace["max_evidence_observed_at_utc"], utc=True, errors="coerce")
    flag = trace["evidence_observed_no_later_than_prediction"].astype(bool)

    violations = trace[observed.gt(prediction) | ~flag]
    return {
        "skipped": False,
        "n_trace_rows": int(len(trace)),
        "trace_violations": int(len(violations)),
        "flag_all_true": bool(flag.all()),
        "pass": bool(len(violations) == 0 and flag.all()),
    }


def write_md(path: Path, report: dict) -> None:
    lines = [
        "# Temporal Cutoff Verification",
        "",
        "Audits the released feature table and case-study trace against the",
        "contract `max_included_time <= prediction_time` (zero tolerance).",
        "",
        "Boundary note: historical RPC and participant-graph windows were indexed",
        "by estimated prediction blocks. `feature_observed_at` is the declared",
        "cutoff recorded at collection time; this audit verifies that declared",
        "cutoff against `prediction_time`, matching the paper's stated boundary.",
        "It does not claim timestamp-level visibility of individual logs.",
        "",
        "## Feature cutoff (main table)",
        "",
        f"- Samples: {report['feature_cutoff']['n_samples']}",
        f"- Feature rows: {report['feature_cutoff']['n_feature_rows']}",
        f"- Feature names: {report['feature_cutoff']['n_feature_names']}",
        f"- Row violations (observed_at > prediction_time): {report['feature_cutoff']['row_violations']}",
        f"- Sample violations (max included time > prediction time): {report['feature_cutoff']['sample_violations']}",
        f"- Min margin (seconds): {report['feature_cutoff']['min_margin_seconds']}",
        f"- Samples with zero margin (cutoff exactly at prediction time): {report['feature_cutoff']['samples_with_zero_margin']}",
        "",
        "## Pre-event validity",
        "",
        f"- Positive samples warned not strictly before event: {report['pre_event']['positive_warning_not_strictly_before_event']}",
        f"- Samples warned before pool creation: {report['pre_event']['n_samples_before_pool_creation']}",
        "",
        "## Case-study trace",
        "",
        f"- Skipped: {report['trace'].get('skipped', False)}",
    ]
    if not report["trace"].get("skipped", False):
        lines += [
            f"- Trace rows: {report['trace']['n_trace_rows']}",
            f"- Trace violations: {report['trace']['trace_violations']}",
            f"- Flags all true: {report['trace']['flag_all_true']}",
        ]
    lines += [
        "",
        f"Overall pass: **{report['overall_pass']}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(ROOT / args.samples)
    features = pd.read_parquet(ROOT / args.features)
    candidates = pd.read_parquet(ROOT / args.candidates)

    report = {
        "feature_cutoff": feature_cutoff_check(samples, features),
        "pre_event": pre_event_check(samples, candidates),
        "trace": trace_check(ROOT / args.trace),
    }
    report["overall_pass"] = bool(
        report["feature_cutoff"]["pass"] and report["pre_event"]["pass"] and report["trace"]["pass"]
    )

    output_json = ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(ROOT / args.output_md, report)
    print(json.dumps(report, indent=2))

    if not report["overall_pass"]:
        raise AssertionError("Temporal cutoff verification found violations.")
    print("PASS: max_included_time <= prediction_time holds for every released sample.")


if __name__ == "__main__":
    main()
