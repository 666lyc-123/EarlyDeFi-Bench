from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_baselines_for_dataset import run_baselines


FORBIDDEN_FUSION_FEATURES = {
    "pool_creation_block",
    "token_creation_block",
    "rpc_pool_events_cache_complete",
    "rpc_pool_events_rpc_error",
    "opcode_features_available",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce opcode diagnostics.")
    parser.add_argument(
        "--samples",
        default="data/processed/pilot_samples_200_200_full_opcode_complete.parquet",
    )
    parser.add_argument(
        "--features",
        default="data/processed/pilot_features_200_200_full_opcode_complete.parquet",
    )
    parser.add_argument("--output-dir", default="reproduced/opcode_diagnostic")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_results(output_dir: Path, results: pd.DataFrame, scores: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "baseline_results.csv", index=False)
    scores.to_csv(output_dir / "baseline_scores.csv", index=False)
    (output_dir / "baseline_results.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(args.samples)
    features = pd.read_parquet(args.features)
    output = Path(args.output_dir)

    opcode_only = features[features["feature_name"].str.startswith("opcode__")].copy()
    opcode_results, opcode_scores = run_baselines(
        samples, opcode_only, top_k=10, random_state=args.random_state
    )
    write_results(output / "opcode_only", opcode_results, opcode_scores)

    fusion = features[~features["feature_name"].isin(FORBIDDEN_FUSION_FEATURES)].copy()
    fusion_results, fusion_scores = run_baselines(
        samples, fusion, top_k=10, random_state=args.random_state
    )
    write_results(output / "opcode_fusion", fusion_results, fusion_scores)

    print(opcode_results[["model", "auroc", "auprc", "f1"]].to_string(index=False))
    print(fusion_results[["model", "auroc", "auprc", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
