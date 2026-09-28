from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from earlydefi.data.splits import SplitRatios, stratified_temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create label-stratified temporal splits for small pilot datasets.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(args.samples)
    split_samples = stratified_temporal_split(
        samples,
        SplitRatios(args.train_ratio, args.validation_ratio, args.test_ratio),
        group_by_entity=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_samples.to_parquet(output, index=False)
    print(f"Wrote stratified temporal split samples to {output}")
    print(split_samples.groupby(["split", "label"]).size().to_string())
    print(split_samples.groupby("split")["entity_id"].nunique().to_string())


if __name__ == "__main__":
    main()
