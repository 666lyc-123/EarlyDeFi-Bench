from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create intentionally leaky random sample splits for diagnostics.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    samples = pd.read_parquet(args.samples)
    pieces = []
    for _, group in samples.groupby("label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n = len(shuffled)
        train_end = int(n * args.train_ratio)
        validation_end = train_end + int(n * args.validation_ratio)
        shuffled["split"] = "test"
        shuffled.loc[: train_end - 1, "split"] = "train"
        shuffled.loc[train_end : validation_end - 1, "split"] = "validation"
        pieces.append(shuffled)
    out = pd.concat(pieces, ignore_index=True).sort_values(["split", "label", "sample_id"])
    out.to_parquet(args.output, index=False)
    print(f"Wrote leaky random sample split to {args.output}")
    print(out.groupby(["split", "label"]).size().to_string())
    overlap = out.groupby("entity_id")["split"].nunique()
    print(f"entities_crossing_splits {int((overlap > 1).sum())}")


if __name__ == "__main__":
    main()
