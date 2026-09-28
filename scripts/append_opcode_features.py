from __future__ import annotations

import argparse

from earlydefi.data.leakage import assert_no_feature_leakage
from earlydefi.data.static_features import append_opcode_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append static opcode-count features to pilot feature rows.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--opcodes",
        default="data/raw/sources/trapdoor_data_repo/opcodes_based_dataset.csv",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = append_opcode_features(args.samples, args.features, args.opcodes, args.output)
    assert_no_feature_leakage(
        __import__("pandas").read_parquet(args.samples),
        out,
    )
    print(f"Wrote {len(out)} feature rows to {args.output}")
    print(out["feature_name"].str.startswith("opcode__").value_counts().to_string())


if __name__ == "__main__":
    main()
