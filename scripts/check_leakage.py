from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from earlydefi.config import load_config
from earlydefi.data.leakage import assert_no_feature_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check feature timestamp leakage.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--samples", default=None, help="Samples parquet path.")
    parser.add_argument("--features", default=None, help="Features parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples_path = Path(args.samples or config["paths"]["samples"])
    features_path = Path(args.features or config["paths"]["features"])

    samples = pd.read_parquet(samples_path)
    features = pd.read_parquet(features_path)
    assert_no_feature_leakage(samples, features)
    print("No feature leakage found.")


if __name__ == "__main__":
    main()
