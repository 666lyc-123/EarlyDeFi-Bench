from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from earlydefi.evaluation.early_warning import event_warning_metrics, horizon_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sample scores into horizon and event-warning metrics.")
    parser.add_argument("--scores", required=True, help="Path to baseline_scores.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--event-top-k", default="5,10,20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizon = horizon_metrics(scores, top_k=args.top_k)
    event = event_warning_metrics(
        scores,
        top_k_values=[int(item.strip()) for item in args.event_top_k.split(",") if item.strip()],
    )
    horizon.to_csv(output_dir / "horizon_metrics.csv", index=False)
    event.to_csv(output_dir / "event_warning_metrics.csv", index=False)
    print("Horizon metrics")
    print(horizon.to_string(index=False))
    print("\nEvent warning metrics")
    print(event.to_string(index=False))


if __name__ == "__main__":
    main()
