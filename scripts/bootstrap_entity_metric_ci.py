from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    if len(y_true) == 0:
        return np.nan, np.nan
    order = np.argsort(-scores, kind="mergesort")
    top = order[: min(k, len(order))]
    tp = float(y_true[top].sum())
    precision = tp / max(1, len(top))
    positives = float(y_true.sum())
    recall = np.nan if positives == 0 else tp / positives
    return precision, recall


def sample_metrics(df: pd.DataFrame, k: int) -> dict[str, float]:
    y = df["label"].to_numpy(dtype=int)
    s = df["risk_score"].to_numpy(dtype=float)
    out = {}
    out["auroc"] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    out["auprc"] = average_precision_score(y, s) if y.sum() > 0 else np.nan
    out["precision_at_k"], out["recall_at_k"] = precision_recall_at_k(y, s, k)
    return out


def event_metrics(df: pd.DataFrame, k: int) -> dict[str, float]:
    ent = (
        df.groupby("entity_id", as_index=False)
        .agg(label=("label", "max"), risk_score=("risk_score", "max"))
    )
    return {f"event_{key}": value for key, value in sample_metrics(ent, k).items()}


def ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    scores = scores[scores["model"].eq(args.model)].copy()
    if scores.empty:
        raise ValueError(f"No rows for model {args.model}")

    entity_frames = [frame.copy() for _, frame in scores.groupby("entity_id", sort=False)]
    entity_event = (
        scores.groupby("entity_id", as_index=False)
        .agg(label=("label", "max"), risk_score=("risk_score", "max"))
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(args.seed)

    observed = {"replicate": "observed"}
    observed.update(sample_metrics(scores, args.top_k))
    observed.update(event_metrics(scores, args.top_k))

    rows = []
    n_entities = len(entity_frames)
    for idx in range(args.n_bootstrap):
        sampled_idx = rng.integers(0, n_entities, size=n_entities)
        boot = pd.concat([entity_frames[i] for i in sampled_idx], ignore_index=True)
        boot_event = entity_event.iloc[sampled_idx].copy()
        row = {"replicate": idx}
        row.update(sample_metrics(boot, args.top_k))
        row.update({f"event_{key}": value for key, value in sample_metrics(boot_event, args.top_k).items()})
        rows.append(row)

    boot = pd.DataFrame(rows)
    metrics = [c for c in boot.columns if c != "replicate"]
    summary_rows = []
    for metric in metrics:
        mean, lo, hi = ci(boot[metric].tolist())
        summary_rows.append(
            {
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_bootstrap_valid": int(np.isfinite(boot[metric]).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{args.model}_entity_bootstrap_ci_summary.csv", index=False)
    boot.to_csv(out_dir / f"{args.model}_entity_bootstrap_ci_replicates.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
