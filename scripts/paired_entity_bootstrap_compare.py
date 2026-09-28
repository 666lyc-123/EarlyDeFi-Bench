from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    top = order[: min(k, len(order))]
    positives = float(y_true.sum())
    precision = float(y_true[top].sum()) / max(1, len(top))
    recall = np.nan if positives == 0 else float(y_true[top].sum()) / positives
    return precision, recall


def metrics(df: pd.DataFrame, k: int) -> dict[str, float]:
    y = df["label"].to_numpy(dtype=int)
    s = df["risk_score"].to_numpy(dtype=float)
    p_at_k, r_at_k = precision_recall_at_k(y, s, k)
    return {
        "auroc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        "auprc": average_precision_score(y, s) if y.sum() > 0 else np.nan,
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
    }


def event_view(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("entity_id", as_index=False)
        .agg(label=("label", "max"), risk_score=("risk_score", "max"))
    )


def read_scores(path: Path, model: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].eq(model)].copy()
    if df.empty:
        raise ValueError(f"No rows for model={model} in {path}")
    df["method"] = method
    return df


def finite_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired entity-level bootstrap comparison of two score files.")
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-scores", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    baseline = read_scores(Path(args.baseline_scores), args.baseline_model, args.baseline_name)
    candidate = read_scores(Path(args.candidate_scores), args.candidate_model, args.candidate_name)
    shared_entities = sorted(set(baseline["entity_id"]) & set(candidate["entity_id"]))
    if not shared_entities:
        raise ValueError("No shared entities between baseline and candidate score files.")
    baseline = baseline[baseline["entity_id"].isin(shared_entities)].copy()
    candidate = candidate[candidate["entity_id"].isin(shared_entities)].copy()

    baseline_by_entity = [frame.copy() for _, frame in baseline.groupby("entity_id", sort=False)]
    candidate_by_entity = [frame.copy() for _, frame in candidate.groupby("entity_id", sort=False)]
    baseline_event = event_view(baseline)
    candidate_event = event_view(candidate)
    baseline_event_by_entity = [frame.copy() for _, frame in baseline_event.groupby("entity_id", sort=False)]
    candidate_event_by_entity = [frame.copy() for _, frame in candidate_event.groupby("entity_id", sort=False)]
    if len(baseline_by_entity) != len(candidate_by_entity):
        raise ValueError("Mismatched entity grouping after intersection.")

    def combined_metrics(frames: list[pd.DataFrame]) -> dict[str, float]:
        return metrics(pd.concat(frames, ignore_index=True), args.top_k)

    observed_baseline = combined_metrics(baseline_by_entity)
    observed_candidate = combined_metrics(candidate_by_entity)
    observed_baseline_event = combined_metrics(baseline_event_by_entity)
    observed_candidate_event = combined_metrics(candidate_event_by_entity)

    rng = np.random.default_rng(args.seed)
    n_entities = len(baseline_by_entity)
    rows = []
    for replicate in range(args.n_bootstrap):
        sample_idx = rng.integers(0, n_entities, size=n_entities)
        b = combined_metrics([baseline_by_entity[i] for i in sample_idx])
        c = combined_metrics([candidate_by_entity[i] for i in sample_idx])
        be = combined_metrics([baseline_event_by_entity[i] for i in sample_idx])
        ce = combined_metrics([candidate_event_by_entity[i] for i in sample_idx])
        row = {"replicate": replicate}
        for metric_name in ["auroc", "auprc", "p_at_k", "r_at_k"]:
            row[f"delta_{metric_name}"] = c[metric_name] - b[metric_name]
            row[f"delta_event_{metric_name}"] = ce[metric_name] - be[metric_name]
        rows.append(row)

    boot = pd.DataFrame(rows)
    summary_rows = []
    for col in [c for c in boot.columns if c != "replicate"]:
        mean, lo, hi = finite_ci(boot[col].tolist())
        metric_name = col.removeprefix("delta_")
        if metric_name.startswith("event_"):
            base_observed = observed_baseline_event[metric_name.removeprefix("event_")]
            cand_observed = observed_candidate_event[metric_name.removeprefix("event_")]
        else:
            base_observed = observed_baseline[metric_name]
            cand_observed = observed_candidate[metric_name]
        summary_rows.append(
            {
                "candidate": args.candidate_name,
                "baseline": args.baseline_name,
                "metric": col,
                "baseline_observed": base_observed,
                "candidate_observed": cand_observed,
                "delta_observed": cand_observed - base_observed,
                "delta_mean": mean,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "win_rate": float((boot[col] > 0).mean()),
                "n_shared_entities": n_entities,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{args.candidate_name}_vs_{args.baseline_name}".replace(" ", "_").replace("/", "_")
    pd.DataFrame(summary_rows).to_csv(out_dir / f"{safe_name}_summary.csv", index=False)
    boot.to_csv(out_dir / f"{safe_name}_replicates.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
