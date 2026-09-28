from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from earlydefi.data.leakage import assert_no_feature_leakage
from earlydefi.evaluation.metrics import best_f1_threshold
from earlydefi.models.sklearn_baselines import make_linear_svm

from scripts.run_baselines_for_dataset import make_feature_matrix, predict_scores


DEFAULT_SAMPLES = Path("data/processed/pilot_samples_200_200_full_matched_stratified_temporal.parquet")
DEFAULT_FEATURES = Path("data/processed/pilot_features_200_200_full_matched_rpc_events_participant_graph.parquet")
DEFAULT_BASELINE_RESULTS = Path(
    "outputs/pilot_200_200_full_matched_rpc_events_participant_graph/"
    "feature_family_ablation/base_plus_graph_no_counts/baseline_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/calibration_diagnostic")
DEFAULT_REPORT = Path("docs/calibration_diagnostic.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only Platt calibration diagnostic for the main "
            "DeFiGuard-NC Linear SVM benchmark row."
        )
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def expected_calibration_error(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    n_bins: int = 5,
) -> tuple[float, float, pd.DataFrame]:
    """Compute equal-width ECE/MCE and per-bin calibration rows."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    y_true = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    if y_true.shape != y_probability.shape:
        raise ValueError("y_true and y_probability must have the same shape")
    if len(y_true) == 0:
        raise ValueError("at least one observation is required")
    if np.any((y_probability < 0.0) | (y_probability > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    mce = 0.0
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        if idx == n_bins - 1:
            mask = (y_probability >= left) & (y_probability <= right)
        else:
            mask = (y_probability >= left) & (y_probability < right)

        bin_count = int(mask.sum())
        if bin_count:
            mean_probability = float(y_probability[mask].mean())
            empirical_rate = float(y_true[mask].mean())
            abs_gap = abs(mean_probability - empirical_rate)
        else:
            mean_probability = float("nan")
            empirical_rate = float("nan")
            abs_gap = float("nan")

        weighted_abs_gap = 0.0 if bin_count == 0 else float((bin_count / len(y_true)) * abs_gap)
        ece += weighted_abs_gap
        if bin_count:
            mce = max(mce, abs_gap)
        rows.append(
            {
                "bin": idx + 1,
                "lower": float(left),
                "upper": float(right),
                "count": bin_count,
                "mean_probability": mean_probability,
                "empirical_positive_rate": empirical_rate,
                "abs_gap": abs_gap,
                "weighted_abs_gap": weighted_abs_gap,
            }
        )

    return float(ece), float(mce), pd.DataFrame(rows)


def load_main_feature_names(baseline_results: Path, model_name: str) -> list[str]:
    results = pd.read_csv(baseline_results)
    rows = results[results["model"].eq(model_name)]
    if rows.empty:
        raise ValueError(f"Model {model_name!r} was not found in {baseline_results}")
    features = rows.iloc[0]["features"]
    feature_names = [name.strip() for name in str(features).split(",") if name.strip()]
    if not feature_names:
        raise ValueError(f"No feature list found for {model_name!r} in {baseline_results}")
    return feature_names


def run_diagnostic(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_names: list[str],
    baseline_results: pd.DataFrame,
    model_name: str,
    n_bins: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_no_feature_leakage(samples, features)
    selected_features = features[features["feature_name"].isin(feature_names)].copy()
    available = set(selected_features["feature_name"].dropna())
    missing = sorted(set(feature_names) - available)
    if missing:
        raise ValueError(f"Feature rows are missing requested features: {missing}")

    matrix = make_feature_matrix(samples, selected_features)
    feature_cols = sorted(feature_names)
    train = matrix[matrix["split"].eq("train")].copy()
    validation = matrix[matrix["split"].eq("validation")].copy()
    test = matrix[matrix["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty train, validation, and test splits.")

    model = make_linear_svm(random_state=random_state)
    model.fit(train[feature_cols], train["label"])
    validation_scores = predict_scores(model, validation[feature_cols])
    threshold, validation_f1 = best_f1_threshold(validation["label"].to_numpy(), validation_scores)
    test_scores = predict_scores(model, test[feature_cols])

    calibrator = LogisticRegression(max_iter=1000, random_state=random_state)
    calibrator.fit(validation_scores.reshape(-1, 1), validation["label"].to_numpy())
    validation_probability = calibrator.predict_proba(validation_scores.reshape(-1, 1))[:, 1]
    test_probability = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    test_labels = test["label"].to_numpy()
    ece, mce, bins = expected_calibration_error(test_labels, test_probability, n_bins=n_bins)
    saved = baseline_results[baseline_results["model"].eq(model_name)].iloc[0]
    reproduced_auroc = float(roc_auc_score(test_labels, test_scores))
    reproduced_auprc = float(average_precision_score(test_labels, test_scores))
    summary = pd.DataFrame(
        [
            {
                "diagnostic": "validation_platt_test_calibration",
                "model": model_name,
                "n_features": len(feature_cols),
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "validation_positive_rate": float(validation["label"].mean()),
                "test_positive_rate": float(test["label"].mean()),
                "reproduced_validation_threshold": float(threshold),
                "saved_validation_threshold": float(saved["threshold"]),
                "reproduced_validation_best_f1": float(validation_f1),
                "saved_validation_best_f1": float(saved["validation_best_f1"]),
                "reproduced_test_auroc": reproduced_auroc,
                "saved_test_auroc": float(saved["auroc"]),
                "reproduced_test_auprc": reproduced_auprc,
                "saved_test_auprc": float(saved["auprc"]),
                "test_brier": float(brier_score_loss(test_labels, test_probability)),
                "test_log_loss": float(log_loss(test_labels, test_probability, labels=[0, 1])),
                f"test_ece_{n_bins}_bins": ece,
                f"test_mce_{n_bins}_bins": mce,
                "test_mean_probability": float(test_probability.mean()),
                "test_empirical_positive_rate": float(test_labels.mean()),
                "platt_coef": float(calibrator.coef_[0, 0]),
                "platt_intercept": float(calibrator.intercept_[0]),
                "validation_raw_score_min": float(np.min(validation_scores)),
                "validation_raw_score_max": float(np.max(validation_scores)),
                "test_raw_score_min": float(np.min(test_scores)),
                "test_raw_score_max": float(np.max(test_scores)),
            }
        ]
    )

    scores = _score_frame(validation, validation_scores, validation_probability, "validation")
    scores = pd.concat(
        [scores, _score_frame(test, test_scores, test_probability, "test")],
        ignore_index=True,
    )
    return summary, bins, scores


def _score_frame(
    split: pd.DataFrame,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "entity_id",
        "label",
        "horizon_hours",
        "prediction_time",
        "event_time",
    ]
    out = split[columns].copy()
    out.insert(2, "split", split_name)
    out["raw_decision_score"] = raw_scores
    out["platt_probability"] = probabilities
    return out


def write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    output_dir: Path,
    n_bins: int,
) -> None:
    row = summary.iloc[0]
    summary_table = _metric_table(
        {
            "n_train": row["n_train"],
            "n_validation": row["n_validation"],
            "n_test": row["n_test"],
            "validation_positive_rate": row["validation_positive_rate"],
            "test_positive_rate": row["test_positive_rate"],
            "reproduced_test_auroc": row["reproduced_test_auroc"],
            "reproduced_test_auprc": row["reproduced_test_auprc"],
            "test_brier": row["test_brier"],
            f"test_ece_{n_bins}_bins": row[f"test_ece_{n_bins}_bins"],
            f"test_mce_{n_bins}_bins": row[f"test_mce_{n_bins}_bins"],
            "test_mean_probability": row["test_mean_probability"],
            "test_empirical_positive_rate": row["test_empirical_positive_rate"],
        }
    )
    reproduction_table = _metric_table(
        {
            "saved_validation_threshold": row["saved_validation_threshold"],
            "reproduced_validation_threshold": row["reproduced_validation_threshold"],
            "saved_test_auroc": row["saved_test_auroc"],
            "reproduced_test_auroc": row["reproduced_test_auroc"],
            "saved_test_auprc": row["saved_test_auprc"],
            "reproduced_test_auprc": row["reproduced_test_auprc"],
        }
    )
    lines = [
        "# DeFiGuard-NC Calibration Diagnostic",
        "",
        "This diagnostic audits probability calibration for the main DeFiGuard-NC Linear SVM row. It is not part of the main benchmark claim: the manuscript still treats the published model outputs as ranking scores and validation-threshold alert scores, not calibrated loss probabilities.",
        "",
        "## Protocol",
        "",
        "- Rebuild the main Linear SVM from the saved `baseline_results.csv` feature list.",
        "- Fit the classifier on the train split only.",
        "- Select the alert threshold on the validation split, matching the benchmark protocol.",
        "- Fit a Platt-style logistic calibrator on validation decision scores only.",
        "- Report Brier score, log loss, and equal-width ECE/MCE on the held-out test split.",
        "",
        "## Baseline Reproduction Check",
        "",
        "The rerun reproduces the held-out ranking metrics exactly; the validation threshold is shown as an audit value because small solver-tolerance differences can move an equivalent margin threshold in the fourth decimal place.",
        "",
        reproduction_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Test Calibration Summary",
        "",
        summary_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"## Equal-Width Calibration Bins ({n_bins})",
        "",
        bins.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "The diagnostic provides a deployment-facing audit trail for probability calibration, but it does not convert the benchmark into a calibrated probability-estimation claim. Any deployed monitor should recalibrate and re-audit probabilities on its own validation stream, especially after market-regime or chain-coverage shifts.",
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{output_dir / 'defiguard_nc_platt_calibration_summary.csv'}`",
        f"- Calibration bins CSV: `{output_dir / 'defiguard_nc_calibration_bins.csv'}`",
        f"- Validation/test score CSV: `{output_dir / 'defiguard_nc_validation_test_scores.csv'}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_table(values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in values.items()])


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(args.samples)
    features = pd.read_parquet(args.features)
    baseline_results = pd.read_csv(args.baseline_results)
    feature_names = load_main_feature_names(args.baseline_results, args.model)

    summary, bins, scores = run_diagnostic(
        samples,
        features,
        feature_names=feature_names,
        baseline_results=baseline_results,
        model_name=args.model,
        n_bins=args.n_bins,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "defiguard_nc_platt_calibration_summary.csv", index=False)
    bins.to_csv(args.output_dir / "defiguard_nc_calibration_bins.csv", index=False)
    scores.to_csv(args.output_dir / "defiguard_nc_validation_test_scores.csv", index=False)
    write_report(args.report, summary=summary, bins=bins, output_dir=args.output_dir, n_bins=args.n_bins)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
