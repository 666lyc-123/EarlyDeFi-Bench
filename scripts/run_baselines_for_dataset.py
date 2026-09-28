from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from earlydefi.data.leakage import assert_no_feature_leakage
from earlydefi.evaluation.metrics import best_f1_threshold, binary_metrics, precision_recall_at_k
from earlydefi.models.sklearn_baselines import (
    make_extra_trees,
    make_hist_gradient_boosting,
    make_isolation_forest,
    make_linear_svm,
    make_logistic_regression,
    make_one_class_svm,
    make_random_forest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sklearn baselines for a leakage-checked dataset.")
    parser.add_argument("--samples", required=True, help="Path to pilot samples parquet.")
    parser.add_argument("--features", required=True, help="Path to pilot feature rows parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON results.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--drop-features",
        default="",
        help="Comma-separated feature names to exclude before training.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(args.samples)
    features = pd.read_parquet(args.features)
    assert_no_feature_leakage(samples, features)

    drop_features = {name.strip() for name in args.drop_features.split(",") if name.strip()}
    if drop_features:
        features = features[~features["feature_name"].isin(drop_features)].copy()

    results, scores = run_baselines(samples, features, top_k=args.top_k, random_state=args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "baseline_results.csv", index=False)
    scores.to_csv(output_dir / "baseline_scores.csv", index=False)
    (output_dir / "baseline_results.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


def run_baselines(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    *,
    top_k: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = make_feature_matrix(samples, features)
    feature_cols = sorted(features["feature_name"].dropna().unique().tolist())
    if not feature_cols:
        raise ValueError("No feature columns available after filtering.")

    train = matrix[matrix["split"] == "train"]
    validation = matrix[matrix["split"] == "validation"]
    test = matrix[matrix["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty train, validation, and test splits.")

    supervised_models = {
        "logistic_regression": make_logistic_regression(random_state=random_state),
        "linear_svm": make_linear_svm(random_state=random_state),
        "random_forest": make_random_forest(random_state=random_state),
        "extra_trees": make_extra_trees(random_state=random_state),
        "hist_gradient_boosting": make_hist_gradient_boosting(random_state=random_state),
    }
    anomaly_models = {
        "isolation_forest": make_isolation_forest(random_state=random_state),
        "one_class_svm": make_one_class_svm(),
    }
    rows = []
    score_rows = []
    for model_name, model in supervised_models.items():
        model.fit(train[feature_cols], train["label"])
        validation_scores = predict_scores(model, validation[feature_cols])
        threshold, validation_f1 = best_f1_threshold(validation["label"].to_numpy(), validation_scores)
        test_scores = predict_scores(model, test[feature_cols])
        rows.append(
            _metrics_row(
                model_name,
                test,
                test_scores,
                threshold,
                validation_f1,
                top_k,
                feature_cols,
                n_train=len(train),
                n_validation=len(validation),
            )
        )
        score_rows.append(_score_rows(model_name, test, test_scores, threshold))

    normal_train = train[train["label"].eq(0)]
    if not normal_train.empty:
        for model_name, model in anomaly_models.items():
            model.fit(normal_train[feature_cols])
            validation_scores = predict_anomaly_scores(model, validation[feature_cols])
            threshold, validation_f1 = best_f1_threshold(validation["label"].to_numpy(), validation_scores)
            test_scores = predict_anomaly_scores(model, test[feature_cols])
            rows.append(
                _metrics_row(
                    model_name,
                    test,
                    test_scores,
                    threshold,
                    validation_f1,
                    top_k,
                    feature_cols,
                    n_train=len(normal_train),
                    n_validation=len(validation),
                )
            )
            score_rows.append(_score_rows(model_name, test, test_scores, threshold))

    return pd.DataFrame(rows), pd.concat(score_rows, ignore_index=True)


def make_feature_matrix(samples: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    wide = features.pivot_table(
        index="sample_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="last",
        fill_value=0.0,
    ).reset_index()
    return samples.merge(wide, on="sample_id", how="left").fillna(0.0)


def predict_scores(model, x: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def predict_anomaly_scores(model, x: pd.DataFrame):
    return -model.decision_function(x)


def _metrics_row(
    model_name: str,
    test: pd.DataFrame,
    test_scores,
    threshold: float,
    validation_f1: float,
    top_k: int,
    feature_cols: list[str],
    n_train: int,
    n_validation: int,
) -> dict[str, object]:
    metrics = binary_metrics(test["label"].to_numpy(), test_scores, threshold=threshold)
    metrics.update(precision_recall_at_k(test["label"].to_numpy(), test_scores, k=top_k))
    metrics["model"] = model_name
    metrics["threshold"] = threshold
    metrics["validation_best_f1"] = validation_f1
    metrics["n_train"] = n_train
    metrics["n_validation"] = n_validation
    metrics["n_test"] = len(test)
    metrics["features"] = ",".join(feature_cols)
    return metrics


def _score_rows(model_name: str, test: pd.DataFrame, test_scores, threshold: float) -> pd.DataFrame:
    model_scores = test[["sample_id", "entity_id", "label", "horizon_hours", "prediction_time", "event_time"]].copy()
    model_scores["model"] = model_name
    model_scores["risk_score"] = test_scores
    model_scores["threshold"] = threshold
    return model_scores


if __name__ == "__main__":
    main()
