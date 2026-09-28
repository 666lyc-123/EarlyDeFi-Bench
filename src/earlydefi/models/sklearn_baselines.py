from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def make_extra_trees(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def make_hist_gradient_boosting(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=0.01,
        random_state=random_state,
    )


def make_linear_svm(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=10_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_isolation_forest(random_state: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        contamination="auto",
        n_jobs=-1,
        random_state=random_state,
    )


def make_one_class_svm() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(gamma="scale", nu=0.1)),
        ]
    )
