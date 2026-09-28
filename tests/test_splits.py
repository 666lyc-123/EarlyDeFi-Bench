from __future__ import annotations

import pandas as pd

from earlydefi.data.splits import SplitRatios, stratified_temporal_split, temporal_split


def test_temporal_split_groups_same_entity() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "entity_id": ["a", "a", "b", "b", "c", "c"],
            "prediction_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-03-01",
                    "2024-03-02",
                ],
                utc=True,
            ),
        }
    )

    split = temporal_split(samples, SplitRatios(0.34, 0.33, 0.33), group_by_entity=True)

    assert split.groupby("entity_id")["split"].nunique().max() == 1


def test_temporal_split_orders_by_prediction_time() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["late", "early", "middle"],
            "entity_id": ["c", "a", "b"],
            "prediction_time": pd.to_datetime(
                ["2024-03-01", "2024-01-01", "2024-02-01"],
                utc=True,
            ),
        }
    )

    split = temporal_split(samples, SplitRatios(1 / 3, 1 / 3, 1 / 3), group_by_entity=True)
    entity_split = dict(zip(split["entity_id"], split["split"], strict=True))

    assert entity_split["a"] == "train"
    assert entity_split["b"] == "validation"
    assert entity_split["c"] == "test"


def test_stratified_temporal_split_preserves_label_coverage() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "entity_id": [f"e{i}" for i in range(12)],
            "label": [0] * 6 + [1] * 6,
            "prediction_time": pd.to_datetime(
                [f"2024-01-{i + 1:02d}" for i in range(6)]
                + [f"2024-02-{i + 1:02d}" for i in range(6)],
                utc=True,
            ),
        }
    )

    split = stratified_temporal_split(samples, SplitRatios(0.5, 1 / 6, 1 / 3), group_by_entity=True)

    counts = split.groupby(["split", "label"]).size()
    assert counts.loc[("train", 0)] == 3
    assert counts.loc[("train", 1)] == 3
    assert counts.loc[("validation", 0)] == 1
    assert counts.loc[("validation", 1)] == 1
    assert counts.loc[("test", 0)] == 2
    assert counts.loc[("test", 1)] == 2
