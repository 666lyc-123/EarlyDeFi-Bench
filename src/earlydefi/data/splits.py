from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    validation: float = 0.1
    test: float = 0.2

    def validate(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("Split ratios must be positive")


def temporal_split(
    samples: pd.DataFrame,
    ratios: SplitRatios = SplitRatios(),
    *,
    group_by_entity: bool = True,
) -> pd.DataFrame:
    """Assign train/validation/test splits using chronological order."""
    ratios.validate()
    required = {"prediction_time", "entity_id"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing columns: {sorted(missing)}")

    out = samples.copy()
    out["prediction_time"] = pd.to_datetime(out["prediction_time"], utc=True)

    if group_by_entity:
        entity_times = (
            out.groupby("entity_id", as_index=False)["prediction_time"]
            .min()
            .sort_values(["prediction_time", "entity_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        entity_times["split"] = _split_labels(len(entity_times), ratios)
        out = out.drop(columns=["split"], errors="ignore").merge(
            entity_times[["entity_id", "split"]],
            on="entity_id",
            how="left",
            validate="many_to_one",
        )
        return out

    out = out.sort_values(["prediction_time", "entity_id"], kind="mergesort").reset_index(drop=True)
    out["split"] = _split_labels(len(out), ratios)
    return out


def stratified_temporal_split(
    samples: pd.DataFrame,
    ratios: SplitRatios = SplitRatios(train=0.6, validation=0.2, test=0.2),
    *,
    group_by_entity: bool = True,
    label_col: str = "label",
) -> pd.DataFrame:
    """Assign chronological splits within each label for small imbalanced pilots."""
    ratios.validate()
    required = {"prediction_time", "entity_id", label_col}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing columns: {sorted(missing)}")

    out = samples.copy()
    out["prediction_time"] = pd.to_datetime(out["prediction_time"], utc=True)
    pieces = []
    if group_by_entity:
        entity_times = (
            out.groupby("entity_id", as_index=False)
            .agg(prediction_time=("prediction_time", "min"), label=(label_col, "max"))
            .sort_values(["label", "prediction_time", "entity_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        for _, group in entity_times.groupby("label", sort=True):
            group = group.sort_values(["prediction_time", "entity_id"], kind="mergesort").reset_index(drop=True)
            group["split"] = _split_labels(len(group), ratios)
            pieces.append(group)
        split_entities = pd.concat(pieces, ignore_index=True)
        return out.drop(columns=["split"], errors="ignore").merge(
            split_entities[["entity_id", "split"]],
            on="entity_id",
            how="left",
            validate="many_to_one",
        )

    for _, group in out.groupby(label_col, sort=True):
        group = group.sort_values(["prediction_time", "entity_id"], kind="mergesort").reset_index(drop=True)
        group["split"] = _split_labels(len(group), ratios)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _split_labels(n_rows: int, ratios: SplitRatios) -> list[str]:
    if n_rows == 0:
        return []

    train_end = int(n_rows * ratios.train)
    validation_end = train_end + int(n_rows * ratios.validation)

    if train_end == 0:
        train_end = 1
    if validation_end <= train_end and n_rows >= 3:
        validation_end = train_end + 1
    if validation_end >= n_rows and n_rows >= 3:
        validation_end = n_rows - 1

    labels = []
    for idx in range(n_rows):
        if idx < train_end:
            labels.append("train")
        elif idx < validation_end:
            labels.append("validation")
        else:
            labels.append("test")
    return labels
