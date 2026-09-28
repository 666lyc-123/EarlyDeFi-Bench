from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_trapdoor_positive_pairs(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = _normalize_pair_columns(raw)
    raw["chain"] = "ethereum"
    raw["event_type"] = "rug_pull"
    raw["label"] = 1
    raw["source_name"] = "bsdp2023/trapdoor_data"
    raw["label_source"] = "confirmed"
    raw["trapdoor_type"] = raw["type"].astype("string")
    return _to_pair_labels(raw)


def load_trapdoor_negative_pairs(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = _normalize_pair_columns(raw)
    raw["chain"] = "ethereum"
    raw["event_type"] = "normal"
    raw["label"] = 0
    raw["source_name"] = "bsdp2023/trapdoor_data_non_malicious"
    raw["label_source"] = "curated"
    raw["trapdoor_type"] = pd.NA
    return _to_pair_labels(raw)


def build_trapdoor_pair_labels(raw_root: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw_root = Path(raw_root)
    positives = load_trapdoor_positive_pairs(raw_root / "trapdoor_data_repo" / "trapdoor_list.csv")
    negatives = load_trapdoor_negative_pairs(raw_root / "trapdoor_data_repo" / "non_malicious_list.csv")
    labels = pd.concat([positives, negatives], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    return labels


def _normalize_pair_columns(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"token_address", "pool_address"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Trapdoor data missing columns: {sorted(missing)}")
    out = raw.copy()
    out["token_address"] = out["token_address"].astype("string").str.lower()
    out["pool_address"] = out["pool_address"].astype("string").str.lower()
    return out


def _to_pair_labels(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": raw["chain"] + ":" + raw["token_address"],
            "entity_type": "token_pool",
            "chain": raw["chain"],
            "token_address": raw["token_address"],
            "pool_address": raw["pool_address"],
            "event_type": raw["event_type"],
            "event_time": pd.NaT,
            "label": raw["label"].astype(int),
            "label_source": raw["label_source"],
            "source_name": raw["source_name"],
            "trapdoor_type": raw["trapdoor_type"],
        }
    )
