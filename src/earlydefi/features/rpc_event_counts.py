from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from earlydefi.data.pool_events import EVENT_TOPICS
from earlydefi.data.rpc import DEFAULT_RPC_URLS, RpcClient, hex_block


EVENT_NAMES = ["mint", "burn", "swap", "sync"]
LOOKBACK_HOURS = [1, 6, 24, 72]


def collect_rpc_event_count_features(
    samples_path: str | Path,
    sample_blocks_path: str | Path,
    output_path: str | Path,
    *,
    cache_path: str | Path = "data/interim/rpc_event_count_cache.jsonl",
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = False,
    event_names: list[str] | None = None,
    lookback_hours: list[int] | None = None,
    max_block_span: int = 2_000,
    retry_errors: bool = True,
) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    sample_blocks = pd.read_parquet(sample_blocks_path)
    rows = collect_rpc_event_count_features_frame(
        samples,
        sample_blocks,
        cache_path=cache_path,
        limit=limit,
        offset=offset,
        missing_only=missing_only,
        event_names=event_names,
        lookback_hours=lookback_hours,
        max_block_span=max_block_span,
        retry_errors=retry_errors,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_path, index=False)
    return rows


def collect_rpc_event_count_features_frame(
    samples: pd.DataFrame,
    sample_blocks: pd.DataFrame,
    *,
    cache_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = False,
    event_names: list[str] | None = None,
    lookback_hours: list[int] | None = None,
    max_block_span: int = 2_000,
    retry_errors: bool = True,
) -> pd.DataFrame:
    required_samples = {"sample_id", "chain", "pool_address", "prediction_time"}
    required_blocks = {"sample_id", "prediction_block_estimate", "block_seconds", "status"}
    missing_samples = required_samples - set(samples.columns)
    missing_blocks = required_blocks - set(sample_blocks.columns)
    if missing_samples:
        raise ValueError(f"samples is missing columns: {sorted(missing_samples)}")
    if missing_blocks:
        raise ValueError(f"sample_blocks is missing columns: {sorted(missing_blocks)}")

    merged = samples.merge(
        sample_blocks[["sample_id", "prediction_block_estimate", "block_seconds", "status"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    event_names = event_names or EVENT_NAMES
    lookback_hours = lookback_hours or LOOKBACK_HOURS
    unknown_events = sorted(set(event_names) - set(EVENT_NAMES))
    if unknown_events:
        raise ValueError(f"Unknown event names: {unknown_events}")

    cache_path = Path(cache_path)
    cache = _read_cache(cache_path)
    if missing_only:
        merged = _filter_missing_samples(
            merged,
            cache,
            event_names=event_names,
            lookback_hours=lookback_hours,
        )
    if offset:
        merged = merged.iloc[offset:]
    if limit is not None:
        merged = merged.head(limit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clients = {chain: RpcClient(DEFAULT_RPC_URLS[chain]) for chain in merged["chain"].unique() if chain in DEFAULT_RPC_URLS}
    rows = []
    for idx, sample in enumerate(merged.itertuples(index=False), start=1):
        prediction_time = pd.to_datetime(sample.prediction_time, utc=True)
        if getattr(sample, "status") != "found" or pd.isna(sample.prediction_block_estimate):
            rows.extend(
                _missing_rows(
                    sample.sample_id,
                    prediction_time,
                    "missing_prediction_block",
                    event_names=event_names,
                    lookback_hours=lookback_hours,
                )
            )
            continue
        if sample.chain not in clients:
            rows.extend(
                _missing_rows(
                    sample.sample_id,
                    prediction_time,
                    "unsupported_chain",
                    event_names=event_names,
                    lookback_hours=lookback_hours,
                )
            )
            continue

        block_seconds = float(sample.block_seconds)
        prediction_block = int(sample.prediction_block_estimate)
        status_values = []
        for window in lookback_hours:
            from_block = max(0, prediction_block - int(round(window * 3600 / block_seconds)))
            to_block = max(from_block, prediction_block)
            total = 0
            for event_name in event_names:
                count, status = _cached_log_count(
                    clients[sample.chain],
                    cache,
                    cache_path,
                    chain=sample.chain,
                    pool_address=sample.pool_address,
                    event_name=event_name,
                    from_block=from_block,
                    to_block=to_block,
                    max_block_span=max_block_span,
                    retry_errors=retry_errors,
                )
                total += count
                status_values.append(status)
                rows.append(_row(sample.sample_id, f"rpc_pool_events_{event_name}_count_last_{window}h", count, prediction_time))
            rows.append(_row(sample.sample_id, f"rpc_pool_events_count_last_{window}h", float(total), prediction_time))
            rows.append(_row(sample.sample_id, f"rpc_pool_events_rate_last_{window}h", float(total / window), prediction_time))
        rows.append(_row(sample.sample_id, "rpc_pool_events_available", float(any(status == "found" for status in status_values)), prediction_time))
        rows.append(_row(sample.sample_id, "rpc_pool_events_rpc_error", float(any(status == "rpc_error" for status in status_values)), prediction_time))
        if idx % 10 == 0:
            print(f"RPC event count features processed {idx}/{len(merged)} samples")
    return pd.DataFrame(rows)


def _filter_missing_samples(
    merged: pd.DataFrame,
    cache: dict[str, dict[str, object]],
    *,
    event_names: list[str],
    lookback_hours: list[int],
) -> pd.DataFrame:
    keep = []
    for sample in merged.itertuples(index=False):
        if getattr(sample, "status") != "found" or pd.isna(sample.prediction_block_estimate):
            keep.append(True)
            continue
        block_seconds = float(sample.block_seconds)
        prediction_block = int(sample.prediction_block_estimate)
        complete = True
        for window in lookback_hours:
            from_block = max(0, prediction_block - int(round(window * 3600 / block_seconds)))
            to_block = max(from_block, prediction_block)
            for event_name in event_names:
                key = f"{sample.chain}:{str(sample.pool_address).lower()}:{event_name}:{from_block}:{to_block}"
                if key not in cache:
                    complete = False
                    break
            if not complete:
                break
        keep.append(not complete)
    return merged.loc[keep].copy()


def _cached_log_count(
    client: RpcClient,
    cache: dict[str, dict[str, object]],
    cache_path: Path,
    *,
    chain: str,
    pool_address: str,
    event_name: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
    retry_errors: bool,
) -> tuple[float, str]:
    key = f"{chain}:{str(pool_address).lower()}:{event_name}:{from_block}:{to_block}"
    if key in cache and (not retry_errors or cache[key].get("status") != "rpc_error"):
        record = cache[key]
        return float(record["count"]), str(record["status"])

    try:
        count = _count_logs_chunked(
            client,
            pool_address=str(pool_address).lower(),
            topic=EVENT_TOPICS[event_name],
            from_block=from_block,
            to_block=to_block,
            max_block_span=max_block_span,
        )
        record = {
            "key": key,
            "chain": chain,
            "pool_address": str(pool_address).lower(),
            "event_name": event_name,
            "from_block": from_block,
            "to_block": to_block,
            "count": count,
            "status": "found",
            "error": None,
        }
    except Exception as error:
        record = {
            "key": key,
            "chain": chain,
            "pool_address": str(pool_address).lower(),
            "event_name": event_name,
            "from_block": from_block,
            "to_block": to_block,
            "count": 0,
            "status": "rpc_error",
            "error": str(error)[:500],
        }
    cache[key] = record
    with cache_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return float(record["count"]), str(record["status"])


def _count_logs_chunked(
    client: RpcClient,
    *,
    pool_address: str,
    topic: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
) -> int:
    if to_block < from_block:
        return 0
    total = 0
    start = from_block
    while start <= to_block:
        end = min(to_block, start + max_block_span - 1)
        logs = client.get_logs(
            {
                "address": pool_address,
                "fromBlock": hex_block(start),
                "toBlock": hex_block(end),
                "topics": [topic],
            }
        )
        total += len(logs)
        start = end + 1
    return total


def _missing_rows(
    sample_id: str,
    prediction_time: pd.Timestamp,
    reason: str,
    *,
    event_names: list[str] | None = None,
    lookback_hours: list[int] | None = None,
) -> list[dict[str, object]]:
    event_names = event_names or EVENT_NAMES
    lookback_hours = lookback_hours or LOOKBACK_HOURS
    rows = []
    for window in lookback_hours:
        for event_name in event_names:
            rows.append(_row(sample_id, f"rpc_pool_events_{event_name}_count_last_{window}h", 0.0, prediction_time))
        rows.append(_row(sample_id, f"rpc_pool_events_count_last_{window}h", 0.0, prediction_time))
        rows.append(_row(sample_id, f"rpc_pool_events_rate_last_{window}h", 0.0, prediction_time))
    rows.append(_row(sample_id, "rpc_pool_events_available", 0.0, prediction_time))
    rows.append(_row(sample_id, "rpc_pool_events_rpc_error", float(reason != "missing_prediction_block"), prediction_time))
    return rows


def _row(sample_id: str, feature_name: str, value: float, observed_at: pd.Timestamp) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "feature_name": feature_name,
        "feature_value": float(value),
        "feature_observed_at": observed_at,
    }


def _read_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        return {}
    cache = {}
    with cache_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["key"]] = record
    return cache
