from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from earlydefi.data.pool_events import EVENT_TOPICS
from earlydefi.data.rpc import DEFAULT_RPC_URLS, RpcClient, hex_block


EVENT_NAMES = ("mint", "burn", "swap")
LOOKBACK_HOURS = (1, 6, 24)
FEATURE_NAMES = (
    "event_count",
    "mint_event_count",
    "burn_event_count",
    "swap_event_count",
    "unique_actor_count",
    "swap_sender_count",
    "swap_receiver_count",
    "mint_sender_count",
    "burn_sender_count",
    "burn_receiver_count",
    "burn_swap_actor_overlap",
    "mint_burn_actor_overlap",
    "actor_max_share",
    "actor_hhi",
    "lp_actor_share",
    "burn_event_share",
    "swap_event_share",
    "actor_reuse_rate",
)


def collect_pool_participant_graph_features(
    samples_path: str | Path,
    sample_blocks_path: str | Path,
    output_path: str | Path,
    *,
    cache_path: str | Path = "data/interim/pool_participant_graph_cache.jsonl",
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = False,
    lookback_hours: list[int] | None = None,
    max_block_span: int = 5_000,
    retry_errors: bool = True,
) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    sample_blocks = pd.read_parquet(sample_blocks_path)
    rows = collect_pool_participant_graph_features_frame(
        samples,
        sample_blocks,
        cache_path=cache_path,
        limit=limit,
        offset=offset,
        missing_only=missing_only,
        lookback_hours=lookback_hours,
        max_block_span=max_block_span,
        retry_errors=retry_errors,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_path, index=False)
    return rows


def collect_pool_participant_graph_features_frame(
    samples: pd.DataFrame,
    sample_blocks: pd.DataFrame,
    *,
    cache_path: str | Path,
    limit: int | None = None,
    offset: int = 0,
    missing_only: bool = False,
    lookback_hours: list[int] | None = None,
    max_block_span: int = 5_000,
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
    if max_block_span <= 0:
        raise ValueError("max_block_span must be positive")

    lookback_hours = lookback_hours or list(LOOKBACK_HOURS)
    merged = samples.merge(
        sample_blocks[["sample_id", "prediction_block_estimate", "block_seconds", "status"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    cache_path = Path(cache_path)
    cache = _read_cache(cache_path)
    if missing_only:
        merged = _filter_missing_samples(merged, cache, lookback_hours=lookback_hours)
    if offset:
        merged = merged.iloc[offset:]
    if limit is not None:
        merged = merged.head(limit)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clients = {chain: RpcClient(DEFAULT_RPC_URLS[chain]) for chain in merged["chain"].unique() if chain in DEFAULT_RPC_URLS}
    rows: list[dict[str, object]] = []
    for idx, sample in enumerate(merged.itertuples(index=False), start=1):
        prediction_time = pd.to_datetime(sample.prediction_time, utc=True)
        if getattr(sample, "status") != "found" or pd.isna(sample.prediction_block_estimate):
            rows.extend(_missing_rows(sample.sample_id, prediction_time, lookback_hours, reason="missing_prediction_block"))
            continue
        if sample.chain not in clients:
            rows.extend(_missing_rows(sample.sample_id, prediction_time, lookback_hours, reason="unsupported_chain"))
            continue

        complete = True
        any_rpc_error = False
        block_seconds = float(sample.block_seconds)
        prediction_block = int(sample.prediction_block_estimate)
        window_summaries = _window_summaries_for_sample(
            clients[sample.chain],
            cache,
            cache_path,
            chain=sample.chain,
            pool_address=sample.pool_address,
            prediction_block=prediction_block,
            block_seconds=block_seconds,
            lookback_hours=lookback_hours,
            max_block_span=max_block_span,
            retry_errors=retry_errors,
        )
        for window in lookback_hours:
            summary, status = window_summaries[window]
            complete = complete and status == "found"
            any_rpc_error = any_rpc_error or status == "rpc_error"
            rows.extend(_summary_rows(sample.sample_id, prediction_time, summary, window))
        rows.append(_row(sample.sample_id, "graph_pool_participant_cache_complete", float(complete), prediction_time))
        rows.append(_row(sample.sample_id, "graph_pool_participant_rpc_error", float(any_rpc_error), prediction_time))
        if idx % 25 == 0:
            print(f"Pool participant graph features processed {idx}/{len(merged)} samples")
    return pd.DataFrame(rows)


def _filter_missing_samples(
    merged: pd.DataFrame,
    cache: dict[str, dict[str, object]],
    *,
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
            key = _cache_key(sample.chain, sample.pool_address, from_block, to_block)
            if key not in cache:
                complete = False
                break
        keep.append(not complete)
    return merged.loc[keep].copy()


def _window_summaries_for_sample(
    client: RpcClient,
    cache: dict[str, dict[str, object]],
    cache_path: Path,
    *,
    chain: str,
    pool_address: str,
    prediction_block: int,
    block_seconds: float,
    lookback_hours: list[int],
    max_block_span: int,
    retry_errors: bool,
) -> dict[int, tuple[dict[str, float], str]]:
    windows = {
        window: (
            max(0, prediction_block - int(round(window * 3600 / block_seconds))),
            prediction_block,
        )
        for window in lookback_hours
    }
    out: dict[int, tuple[dict[str, float], str]] = {}
    missing: dict[int, tuple[int, int]] = {}
    for window, (from_block, to_block) in windows.items():
        key = _cache_key(chain, pool_address, from_block, to_block)
        cached = cache.get(key)
        if cached is not None and (not retry_errors or cached.get("status") != "rpc_error"):
            out[window] = (_coerce_summary(cached.get("summary", {})), str(cached.get("status", "found")))
        else:
            missing[window] = (from_block, to_block)
    if not missing:
        return out

    query_from_block = min(from_block for from_block, _ in missing.values())
    query_to_block = prediction_block
    try:
        logs = _get_logs_chunked(
            client,
            pool_address=str(pool_address).lower(),
            from_block=query_from_block,
            to_block=query_to_block,
            max_block_span=max_block_span,
        )
        status = "found"
        error = None
    except Exception as caught:
        logs = []
        status = "rpc_error"
        error = str(caught)[:500]

    for window, (from_block, to_block) in missing.items():
        if status == "found":
            summary = _summarize_logs(_filter_logs_by_block(logs, from_block=from_block, to_block=to_block))
        else:
            summary = _zero_summary()
        record = {
            "key": _cache_key(chain, pool_address, from_block, to_block),
            "chain": chain,
            "pool_address": str(pool_address).lower(),
            "from_block": from_block,
            "to_block": to_block,
            "summary": summary,
            "status": status,
            "error": error,
        }
        cache[record["key"]] = record
        with cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out[window] = (summary, status)
    return out


def _get_logs_chunked(
    client: RpcClient,
    *,
    pool_address: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
) -> list[dict[str, object]]:
    logs: list[dict[str, object]] = []
    start = from_block
    while start <= to_block:
        end = min(to_block, start + max_block_span - 1)
        logs.extend(_get_logs_adaptive(client, pool_address=pool_address, from_block=start, to_block=end))
        start = end + 1
    return logs


def _get_logs_adaptive(
    client: RpcClient,
    *,
    pool_address: str,
    from_block: int,
    to_block: int,
) -> list[dict[str, object]]:
    try:
        return client.get_logs(
            {
                "address": pool_address,
                "fromBlock": hex_block(from_block),
                "toBlock": hex_block(to_block),
                "topics": [[EVENT_TOPICS[event_name] for event_name in EVENT_NAMES]],
            }
        )
    except Exception:
        if to_block <= from_block:
            raise
        midpoint = (from_block + to_block) // 2
        return _get_logs_adaptive(
            client,
            pool_address=pool_address,
            from_block=from_block,
            to_block=midpoint,
        ) + _get_logs_adaptive(
            client,
            pool_address=pool_address,
            from_block=midpoint + 1,
            to_block=to_block,
        )


def _summarize_logs(logs: list[dict[str, object]]) -> dict[str, float]:
    topic_to_event = {str(topic).lower(): name for name, topic in EVENT_TOPICS.items() if name in EVENT_NAMES}
    event_counts: Counter[str] = Counter()
    actor_counts: Counter[str] = Counter()
    role_sets: dict[str, set[str]] = defaultdict(set)
    lp_actor_appearances = 0

    for log in logs:
        topics = [str(topic).lower() for topic in log.get("topics", [])]
        if not topics:
            continue
        event_name = topic_to_event.get(topics[0])
        if event_name is None:
            continue
        event_counts[event_name] += 1
        roles = _roles_for_event(event_name, topics)
        for role_name, address in roles:
            if address is None:
                continue
            role_sets[role_name].add(address)
            role_sets[f"{event_name}_actors"].add(address)
            actor_counts[address] += 1
            if event_name in {"mint", "burn"}:
                lp_actor_appearances += 1

    total_events = sum(event_counts.values())
    total_actor_appearances = sum(actor_counts.values())
    unique_actors = len(actor_counts)
    max_share = max(actor_counts.values()) / total_actor_appearances if total_actor_appearances else 0.0
    hhi = (
        sum((count / total_actor_appearances) ** 2 for count in actor_counts.values())
        if total_actor_appearances
        else 0.0
    )
    lp_share = lp_actor_appearances / total_actor_appearances if total_actor_appearances else 0.0
    reused = sum(1 for count in actor_counts.values() if count > 1)
    actor_reuse_rate = reused / unique_actors if unique_actors else 0.0

    burn_actors = role_sets["burn_actors"]
    swap_actors = role_sets["swap_actors"]
    mint_actors = role_sets["mint_actors"]
    summary = {
        "event_count": float(total_events),
        "mint_event_count": float(event_counts["mint"]),
        "burn_event_count": float(event_counts["burn"]),
        "swap_event_count": float(event_counts["swap"]),
        "unique_actor_count": float(unique_actors),
        "swap_sender_count": float(len(role_sets["swap_sender"])),
        "swap_receiver_count": float(len(role_sets["swap_receiver"])),
        "mint_sender_count": float(len(role_sets["mint_sender"])),
        "burn_sender_count": float(len(role_sets["burn_sender"])),
        "burn_receiver_count": float(len(role_sets["burn_receiver"])),
        "burn_swap_actor_overlap": float(len(burn_actors & swap_actors)),
        "mint_burn_actor_overlap": float(len(mint_actors & burn_actors)),
        "actor_max_share": float(max_share),
        "actor_hhi": float(hhi),
        "lp_actor_share": float(lp_share),
        "burn_event_share": float(event_counts["burn"] / total_events) if total_events else 0.0,
        "swap_event_share": float(event_counts["swap"] / total_events) if total_events else 0.0,
        "actor_reuse_rate": float(actor_reuse_rate),
    }
    return {name: float(summary.get(name, 0.0)) for name in FEATURE_NAMES}


def _filter_logs_by_block(
    logs: list[dict[str, object]],
    *,
    from_block: int,
    to_block: int,
) -> list[dict[str, object]]:
    visible = []
    for log in logs:
        block_number = _log_block_number(log)
        if block_number is None:
            continue
        if from_block <= block_number <= to_block:
            visible.append(log)
    return visible


def _log_block_number(log: dict[str, object]) -> int | None:
    block_number = log.get("blockNumber")
    if block_number is None:
        return None
    try:
        return int(str(block_number), 16)
    except ValueError:
        return None


def _roles_for_event(event_name: str, topics: list[str]) -> list[tuple[str, str | None]]:
    if event_name == "mint":
        return [("mint_sender", _topic_to_address(topics[1] if len(topics) > 1 else None))]
    if event_name == "burn":
        return [
            ("burn_sender", _topic_to_address(topics[1] if len(topics) > 1 else None)),
            ("burn_receiver", _topic_to_address(topics[2] if len(topics) > 2 else None)),
        ]
    if event_name == "swap":
        return [
            ("swap_sender", _topic_to_address(topics[1] if len(topics) > 1 else None)),
            ("swap_receiver", _topic_to_address(topics[2] if len(topics) > 2 else None)),
        ]
    return []


def _topic_to_address(topic: str | None) -> str | None:
    if not topic:
        return None
    clean = str(topic).lower().removeprefix("0x")
    if len(clean) != 64:
        return None
    return "0x" + clean[-40:]


def _summary_rows(
    sample_id: str,
    prediction_time: pd.Timestamp,
    summary: dict[str, float],
    window: int,
) -> list[dict[str, object]]:
    rows = []
    clean = _coerce_summary(summary)
    for feature_name in FEATURE_NAMES:
        rows.append(
            _row(
                sample_id,
                f"graph_pool_participant_{feature_name}_last_{window}h",
                clean[feature_name],
                prediction_time,
            )
        )
    return rows


def _missing_rows(
    sample_id: str,
    prediction_time: pd.Timestamp,
    lookback_hours: list[int],
    *,
    reason: str,
) -> list[dict[str, object]]:
    rows = []
    for window in lookback_hours:
        rows.extend(_summary_rows(sample_id, prediction_time, _zero_summary(), window))
    rows.append(_row(sample_id, "graph_pool_participant_cache_complete", 0.0, prediction_time))
    rows.append(_row(sample_id, "graph_pool_participant_rpc_error", float(reason != "missing_prediction_block"), prediction_time))
    return rows


def _coerce_summary(summary: object) -> dict[str, float]:
    if not isinstance(summary, dict):
        return _zero_summary()
    return {name: float(summary.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}


def _zero_summary() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def _row(sample_id: str, feature_name: str, value: float, observed_at: pd.Timestamp) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "feature_name": feature_name,
        "feature_value": float(value),
        "feature_observed_at": observed_at,
    }


def _cache_key(chain: str, pool_address: str, from_block: int, to_block: int) -> str:
    return f"{chain}:{str(pool_address).lower()}:participants:{from_block}:{to_block}"


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
