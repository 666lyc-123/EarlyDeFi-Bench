from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from eth_utils import keccak

from earlydefi.data.rpc import DEFAULT_RPC_URLS, RpcClient, hex_block


EVENT_TOPICS = {
    "mint": None,
    "burn": None,
    "swap": None,
    "sync": None,
}
EVENT_SIGNATURES = {
    "mint": "Mint(address,uint256,uint256)",
    "burn": "Burn(address,uint256,uint256,address)",
    "swap": "Swap(address,uint256,uint256,uint256,uint256,address)",
    "sync": "Sync(uint112,uint112)",
}
for _name, _signature in EVENT_SIGNATURES.items():
    EVENT_TOPICS[_name] = "0x" + keccak(text=_signature).hex()


def recover_pool_event_times(
    enriched_pairs: pd.DataFrame,
    output_path: str | Path,
    *,
    limit: int = 10,
    block_window_after_creation: int = 250_000,
    block_time_cache_path: str | Path = "data/interim/block_time_cache.jsonl",
    event_types: tuple[str, ...] = ("rug_pull",),
) -> pd.DataFrame:
    """Collect local V2 pool events after pool creation and derive heuristic event time candidates."""
    candidates = enriched_pairs[
        enriched_pairs["event_type"].isin(event_types)
        & enriched_pairs["pool_creation_block"].notna()
        & enriched_pairs["pool_address"].notna()
        & enriched_pairs["chain"].isin(DEFAULT_RPC_URLS)
    ].head(limit)
    clients = {chain: RpcClient(DEFAULT_RPC_URLS[chain]) for chain in candidates["chain"].unique()}
    latest_blocks = {chain: clients[chain].block_number() for chain in clients}
    block_time_cache_path = Path(block_time_cache_path)
    block_time_cache = _read_block_time_cache(block_time_cache_path)

    rows = []
    for idx, row in enumerate(candidates.itertuples(index=False), start=1):
        chain = row.chain
        client = clients[chain]
        from_block = int(row.pool_creation_block)
        to_block = min(from_block + block_window_after_creation, latest_blocks[chain])
        for event_name, topic in EVENT_TOPICS.items():
            params = {
                "address": row.pool_address,
                "fromBlock": hex_block(from_block),
                "toBlock": hex_block(to_block),
                "topics": [topic],
            }
            try:
                logs = client.get_logs(params)
            except Exception as error:
                rows.append(
                    {
                        "chain": chain,
                        "token_address": row.token_address,
                        "pool_address": row.pool_address,
                        "event_name": event_name,
                        "block_number": pd.NA,
                        "tx_hash": pd.NA,
                        "log_index": pd.NA,
                        "event_time": pd.NaT,
                        "status": "rpc_error",
                        "error": str(error)[:500],
                    }
                )
                continue
            block_times, new_block_times = _block_times(client, chain, logs, block_time_cache)
            if new_block_times:
                block_time_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with block_time_cache_path.open("a", encoding="utf-8") as file:
                    for record in new_block_times:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
            for log in logs:
                block_number = int(str(log["blockNumber"]), 16)
                rows.append(
                    {
                        "chain": chain,
                        "token_address": row.token_address,
                        "pool_address": row.pool_address,
                        "event_name": event_name,
                        "block_number": block_number,
                        "tx_hash": log.get("transactionHash"),
                        "log_index": int(str(log.get("logIndex", "0x0")), 16),
                        "event_time": block_times.get(block_number, pd.NaT),
                        "status": "found",
                        "error": pd.NA,
                    }
                )
        if idx % 5 == 0:
            print(f"Pool event recovery processed {idx}/{len(candidates)} pools")

    events = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_path, index=False)
    return events


def recover_pool_event_summaries(
    enriched_pairs: pd.DataFrame,
    output_path: str | Path,
    *,
    limit: int = 10,
    offset: int = 0,
    block_window_after_creation: int = 250_000,
    block_time_cache_path: str | Path = "data/interim/block_time_cache.jsonl",
    cache_path: str | Path = "data/interim/pool_event_summary_cache.jsonl",
    event_types: tuple[str, ...] = ("rug_pull",),
    event_names: tuple[str, ...] = ("mint", "burn", "swap", "sync"),
    retry_errors: bool = False,
    max_block_span: int = 50_000,
) -> pd.DataFrame:
    """Recover compact per-pool event summaries with only first/last block timestamps."""
    candidates = enriched_pairs[
        enriched_pairs["event_type"].isin(event_types)
        & enriched_pairs["pool_creation_block"].notna()
        & enriched_pairs["pool_address"].notna()
        & enriched_pairs["chain"].isin(DEFAULT_RPC_URLS)
    ].copy()
    if offset:
        candidates = candidates.iloc[offset:]
    candidates = candidates.head(limit)
    unknown_events = sorted(set(event_names) - set(EVENT_TOPICS))
    if unknown_events:
        raise ValueError(f"Unknown event names: {unknown_events}")

    clients = {chain: RpcClient(DEFAULT_RPC_URLS[chain]) for chain in candidates["chain"].unique()}
    latest_blocks = {chain: clients[chain].block_number() for chain in clients}
    block_time_cache_path = Path(block_time_cache_path)
    block_time_cache = _read_block_time_cache(block_time_cache_path)
    cache_path = Path(cache_path)
    summary_cache = _read_summary_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in enumerate(candidates.itertuples(index=False), start=1):
        chain = row.chain
        client = clients[chain]
        from_block = int(row.pool_creation_block)
        to_block = min(from_block + block_window_after_creation, latest_blocks[chain])
        for event_name in event_names:
            key = f"{chain}:{str(row.pool_address).lower()}:{event_name}:{from_block}:{to_block}"
            record = summary_cache.get(key)
            if record is None or (retry_errors and record.get("status") == "rpc_error"):
                record = _lookup_event_summary(
                    client,
                    chain=chain,
                    token_address=row.token_address,
                    pool_address=row.pool_address,
                    event_name=event_name,
                    from_block=from_block,
                    to_block=to_block,
                    max_block_span=max_block_span,
                )
                record["key"] = key
                summary_cache[key] = record
                with cache_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            rows.append(record.copy())
        if idx % 10 == 0:
            print(f"Pool event summary recovery processed {idx}/{len(candidates)} pools")

    summaries = pd.DataFrame(rows)
    summaries = _attach_summary_times(summaries, clients, block_time_cache, block_time_cache_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(output_path, index=False)
    return summaries


def build_heuristic_event_times(pool_events: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Use the latest observed pool event as a conservative heuristic event-time candidate."""
    valid = pool_events[pool_events["status"].eq("found") & pool_events["event_time"].notna()].copy()
    if valid.empty:
        out = pd.DataFrame(
            columns=[
                "chain",
                "token_address",
                "pool_address",
                "heuristic_event_time",
                "last_event_name",
                "last_event_block",
                "pool_event_count",
                "heuristic_label_source",
            ]
        )
    else:
        valid = valid.sort_values(["event_time", "block_number", "log_index"])
        grouped = valid.groupby(["chain", "token_address", "pool_address"], as_index=False)
        last = grouped.tail(1).copy()
        counts = grouped.size().rename(columns={"size": "pool_event_count"})
        out = last.merge(counts, on=["chain", "token_address", "pool_address"], how="left")
        out = out.rename(
            columns={
                "event_time": "heuristic_event_time",
                "event_name": "last_event_name",
                "block_number": "last_event_block",
            }
        )[
            [
                "chain",
                "token_address",
                "pool_address",
                "heuristic_event_time",
                "last_event_name",
                "last_event_block",
                "pool_event_count",
            ]
        ]
        out["heuristic_label_source"] = "latest_pool_event_within_window"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def build_heuristic_event_times_from_summaries(
    summaries: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    valid = summaries[summaries["status"].eq("found") & summaries["last_event_time"].notna()].copy()
    if valid.empty:
        out = pd.DataFrame(
            columns=[
                "chain",
                "token_address",
                "pool_address",
                "heuristic_event_time",
                "last_event_name",
                "last_event_block",
                "pool_event_count",
                "heuristic_label_source",
            ]
        )
    else:
        valid = valid.sort_values(["last_event_time", "last_event_block"])
        grouped = valid.groupby(["chain", "token_address", "pool_address"], as_index=False)
        last = grouped.tail(1).copy()
        counts = grouped["event_count"].sum().rename(columns={"event_count": "pool_event_count"})
        out = last.merge(counts, on=["chain", "token_address", "pool_address"], how="left")
        out = out.rename(
            columns={
                "last_event_time": "heuristic_event_time",
                "event_name": "last_event_name",
                "last_event_block": "last_event_block",
            }
        )[
            [
                "chain",
                "token_address",
                "pool_address",
                "heuristic_event_time",
                "last_event_name",
                "last_event_block",
                "pool_event_count",
            ]
        ]
        out["heuristic_label_source"] = "latest_pool_event_summary_within_window"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def _lookup_event_summary(
    client: RpcClient,
    *,
    chain: str,
    token_address: str,
    pool_address: str,
    event_name: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
) -> dict[str, object]:
    try:
        logs = _get_logs_chunked(
            client,
            pool_address=pool_address,
            event_name=event_name,
            from_block=from_block,
            to_block=to_block,
            max_block_span=max_block_span,
        )
    except Exception as error:
        return {
            "chain": chain,
            "token_address": str(token_address).lower(),
            "pool_address": str(pool_address).lower(),
            "event_name": event_name,
            "from_block": from_block,
            "to_block": to_block,
            "event_count": 0,
            "first_event_block": None,
            "last_event_block": None,
            "status": "rpc_error",
            "error": str(error)[:500],
        }

    blocks = sorted(int(str(log["blockNumber"]), 16) for log in logs)
    return {
        "chain": chain,
        "token_address": str(token_address).lower(),
        "pool_address": str(pool_address).lower(),
        "event_name": event_name,
        "from_block": from_block,
        "to_block": to_block,
        "event_count": len(blocks),
        "first_event_block": blocks[0] if blocks else None,
        "last_event_block": blocks[-1] if blocks else None,
        "status": "found",
        "error": None,
    }


def _get_logs_chunked(
    client: RpcClient,
    *,
    pool_address: str,
    event_name: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
) -> list[dict[str, object]]:
    if max_block_span <= 0:
        raise ValueError("max_block_span must be positive")
    logs: list[dict[str, object]] = []
    start = from_block
    while start <= to_block:
        end = min(to_block, start + max_block_span - 1)
        params = {
            "address": pool_address,
            "fromBlock": hex_block(start),
            "toBlock": hex_block(end),
            "topics": [EVENT_TOPICS[event_name]],
        }
        logs.extend(client.get_logs(params))
        start = end + 1
    return logs


def _attach_summary_times(
    summaries: pd.DataFrame,
    clients: dict[str, RpcClient],
    block_time_cache: dict[str, pd.Timestamp],
    block_time_cache_path: Path,
) -> pd.DataFrame:
    if summaries.empty:
        return summaries
    out = summaries.copy()
    for col in ["first_event_time", "last_event_time"]:
        if col not in out.columns:
            out[col] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        else:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    new_block_times = []
    for idx, row in out.iterrows():
        chain = row["chain"]
        client = clients.get(chain)
        if client is None:
            continue
        for block_col, time_col in [("first_event_block", "first_event_time"), ("last_event_block", "last_event_time")]:
            block_number = row.get(block_col)
            if pd.isna(block_number):
                continue
            block_number = int(block_number)
            key = f"{chain}:{block_number}"
            if key not in block_time_cache:
                block = client.get_block(block_number)
                block_time_cache[key] = pd.to_datetime(int(str(block["timestamp"]), 16), unit="s", utc=True)
                new_block_times.append(
                    {
                        "chain": chain,
                        "block_number": block_number,
                        "block_time": block_time_cache[key].isoformat(),
                    }
                )
            out.at[idx, time_col] = block_time_cache[key]
    if new_block_times:
        block_time_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with block_time_cache_path.open("a", encoding="utf-8") as file:
            for record in new_block_times:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def _read_summary_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            existing = cache.get(record["key"])
            if existing is not None and existing.get("status") == "found" and record.get("status") != "found":
                continue
            cache[record["key"]] = record
    return cache


def _block_times(
    client: RpcClient,
    chain: str,
    logs: list[dict[str, object]],
    cache: dict[str, pd.Timestamp],
) -> tuple[dict[int, pd.Timestamp], list[dict[str, object]]]:
    times = {}
    new_records = []
    for block_number in sorted({int(str(log["blockNumber"]), 16) for log in logs}):
        key = f"{chain}:{block_number}"
        if key in cache:
            times[block_number] = cache[key]
            continue
        block = client.get_block(block_number)
        timestamp = pd.to_datetime(int(str(block["timestamp"]), 16), unit="s", utc=True)
        times[block_number] = timestamp
        cache[key] = timestamp
        new_records.append(
            {
                "chain": chain,
                "block_number": block_number,
                "block_time": timestamp.isoformat(),
            }
        )
    return times, new_records


def _read_block_time_cache(path: Path) -> dict[str, pd.Timestamp]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['chain']}:{record['block_number']}"
            cache[key] = pd.to_datetime(record["block_time"], utc=True, errors="coerce")
    return cache
