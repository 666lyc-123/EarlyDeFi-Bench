from __future__ import annotations

import pandas as pd

from earlydefi.data.rpc import topic_address
from earlydefi.features.pool_participant_graph import collect_pool_participant_graph_features_frame


class DummyClient:
    def __init__(self, url: str):
        self.url = url

    def get_logs(self, params):
        sender_a = topic_address("0x00000000000000000000000000000000000000aa")
        sender_b = topic_address("0x00000000000000000000000000000000000000bb")
        return [
            {"blockNumber": "0x64", "topics": ["0xswap", sender_a, sender_b]},
            {"blockNumber": "0x64", "topics": ["0xburn", sender_a, sender_a]},
            {"blockNumber": "0x64", "topics": ["0xmint", sender_b]},
        ]


def test_pool_participant_graph_parses_indexed_event_addresses(monkeypatch, tmp_path) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "chain": ["test"],
            "pool_address": ["0xpool"],
            "prediction_time": [pd.Timestamp("2021-01-01T00:00:00Z")],
        }
    )
    sample_blocks = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prediction_block_estimate": [100],
            "block_seconds": [3600.0],
            "status": ["found"],
        }
    )
    monkeypatch.setattr("earlydefi.features.pool_participant_graph.DEFAULT_RPC_URLS", {"test": "http://example"})
    monkeypatch.setattr("earlydefi.features.pool_participant_graph.RpcClient", DummyClient)
    monkeypatch.setattr(
        "earlydefi.features.pool_participant_graph.EVENT_TOPICS",
        {"mint": "0xmint", "burn": "0xburn", "swap": "0xswap", "sync": "0xsync"},
    )

    features = collect_pool_participant_graph_features_frame(
        samples,
        sample_blocks,
        cache_path=tmp_path / "participants.jsonl",
        lookback_hours=[1],
        max_block_span=100000,
    )
    values = features.set_index("feature_name")["feature_value"]

    assert values["graph_pool_participant_event_count_last_1h"] == 3.0
    assert values["graph_pool_participant_unique_actor_count_last_1h"] == 2.0
    assert values["graph_pool_participant_burn_swap_actor_overlap_last_1h"] == 1.0
    assert values["graph_pool_participant_mint_burn_actor_overlap_last_1h"] == 0.0
    assert values["graph_pool_participant_actor_max_share_last_1h"] == 0.6
    assert values["graph_pool_participant_cache_complete"] == 1.0


def test_pool_participant_graph_missing_only_skips_complete_cache(monkeypatch, tmp_path) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "chain": ["test"],
            "pool_address": ["0xpool"],
            "prediction_time": [pd.Timestamp("2021-01-01T00:00:00Z")],
        }
    )
    sample_blocks = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "prediction_block_estimate": [100],
            "block_seconds": [3600.0],
            "status": ["found"],
        }
    )
    cache_path = tmp_path / "participants.jsonl"
    cache_path.write_text(
        '{"key":"test:0xpool:participants:99:100","chain":"test","pool_address":"0xpool","from_block":99,"to_block":100,"summary":{"event_count":1},"status":"found","error":null}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("earlydefi.features.pool_participant_graph.DEFAULT_RPC_URLS", {"test": "http://example"})

    class FailingClient:
        def __init__(self, url: str):
            self.url = url

        def get_logs(self, params):
            raise AssertionError("missing_only should not call RPC for complete cache")

    monkeypatch.setattr("earlydefi.features.pool_participant_graph.RpcClient", FailingClient)

    features = collect_pool_participant_graph_features_frame(
        samples,
        sample_blocks,
        cache_path=cache_path,
        lookback_hours=[1],
        missing_only=True,
    )

    assert features.empty
