from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_RPC_URLS = {
    "ethereum": "https://ethereum.publicnode.com",
    "bsc": "https://bsc-dataseed.binance.org",
}


@dataclass(frozen=True)
class RpcClient:
    url: str
    sleep_seconds: float = 0.05
    retries: int = 3

    def call(self, method: str, params: list[object]) -> object:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "EarlyDeFi-Bench research bot"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                time.sleep(self.sleep_seconds)
                break
            except (TimeoutError, URLError, OSError) as error:
                last_error = error
                if attempt >= self.retries:
                    raise
                time.sleep(self.sleep_seconds * (2 ** attempt + 1))
        else:
            raise RuntimeError(f"RPC request failed: {last_error}")
        if "error" in payload:
            raise RuntimeError(f"RPC error from {self.url}: {payload['error']}")
        return payload.get("result")

    def batch_call(self, calls: list[tuple[str, list[object]]]) -> list[object]:
        if not calls:
            return []
        body = json.dumps(
            [
                {"jsonrpc": "2.0", "id": idx, "method": method, "params": params}
                for idx, (method, params) in enumerate(calls, start=1)
            ]
        ).encode()
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "EarlyDeFi-Bench research bot"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                time.sleep(self.sleep_seconds)
                break
            except (TimeoutError, URLError, OSError) as error:
                last_error = error
                if attempt >= self.retries:
                    raise
                time.sleep(self.sleep_seconds * (2 ** attempt + 1))
        else:
            raise RuntimeError(f"RPC batch request failed: {last_error}")
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected RPC batch payload from {self.url}: {payload}")
        by_id = {int(item["id"]): item for item in payload}
        results = []
        for idx in range(1, len(calls) + 1):
            item = by_id.get(idx)
            if item is None:
                raise RuntimeError(f"Missing RPC batch response id {idx} from {self.url}")
            if "error" in item:
                raise RuntimeError(f"RPC batch error from {self.url}: {item['error']}")
            results.append(item.get("result"))
        return results

    def block_number(self) -> int:
        return int(str(self.call("eth_blockNumber", [])), 16)

    def get_logs(self, params: dict[str, object]) -> list[dict[str, object]]:
        result = self.call("eth_getLogs", [params])
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected eth_getLogs result: {result}")
        return result

    def get_code(self, address: str, block_number: int | str = "latest") -> str:
        block_tag = hex_block(block_number) if isinstance(block_number, int) else block_number
        result = self.call("eth_getCode", [address, block_tag])
        return str(result)

    def get_block(self, block_number: int, *, full_transactions: bool = False) -> dict[str, object]:
        result = self.call("eth_getBlockByNumber", [hex_block(block_number), full_transactions])
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected eth_getBlockByNumber result: {result}")
        return result

    def get_transaction_receipt(self, tx_hash: str) -> dict[str, object]:
        result = self.call("eth_getTransactionReceipt", [tx_hash])
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected eth_getTransactionReceipt result: {result}")
        return result


def topic_address(address: str) -> str:
    clean = address.lower().removeprefix("0x")
    if len(clean) != 40:
        raise ValueError(f"Invalid EVM address: {address}")
    return "0x" + clean.rjust(64, "0")


def hex_block(number: int) -> str:
    return hex(number)
