from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time

import aiohttp
from yarl import URL

try:
    from bittensor_wallet import Keypair
except ImportError:
    from bittensor import Keypair


class TaskApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}" if status else detail)
        self.status = status
        self.detail = detail


class TaskApiClient:
    def __init__(self, api: str, key: Keypair | str, timeout: float = 30.0):
        self.key = Keypair.create_from_uri(key) if isinstance(key, str) else key
        self.hotkey: str = self.key.ss58_address
        self.api = api.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: aiohttp.ClientSession | None = None

    async def post(self, path: str, body: dict | None = None) -> dict:
        raw = json.dumps(body or {}, separators=(",", ":")).encode()
        return await self._send("POST", path, raw)

    async def get(self, path: str) -> dict:
        return await self._send("GET", path, b"")

    async def aclose(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def __aenter__(self) -> TaskApiClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    async def _send(self, method: str, path: str, body: bytes) -> dict:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        url = URL(self.api + path, encoded=True)
        headers = self._auth(method, url.path, body)
        if body:
            headers["Content-Type"] = "application/json"

        try:
            async with self.session.request(
                method, url, data=body or None, headers=headers
            ) as response:
                raw = await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise TaskApiError(0, f"{type(exc).__name__}: {exc}") from exc
        if response.status >= 400:
            raise TaskApiError(response.status, _detail(raw))
        return json.loads(raw)

    def _auth(self, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        digest = hashlib.sha256(body).hexdigest()
        payload = f"{method}\n{path}\n{digest}\n{timestamp}\n{nonce}".encode()
        return {
            "X-Hotkey": self.hotkey,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": self.key.sign(payload).hex(),
        }


def _detail(raw: bytes) -> str:
    try:
        detail = json.loads(raw)["detail"]
    except (ValueError, KeyError, TypeError):
        return raw.decode("utf-8", "replace")
    return detail if isinstance(detail, str) else json.dumps(detail)
