from __future__ import annotations

import asyncio
import hashlib
import secrets
import time

import httpx
from bittensor import Keypair


class Client:
    def __init__(self, api: str, seed_uri: str):
        self.api = api.rstrip("/")
        self.key = Keypair.create_from_uri(seed_uri)
        self.hotkey = self.key.ss58_address
        self.http = httpx.AsyncClient(timeout=30)

    def _headers(self, method: str, path: str, body: bytes) -> dict:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        digest = hashlib.sha256(body).hexdigest()
        payload = f"{method}\n{path}\n{digest}\n{timestamp}\n{nonce}".encode()
        return {
            "X-Hotkey": self.hotkey,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": self.key.sign(payload).hex(),
            "Content-Type": "application/json",
        }

    async def post(self, path: str, body: dict | None = None) -> httpx.Response:
        raw = httpx.Request("POST", "http://x", json=body or {}).content
        return await self.http.post(
            f"{self.api}{path}", content=raw, headers=self._headers("POST", path, raw)
        )

    async def lease(self) -> dict:
        return (await self.post("/v1/tasks/lease")).json()

    async def complete(self, task_id: str, records: list[dict]) -> dict:
        return (
            await self.post(f"/v1/tasks/{task_id}/complete", {"records": records})
        ).json()

    async def aclose(self) -> None:
        await self.http.aclose()


def fetch(url: str) -> dict:
    return {
        "url": url,
        "status": 200,
        "sha256": hashlib.sha256(url.encode()).hexdigest(),
    }


class Miner:
    name = "miner"

    def __init__(self, client: Client, seconds: float = 8.0):
        self.client = client
        self.seconds = seconds
        self.leased = 0
        self.completed = 0
        self.refusals: dict[str, int] = {}
        self.polls = 0
        self.receipts: list[dict] = []

    async def run(self) -> None:
        deadline = time.time() + self.seconds
        while time.time() < deadline:
            self.polls += 1
            answer = await self.client.lease()
            if answer.get("receipt"):
                self.receipts.append(answer["receipt"])
            if not answer.get("task"):
                code = (answer.get("refusal") or {}).get("code", "UNKNOWN")
                self.refusals[code] = self.refusals.get(code, 0) + 1
                await asyncio.sleep(0.2)
                continue
            self.leased += 1
            await self.handle(answer["task"])
        await self.client.aclose()

    async def handle(self, task: dict) -> None:
        raise NotImplementedError

    def report(self) -> str:
        refusals = (
            ", ".join(f"{k}×{v}" for k, v in sorted(self.refusals.items())) or "none"
        )
        return (
            f"{self.name:<12} polls={self.polls:<4} leased={self.leased:<3} "
            f"completed={self.completed:<3} refusals: {refusals}"
        )


class Honest(Miner):
    name = "honest"

    async def handle(self, task: dict) -> None:
        records = [fetch(u["url"]) for u in task["urls"]]
        await self.client.complete(task["task_id"], records)
        self.completed += 1


class Hoarder(Miner):
    name = "hoarder"

    async def handle(self, task: dict) -> None:
        await asyncio.sleep(0)


class Omitter(Miner):
    name = "omitter"

    async def handle(self, task: dict) -> None:
        urls = task["urls"]
        kept = urls[: max(1, int(len(urls) * 0.8))]
        await self.client.complete(task["task_id"], [fetch(u["url"]) for u in kept])
        self.completed += 1


class Fabricator(Miner):
    name = "fabricator"

    async def handle(self, task: dict) -> None:
        records = []
        for i, url in enumerate(task["urls"]):
            record = fetch(url["url"])
            if i % 10 == 0:
                record["sha256"] = hashlib.sha256(b"invented").hexdigest()
            records.append(record)
        await self.client.complete(task["task_id"], records)
        self.completed += 1


class PollSpammer(Miner):
    name = "pollspammer"

    async def run(self) -> None:
        deadline = time.time() + self.seconds
        while time.time() < deadline:
            self.polls += 1
            answer = await self.client.lease()
            if answer.get("receipt"):
                self.receipts.append(answer["receipt"])
            if answer.get("task"):
                self.leased += 1
                records = [fetch(u["url"]) for u in answer["task"]["urls"]]
                await self.client.complete(answer["task"]["task_id"], records)
                self.completed += 1
            else:
                code = (answer.get("refusal") or {}).get("code", "UNKNOWN")
                self.refusals[code] = self.refusals.get(code, 0) + 1
        await self.client.aclose()
