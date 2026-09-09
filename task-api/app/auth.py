from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from bittensor import Keypair
from fastapi import HTTPException, Request

TOLERANCE_S = 60
NONCE_TTL_S = 120


@dataclass
class Caller:
    hotkey: str
    uid: int | None
    is_validator: bool
    requested_at: float


def signing_payload(
    method: str, path: str, body: bytes, timestamp: str, nonce: str
) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return f"{method}\n{path}\n{digest}\n{timestamp}\n{nonce}".encode()


def verify_signature(hotkey: str, payload: bytes, signature: str) -> bool:
    try:
        return Keypair(ss58_address=hotkey).verify(payload, bytes.fromhex(signature))
    except Exception:
        return False


class Authenticator:
    def __init__(self, registry, nonces):
        self.registry = registry
        self.nonces = nonces

    async def __call__(self, request: Request) -> Caller:
        headers = request.headers
        hotkey = headers.get("X-Hotkey")
        timestamp = headers.get("X-Timestamp")
        nonce = headers.get("X-Nonce")
        signature = headers.get("X-Signature")
        if not all((hotkey, timestamp, nonce, signature)):
            raise HTTPException(401, "missing auth headers")

        try:
            sent_at = int(timestamp)
        except ValueError:
            raise HTTPException(401, "bad timestamp")
        if abs(time.time() - sent_at) > TOLERANCE_S:
            raise HTTPException(401, "timestamp outside tolerance")

        if not await self.nonces.claim(hotkey, nonce, NONCE_TTL_S):
            raise HTTPException(401, "nonce already used")

        body = await request.body()
        payload = signing_payload(
            request.method, request.url.path, body, timestamp, nonce
        )
        if not verify_signature(hotkey, payload, signature):
            raise HTTPException(401, "bad signature")

        entry = await self.registry.lookup(hotkey)
        if entry is None:
            raise HTTPException(403, "hotkey is not registered on this subnet")

        return Caller(
            hotkey=hotkey,
            uid=entry.uid,
            is_validator=entry.is_validator,
            requested_at=sent_at,
        )
