from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

try:
    from bittensor_wallet import Keypair
except ImportError:
    from bittensor import Keypair

TOLERANCE_S = 60
NONCE_TTL_S = 120
NONCE = re.compile(r"[0-9a-f]{32}")
MAX_HOTKEY_CHARS = 64


@dataclass
class Caller:
    hotkey: str
    uid: int | None
    is_validator: bool
    requested_at: float
    is_admin: bool = False


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
    def __init__(self, registry, nonces, admins: frozenset[str] = frozenset()):
        self.registry = registry
        self.nonces = nonces
        self.admins = admins

    async def __call__(self, request: Request) -> Caller:
        headers = request.headers
        hotkey = headers.get("X-Hotkey")
        timestamp = headers.get("X-Timestamp")
        nonce = headers.get("X-Nonce")
        signature = headers.get("X-Signature")
        if not all((hotkey, timestamp, nonce, signature)):
            raise HTTPException(401, "missing auth headers")
        if (
            len(hotkey) > MAX_HOTKEY_CHARS
            or not NONCE.fullmatch(nonce)
            or not timestamp.isdigit()
        ):
            raise HTTPException(401, "malformed auth headers")

        sent_at = int(timestamp)
        if abs(time.time() - sent_at) > TOLERANCE_S:
            raise HTTPException(401, "timestamp outside tolerance")

        body = await request.body()
        payload = signing_payload(
            request.method, request.url.path, body, timestamp, nonce
        )
        if not verify_signature(hotkey, payload, signature):
            raise HTTPException(401, "bad signature")

        admin = hotkey in self.admins
        entry = None if admin else await self.registry.lookup(hotkey)
        if entry is None and not admin:
            raise HTTPException(403, "hotkey is not registered on this subnet")

        # Claimed last so unauthenticated requests cannot fill the nonce cache.
        if not await self.nonces.claim(hotkey, nonce, NONCE_TTL_S):
            raise HTTPException(401, "nonce already used")

        return Caller(
            hotkey=hotkey,
            uid=entry.uid if entry else None,
            is_validator=bool(entry and entry.is_validator),
            requested_at=sent_at,
            is_admin=admin,
        )
