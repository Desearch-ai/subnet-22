"""
Worker API authentication: signature verification, stake checking, metagraph sync.

Message format: {nonce}.{validator_hotkey}.{miner_hotkey}.{body_hash}
"""

import asyncio
import hashlib
import time
from typing import Optional

import bittensor as bt
from bittensor.core.metagraph import AsyncMetagraph
from fastapi import HTTPException, Request
from substrateinterface import Keypair

import desearch

MINER_HOTKEY: str = ""
NETWORK: str = ""
NETUID: int = 0
METAGRAPH: Optional[AsyncMetagraph] = None
METAGRAPH_SYNC_INTERVAL = 30 * 60


def init(miner_hotkey: str, network: str) -> None:
    global MINER_HOTKEY, NETWORK
    MINER_HOTKEY = miner_hotkey
    NETWORK = network
    bt.logging.info(
        f"[WorkerAuth] miner_hotkey={MINER_HOTKEY[:12]}... network={NETWORK}"
    )


async def init_metagraph(network: str, netuid: int) -> None:
    global METAGRAPH, NETUID
    NETUID = netuid
    METAGRAPH = AsyncMetagraph(netuid=netuid, network=network)
    try:
        await METAGRAPH.sync()
        bt.logging.info(
            f"[WorkerAuth] metagraph synced: "
            f"{len(METAGRAPH.axons)} neurons on netuid={netuid}"
        )
    except Exception as e:
        bt.logging.error(
            f"[WorkerAuth] initial metagraph sync failed: {e}. "
            f"Starting without stake checks, will retry in background."
        )
        METAGRAPH = None


async def sync_metagraph_loop() -> None:
    global METAGRAPH
    while True:
        await asyncio.sleep(METAGRAPH_SYNC_INTERVAL)
        try:
            if METAGRAPH is None:
                METAGRAPH = AsyncMetagraph(netuid=NETUID, network=NETWORK)
            await METAGRAPH.sync()
            bt.logging.info(
                f"[WorkerAuth] metagraph resynced: {len(METAGRAPH.axons)} neurons"
            )
        except Exception as e:
            bt.logging.error(f"[WorkerAuth] metagraph resync failed: {e}")


def verify_signature(
    *,
    validator_hotkey: str,
    miner_hotkey: str,
    nonce: str,
    signature_hex: str,
    body: bytes,
    nonce_tolerance: int = 60,
) -> None:
    """Verify cryptographic auth. Raises ValueError on failure."""
    now = int(time.time())
    try:
        nonce_int = int(nonce)
    except (ValueError, TypeError):
        raise ValueError("Invalid nonce format")

    if abs(now - nonce_int) > nonce_tolerance:
        raise ValueError(
            f"Nonce expired: delta={abs(now - nonce_int)}s > {nonce_tolerance}s"
        )

    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{nonce}.{validator_hotkey}.{miner_hotkey}.{body_hash}"

    sig_bytes = bytes.fromhex(signature_hex.removeprefix("0x"))
    keypair = Keypair(ss58_address=validator_hotkey)

    if not keypair.verify(message.encode(), sig_bytes):
        raise ValueError("Signature verification failed")


def check_stake(hotkey: str, metagraph: AsyncMetagraph, network: str) -> None:
    """Check that hotkey is registered and meets stake requirements.
    Raises ValueError on failure.
    """
    if hotkey in desearch.BLACKLISTED_KEYS:
        raise ValueError(f"Blacklisted hotkey: {hotkey}")

    uid: Optional[int] = None
    for _uid, _axon in enumerate(metagraph.axons):
        if _axon.hotkey == hotkey:
            uid = _uid
            break

    if uid is None:
        raise ValueError(f"Unregistered hotkey: {hotkey}")

    if network == "finney":
        alpha_stake = float(metagraph.alpha_stake[uid].item())
        total_stake = float(metagraph.total_stake[uid].item())

        if (
            alpha_stake < desearch.MIN_ALPHA_STAKE
            or total_stake < desearch.MIN_TOTAL_STAKE
        ):
            raise ValueError(
                f"Low stake: alpha_stake={alpha_stake} < {desearch.MIN_ALPHA_STAKE} "
                f"or total_stake={total_stake} < {desearch.MIN_TOTAL_STAKE}"
            )


async def verify_worker_request(request: Request) -> str:
    """FastAPI dependency — verifies signature and stake. Returns validator hotkey."""
    hotkey = request.headers.get("X-Hotkey", "")
    nonce = request.headers.get("X-Nonce", "")
    signature = request.headers.get("X-Signature", "")
    body_hash = request.headers.get("X-Body-Hash", "")

    if not all([hotkey, nonce, signature, body_hash]):
        raise HTTPException(status_code=401, detail="Missing auth headers")

    body = await request.body()

    try:
        verify_signature(
            validator_hotkey=hotkey,
            miner_hotkey=MINER_HOTKEY,
            nonce=nonce,
            signature_hex=signature,
            body=body,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if METAGRAPH is not None:
        try:
            check_stake(hotkey, METAGRAPH, NETWORK)
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))

    return hotkey
