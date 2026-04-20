import hashlib
import time
from types import SimpleNamespace

import aiohttp
import bittensor as bt

from neurons.validators.scoring import capacity


def _sign_request(wallet, miner_hotkey: str, body: bytes) -> dict:
    """Build auth headers matching neurons/miners/worker_auth.py format."""
    nonce = str(int(time.time()))
    body_hash = hashlib.sha256(body).hexdigest()
    validator_hotkey = wallet.hotkey.ss58_address
    message = f"{nonce}.{validator_hotkey}.{miner_hotkey}.{body_hash}"
    signature = wallet.hotkey.sign(message.encode()).hex()

    return {
        "X-Hotkey": validator_hotkey,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "X-Body-Hash": body_hash,
        "Content-Type": "application/json",
    }


def _attach_metadata(
    response,
    worker_url: str,
    status_code: int,
    process_time: float,
    axon_info=None,
) -> None:
    """Set .axon and .dendrite attrs so reward/penalty models work unchanged."""
    axon = getattr(response, "axon", None)
    if axon is None:
        axon = SimpleNamespace()
        try:
            setattr(response, "axon", axon)
        except Exception:
            object.__setattr__(response, "axon", axon)

    dendrite = getattr(response, "dendrite", None)
    if dendrite is None:
        dendrite = SimpleNamespace()
        try:
            setattr(response, "dendrite", dendrite)
        except Exception:
            object.__setattr__(response, "dendrite", dendrite)

    hotkey = worker_url if axon_info is None else axon_info.hotkey
    coldkey = None if axon_info is None else axon_info.coldkey
    axon.hotkey = hotkey
    axon.coldkey = coldkey
    dendrite.status_code = status_code
    dendrite.process_time = process_time


class WorkerClient:
    """HTTP client for calling miner worker APIs with signature auth."""

    def __init__(self, wallet: "bt.Wallet"):
        self.wallet = wallet
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def call_ai_search(
        self,
        worker_url: str,
        synapse,
        axon_info=None,
        uid: int | None = None,
    ):
        """POST to /ai/search, consume streaming response, return populated synapse."""
        if not worker_url:
            _attach_metadata(
                synapse,
                worker_url="",
                status_code=500,
                process_time=0.0,
                axon_info=axon_info,
            )
            if uid is not None:
                await capacity.note_worker_result(uid, "ai_search", False)
            return synapse

        started_at = time.monotonic()
        body = synapse.model_dump_json().encode()
        miner_hotkey = axon_info.hotkey if axon_info else ""
        headers = _sign_request(self.wallet, miner_hotkey, body)
        timeout = aiohttp.ClientTimeout(total=synapse.max_execution_time + 30)

        if getattr(synapse, "axon", None) is None:
            hotkey = axon_info.hotkey if axon_info else worker_url
            coldkey = axon_info.coldkey if axon_info else None
            try:
                synapse.axon = SimpleNamespace(hotkey=hotkey, coldkey=coldkey)
            except Exception:
                object.__setattr__(
                    synapse, "axon", SimpleNamespace(hotkey=hotkey, coldkey=coldkey)
                )

        session = self._get_session()
        success = False
        try:
            async with session.post(
                f"{worker_url}/ai/search",
                data=body,
                headers=headers,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                async for _ in synapse.process_streaming_response(resp):
                    pass

                _attach_metadata(
                    synapse,
                    worker_url=worker_url,
                    status_code=resp.status,
                    process_time=time.monotonic() - started_at,
                    axon_info=axon_info,
                )
                success = True
        except Exception as e:
            _attach_metadata(
                synapse,
                worker_url=worker_url,
                status_code=500,
                process_time=time.monotonic() - started_at,
                axon_info=axon_info,
            )
            bt.logging.error(f"[WorkerClient] ai_search failed for {worker_url}: {e}")

        if uid is not None:
            await capacity.note_worker_result(uid, "ai_search", success)

        return synapse

    async def call_json_search(
        self,
        worker_url: str,
        endpoint: str,
        synapse,
        synapse_model,
        axon_info=None,
        uid: int | None = None,
        search_type: str | None = None,
    ):
        """POST to a JSON endpoint (/twitter/search, /web/search), return parsed synapse."""
        if not worker_url:
            _attach_metadata(
                synapse,
                worker_url="",
                status_code=500,
                process_time=0.0,
                axon_info=axon_info,
            )
            if uid is not None and search_type is not None:
                await capacity.note_worker_result(uid, search_type, False)
            return synapse

        started_at = time.monotonic()
        body = synapse.model_dump_json().encode()
        miner_hotkey = axon_info.hotkey if axon_info else ""
        headers = _sign_request(self.wallet, miner_hotkey, body)
        timeout = aiohttp.ClientTimeout(total=synapse.max_execution_time + 30)

        session = self._get_session()
        success = False
        result = synapse
        try:
            async with session.post(
                f"{worker_url}{endpoint}",
                data=body,
                headers=headers,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                status_code = resp.status
                payload = await resp.json()

            parsed = synapse_model.model_validate(payload)
            _attach_metadata(
                parsed,
                worker_url=worker_url,
                status_code=status_code,
                process_time=time.monotonic() - started_at,
                axon_info=axon_info,
            )
            result = parsed
            success = True
        except Exception as e:
            bt.logging.error(f"[WorkerClient] {endpoint} failed for {worker_url}: {e}")
            _attach_metadata(
                synapse,
                worker_url=worker_url,
                status_code=500,
                process_time=time.monotonic() - started_at,
                axon_info=axon_info,
            )

        if uid is not None and search_type is not None:
            await capacity.note_worker_result(uid, search_type, success)

        return result

    async def call_ai_search_stream(
        self,
        worker_url: str,
        synapse,
        axon_info=None,
        uid: int | None = None,
    ):
        """POST to /ai/search and yield streaming chunks as they arrive, then
        yield the fully-populated synapse at the end. Mirrors the contract of
        ``bt.Dendrite.call_stream`` so the organic path can consume either
        interchangeably."""
        if not worker_url:
            _attach_metadata(
                synapse,
                worker_url="",
                status_code=500,
                process_time=0.0,
                axon_info=axon_info,
            )
            if uid is not None:
                await capacity.note_worker_result(uid, "ai_search", False)
            yield synapse
            return

        started_at = time.monotonic()
        body = synapse.model_dump_json().encode()
        miner_hotkey = axon_info.hotkey if axon_info else ""
        headers = _sign_request(self.wallet, miner_hotkey, body)
        timeout = aiohttp.ClientTimeout(total=synapse.max_execution_time + 30)

        if getattr(synapse, "axon", None) is None:
            hotkey = axon_info.hotkey if axon_info else worker_url
            coldkey = axon_info.coldkey if axon_info else None
            try:
                synapse.axon = SimpleNamespace(hotkey=hotkey, coldkey=coldkey)
            except Exception:
                object.__setattr__(
                    synapse, "axon", SimpleNamespace(hotkey=hotkey, coldkey=coldkey)
                )

        session = self._get_session()
        success = False
        try:
            async with session.post(
                f"{worker_url}/ai/search",
                data=body,
                headers=headers,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                async for chunk in synapse.process_streaming_response(resp):
                    yield chunk

                _attach_metadata(
                    synapse,
                    worker_url=worker_url,
                    status_code=resp.status,
                    process_time=time.monotonic() - started_at,
                    axon_info=axon_info,
                )
                success = True
        except Exception as e:
            _attach_metadata(
                synapse,
                worker_url=worker_url,
                status_code=500,
                process_time=time.monotonic() - started_at,
                axon_info=axon_info,
            )
            bt.logging.error(
                f"[WorkerClient] ai_search_stream failed for {worker_url}: {e}"
            )

        if uid is not None:
            await capacity.note_worker_result(uid, "ai_search", success)

        yield synapse
