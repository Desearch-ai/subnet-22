import asyncio
from typing import Optional

import aiohttp

from app.domains.miners.schemas import ValidatorInfo
from app.logger import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=8.0)
_NOT_FOUND = object()


def _label_from_hotkey(hotkey: str) -> str:
    if len(hotkey) <= 10:
        return hotkey
    return f"{hotkey[:6]}…{hotkey[-4:]}"


def _validator_info(index: int, payload: Optional[dict], online: bool) -> ValidatorInfo:
    """Build the public ValidatorInfo. Never leaks the validator's URL/IP.

    When the validator is reachable we take its identity (hotkey, uid) from
    the response body. When unreachable, we emit an opaque index-based
    placeholder so the frontend can still render a known-but-down slot.
    """

    ident = (payload or {}).get("validator") or {}
    hotkey = ident.get("hotkey") or ""
    return ValidatorInfo(
        id=hotkey or f"validator-{index}",
        uid=ident.get("uid") if ident.get("uid") is not None else 0,
        hotkey=hotkey,
        label=_label_from_hotkey(hotkey) if hotkey else f"Validator {index + 1}",
        online=online,
    )


async def _fetch_one(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status == 404:
                return _NOT_FOUND
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        logger.warning(f"Validator fetch failed url={url} error={e}")
        return None


async def fetch_all_miners(
    validator_urls: list[str],
) -> list[tuple[ValidatorInfo, Optional[dict]]]:
    """Fan out `/public/miners` to every configured validator in parallel."""

    async with aiohttp.ClientSession() as session:
        coros = [
            _fetch_one(session, f"{url.rstrip('/')}/public/miners")
            for url in validator_urls
        ]
        responses = await asyncio.gather(*coros, return_exceptions=True)

    results: list[tuple[ValidatorInfo, Optional[dict]]] = []
    for idx, data in enumerate(responses):
        if isinstance(data, Exception) or data is None:
            results.append((_validator_info(idx, None, online=False), None))
        else:
            results.append((_validator_info(idx, data, online=True), data))
    return results


async def fetch_miner_detail(
    validator_urls: list[str], miner_hotkey: str
) -> list[tuple[ValidatorInfo, Optional[dict]]]:
    """Fan out `/public/miners/{hotkey}` to every configured validator in parallel."""

    async with aiohttp.ClientSession() as session:
        coros = [
            _fetch_one(session, f"{url.rstrip('/')}/public/miners/{miner_hotkey}")
            for url in validator_urls
        ]
        responses = await asyncio.gather(*coros, return_exceptions=True)

    results: list[tuple[ValidatorInfo, Optional[dict]]] = []
    for idx, data in enumerate(responses):
        if isinstance(data, Exception) or data is None:
            results.append((_validator_info(idx, None, online=False), None))
        elif data is _NOT_FOUND:
            # Validator is up but doesn't know this miner
            results.append((_validator_info(idx, None, online=True), None))
        else:
            results.append((_validator_info(idx, data, online=True), data))
    return results
