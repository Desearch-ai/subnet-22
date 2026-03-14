import logging
import time

import aiohttp
import bittensor as bt

logger = logging.getLogger(__name__)


class UtilityAPIClient:
    """
    Client for the Utility API that provides scoring questions.

    Authenticates each request by signing the current timestamp with the
    validator's hotkey.
    """

    def __init__(self, base_url: str, wallet: bt.Wallet):
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self._session = aiohttp.ClientSession()

    def _auth_headers(self) -> dict[str, str]:
        timestamp = str(int(time.time()))
        signature = self.wallet.hotkey.sign(timestamp.encode()).hex()

        return {
            "X-Hotkey": self.wallet.hotkey.ss58_address,
            "X-Timestamp": timestamp,
            "X-Signature": signature,
        }

    async def fetch_next_question(self) -> dict:
        """
        Fetch one (question, search_type, uid) from the utility API.

        Returns:
            {
                "epoch_id": int,
                "uid": int,
                "search_type": str,      # e.g. "ai_search", "x_search"
                "question": {"query": str}
            }

        Raises:
            aiohttp.ClientResponseError: on 4xx/5xx responses (including 404 when
                                         all questions for this epoch are served,
                                         or 429 for rate limiting)
        """

        async with self._session.get(
            f"{self.base_url}/dataset/next",
            headers=self._auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def save_logs(self, logs: list[dict]) -> dict:
        async with self._session.post(
            f"{self.base_url}/logs",
            headers=self._auth_headers(),
            json={"logs": logs},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def close(self):
        await self._session.close()
