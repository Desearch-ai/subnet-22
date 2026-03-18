import time

import aiohttp
import bittensor as bt


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

    async def _raise_for_status_with_context(
        self,
        response: aiohttp.ClientResponse,
        *,
        context: str,
        skip_logging_statuses: set[int] | None = None,
    ) -> None:
        if response.status < 400:
            return

        if response.status not in (skip_logging_statuses or set()):
            body = (await response.text()).strip()
            body_preview = body[:1000]
            message = (
                f"[UtilityAPIClient] {context} failed status={response.status} "
                f"url={response.url}"
            )
            if body_preview:
                message = f"{message} body={body_preview}"
            bt.logging.error(message)

        response.raise_for_status()

    async def fetch_next_question(self) -> dict:
        """
        Fetch one (question, search_type, uid) from the utility API.

        Returns:
            {
                "time_range_start": str,
                "uid": int,
                "search_type": str,      # e.g. "ai_search", "x_search"
                "question": {"query": str}
            }

        Raises:
            aiohttp.ClientResponseError: on 4xx/5xx responses (including 404 when
                                         all questions for this scoring window are served,
                                         or 429 for rate limiting)
        """

        async with self._session.get(
            f"{self.base_url}/dataset/next",
            headers=self._auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            await self._raise_for_status_with_context(
                response,
                context="fetch_next_question",
                skip_logging_statuses={404, 429},
            )
            return await response.json()

    async def save_logs(self, logs: list[dict]) -> dict:
        async with self._session.post(
            f"{self.base_url}/logs",
            headers=self._auth_headers(),
            json={"logs": logs},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            await self._raise_for_status_with_context(
                response,
                context="save_logs",
            )
            return await response.json()

    async def close(self):
        await self._session.close()
