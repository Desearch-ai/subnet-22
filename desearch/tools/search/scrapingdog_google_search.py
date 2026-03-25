import os
from typing import Optional

import aiohttp
import bittensor as bt


class ScrapingDogGoogleSearch:
    api_url = "https://api.scrapingdog.com/google"

    def __init__(
        self,
        language: str = "en",
        region: str = "us",
        tbs: Optional[str] = None,
        results: int = 10,
        site: Optional[str] = None,
        query_suffix: str = "",
    ) -> None:
        self.language = language or "en"
        self.region = region or "us"
        self.tbs = tbs
        self.results = max(1, min(results or 10, 100))
        self.site = site
        self.query_suffix = query_suffix.strip()
        self.api_key = os.environ.get("SCRAPINGDOG_API_KEY", "")

    def build_query(self, query: str) -> str:
        parts = [query.strip()]

        if self.site:
            parts.append(f"site:{self.site}")

        if self.query_suffix:
            parts.append(self.query_suffix)

        return " ".join(part for part in parts if part).strip()

    async def search(self, query: str, page: int = 0):
        if not self.api_key:
            bt.logging.warning(
                "SCRAPINGDOG_API_KEY is not set. Returning empty search results."
            )
            return []

        params = {
            "api_key": self.api_key,
            "query": self.build_query(query),
            "country": self.region.lower(),
            "advance_search": "false",
            "domain": "google.com",
            "language": self.language.lower(),
            "results": str(self.results),
            "page": str(max(page, 0)),
        }

        if self.tbs:
            params["tbs"] = self.tbs

        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        bt.logging.error(
                            "ScrapingDog search failed with status "
                            f"{response.status}: {error_text}"
                        )
                        return []

                    payload = await response.json()
        except Exception as err:
            bt.logging.error(f"Could not perform ScrapingDog search: {err}")
            return []

        results = []

        organic_results = payload.get("organic_results", [])

        for item in organic_results:
            title = item.get("title")
            link = item.get("link")

            if not title or not link:
                continue

            results.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": item.get("snippet", "") or "",
                }
            )

        return results
