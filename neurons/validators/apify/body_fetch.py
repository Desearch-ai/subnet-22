from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import bittensor as bt

_EXTRACT_LOCK = threading.Lock()

_RAW_CACHE_CHARS = 16000
_CACHE_TTL_S = 600
_MAX_CACHE_ENTRIES = 2000
_MIN_ARTICLE_CHARS = 200

_VERDICT_INJECTION = re.compile(r"(?i)\bverdict\b\s*:")
_JS_GATE = re.compile(
    r"(?i)(please enable javascript|enable javascript and refresh|"
    r"something went wrong\.?\s*(wait a moment|please|try)|"
    r"you (need to )?(enable|turn on) javascript|access denied|are you a robot)"
)


def sanitize_body_text(text: str) -> str:
    return _VERDICT_INJECTION.sub("verdict-", text or "")


def is_usable_article(text: str) -> bool:
    return bool(text) and len(text) >= _MIN_ARTICLE_CHARS and not _JS_GATE.search(text)


def extract_article_text(html: str, url: str, max_chars: int = _RAW_CACHE_CHARS) -> str:
    if not html:
        return ""

    text = ""
    try:
        import trafilatura

        text = (
            trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
                url=url,
            )
            or ""
        ).strip()
    except Exception as e:
        bt.logging.trace(f"trafilatura failed for {url}: {e}")

    return sanitize_body_text(text)[:max_chars]


def _extract_meta(html: str, url: str) -> Tuple[str, str, str]:
    try:
        import trafilatura

        md = trafilatura.extract_metadata(html, default_url=url)
        if md:
            return (md.title or "", md.date or "", md.author or "")
    except Exception:
        pass
    return "", "", ""


def _extract_pair(html: str, url: str, max_chars: int) -> Tuple[str, str, str, str]:
    with _EXTRACT_LOCK:
        title, published_date, author = _extract_meta(html, url)
        return (
            title,
            extract_article_text(html, url, max_chars),
            published_date,
            author,
        )


async def extract_article_async(
    html: str, url: str, max_chars: int = _RAW_CACHE_CHARS
) -> Tuple[str, str, str, str]:
    return await asyncio.to_thread(_extract_pair, html, url, max_chars)


class BodyFetcher:
    def __init__(self) -> None:
        self._cache: Dict[str, tuple] = {}

    def _cached(self, url: str) -> Optional[dict]:
        entry = self._cache.get(url)
        if entry and (time.monotonic() - entry[0]) < _CACHE_TTL_S:
            return entry[1]
        return None

    def _store(self, url: str, record: dict) -> None:
        if len(self._cache) >= _MAX_CACHE_ENTRIES:
            now = time.monotonic()
            self._cache = {
                u: e for u, e in self._cache.items() if (now - e[0]) < _CACHE_TTL_S
            }
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                oldest = sorted(self._cache, key=lambda u: self._cache[u][0])
                for u in oldest[: len(self._cache) // 2]:
                    del self._cache[u]
        self._cache[url] = (time.monotonic(), record)

    @staticmethod
    def _truncate(record: dict, max_chars: int) -> dict:
        out = dict(record)
        out["text"] = (record.get("text") or "")[:max_chars]
        return out

    async def get_many(
        self, urls: List[str], max_chars: int = _RAW_CACHE_CHARS
    ) -> Dict[str, dict]:
        urls = [u for u in dict.fromkeys(urls) if u]
        result: Dict[str, dict] = {}
        to_fetch: List[str] = []

        for url in urls:
            cached = self._cached(url)
            if cached is not None:
                result[url] = self._truncate(cached, max_chars)
            else:
                to_fetch.append(url)

        if not to_fetch:
            return result

        for url, record in (await self._fetch_and_extract(to_fetch)).items():
            self._store(url, record)
            result[url] = self._truncate(record, max_chars)

        return result

    async def _fetch_and_extract(self, urls: List[str]) -> Dict[str, dict]:
        from neurons.validators.apify.scrapingdog_scraper import (
            scrape_links_with_retries,
        )

        try:
            fetched, _ = await scrape_links_with_retries(urls=urls, max_attempts=2)
        except Exception as e:
            bt.logging.warning(f"body fetch failed: {e}")
            fetched = []

        by_url = {item.get("link"): item for item in fetched if item.get("link")}

        async def extract_one(url: str) -> Tuple[str, dict]:
            item = by_url.get(url) or {}
            html = item.get("html_content") or ""
            title, text, published_date, author = await extract_article_async(html, url)
            if not is_usable_article(text):
                text = sanitize_body_text(item.get("html_text") or "")[
                    :_RAW_CACHE_CHARS
                ]
            if not is_usable_article(text):
                text = ""
            return url, {
                "url": url,
                "title": title or item.get("title", "") or "",
                "text": text,
                "published_date": published_date or item.get("date", "") or "",
                "author": author or "",
                "error": "" if text else "no article",
            }

        return dict(await asyncio.gather(*[extract_one(u) for u in urls]))


_fetcher: Optional[BodyFetcher] = None


def get_body_fetcher() -> BodyFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = BodyFetcher()
    return _fetcher
