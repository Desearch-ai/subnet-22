import asyncio
import json
import os
import re
import weakref
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import aiohttp
import bittensor as bt

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be"}
_REDDIT_HOSTS = {"reddit.com", "www.reddit.com"}


def _classify_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in _YOUTUBE_HOSTS:
        return "youtube"
    if host in _REDDIT_HOSTS:
        return "reddit"
    return "default"


def rewrite_to_old_reddit(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in _REDDIT_HOSTS:
        return url.replace(f"://{parsed.hostname}", "://old.reddit.com", 1)
    return url


def _extract_youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            values = parse_qs(parsed.query).get("v")
            return values[0] if values else None

        for prefix in ("/shorts/", "/embed/", "/v/"):
            if parsed.path.startswith(prefix):
                video_id = parsed.path[len(prefix) :].split("/", 1)[0]
                return video_id or None

    return None


def get_scrapingdog_api_key() -> str:
    return os.environ.get("SCRAPINGDOG_API_KEY", "")


def has_scrapingdog_api_key() -> bool:
    return bool(get_scrapingdog_api_key())


class _ScrapingDogHTMLParser(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_chunks: List[str] = []
        self.text_chunks: List[str] = []
        self.description: str = ""
        self.og_description: str = ""
        self.paragraphs: List[str] = []
        self.hacker_news_comments: List[str] = []

        self._ignored_depth = 0
        self._title_depth = 0
        self._paragraph_depth = 0
        self._hacker_news_comment_depth = 0
        self._current_paragraph_chunks: List[str] = []
        self._current_hacker_news_comment_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attrs_dict = {
            key.lower(): (value or "") for key, value in attrs if key is not None
        }

        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        if self._hacker_news_comment_depth:
            self._hacker_news_comment_depth += 1

        if tag == "title":
            self._title_depth += 1
            return

        if tag == "meta":
            content = self._normalize_space(attrs_dict.get("content", ""))

            if not content:
                return

            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()

            if name == "description" and not self.description:
                self.description = content
            elif prop == "og:description" and not self.og_description:
                self.og_description = content

            return

        if tag == "p":
            self._paragraph_depth += 1
            if self._paragraph_depth == 1:
                self._current_paragraph_chunks = []
            return

        classes = set(attrs_dict.get("class", "").split())
        if (
            tag == "div"
            and "commtext" in classes
            and not self._current_hacker_news_comment_chunks
            and self._hacker_news_comment_depth == 0
        ):
            self._hacker_news_comment_depth = 1
            self._current_hacker_news_comment_chunks = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in self._IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return

        if self._ignored_depth:
            return

        finalize_hacker_news_comment = False
        if self._hacker_news_comment_depth:
            self._hacker_news_comment_depth -= 1
            if self._hacker_news_comment_depth == 0:
                finalize_hacker_news_comment = True

        if tag == "title":
            if self._title_depth:
                self._title_depth -= 1
        elif tag == "p" and self._paragraph_depth:
            self._paragraph_depth -= 1
            if self._paragraph_depth == 0:
                self._finalize_capture(self._current_paragraph_chunks, self.paragraphs)
                self._current_paragraph_chunks = []

        if finalize_hacker_news_comment:
            self._finalize_capture(
                self._current_hacker_news_comment_chunks,
                self.hacker_news_comments,
            )
            self._current_hacker_news_comment_chunks = []

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return

        text = self._normalize_space(data)
        if not text:
            return

        self.text_chunks.append(text)

        if self._title_depth:
            self.title_chunks.append(text)

        if self._paragraph_depth:
            self._current_paragraph_chunks.append(text)

        if self._hacker_news_comment_depth:
            self._current_hacker_news_comment_chunks.append(text)

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _finalize_capture(chunks: List[str], output: List[str]) -> None:
        text = " ".join(chunks).strip()
        if text:
            output.append(text)


class ScrapingDogScraper:
    api_url = "https://api.scrapingdog.com/scrape"
    youtube_api_url = "https://api.scrapingdog.com/youtube/video"
    request_timeout_seconds = 30
    max_concurrent_requests = 30
    _shared_semaphores = weakref.WeakKeyDictionary()

    @classmethod
    def _get_shared_semaphore(cls) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = cls._shared_semaphores.get(loop)

        if semaphore is None:
            semaphore = asyncio.Semaphore(cls.max_concurrent_requests)
            cls._shared_semaphores[loop] = semaphore

        return semaphore

    async def scrape_metadata(
        self, urls: List[str], attempt: int = 1
    ) -> List[Dict[str, Optional[str]]]:
        if not urls:
            return []

        semaphore = self._get_shared_semaphore()
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            tasks = [
                asyncio.create_task(self._scrape_url(session, semaphore, url, attempt))
                for url in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_results: List[Dict[str, Optional[str]]] = []

        failed_urls = []

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failed_urls.append(f"{url} -> {type(result).__name__}: {result}")
                continue

            if result:
                scraped_results.append(result)

        if failed_urls:
            bt.logging.warning(
                f"ScrapingDog failed to fetch links (attempt {attempt}): {failed_urls}"
            )

        return scraped_results

    async def scrape_metadata_with_retries(
        self, urls: List[str], max_attempts: int = 3
    ) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
        fetched_links_with_metadata: List[Dict[str, Optional[str]]] = []
        non_fetched_links = list(dict.fromkeys(urls))

        if not has_scrapingdog_api_key():
            bt.logging.warning(
                "SCRAPINGDOG_API_KEY is not set. Returning empty scraped links. "
                f"0 fetched links for {len(non_fetched_links)} urls. "
                "See here: https://github.com/Desearch-ai/subnet-22/blob/main/docs/env_variables.md."
            )
            return fetched_links_with_metadata, non_fetched_links

        attempt = 1

        while attempt <= max_attempts and non_fetched_links:
            bt.logging.info(
                "ScrapingDog attempt "
                f"{attempt}/{max_attempts}, processing "
                f"{len(non_fetched_links)} links with concurrency limit "
                f"{self.max_concurrent_requests}."
            )

            fetched_links_with_metadata.extend(
                await self.scrape_metadata(non_fetched_links, attempt=attempt)
            )

            fetched_urls = {
                link.get("link")
                for link in fetched_links_with_metadata
                if link.get("link")
            }
            non_fetched_links = [
                url for url in non_fetched_links if url not in fetched_urls
            ]
            attempt += 1

        return fetched_links_with_metadata, non_fetched_links

    async def _scrape_url(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        url: str,
        attempt: int,
    ) -> Dict[str, Optional[str]]:
        kind = _classify_url(url)

        if kind == "youtube" and _extract_youtube_video_id(url):
            return await self._scrape_youtube(session, semaphore, url)

        # YouTube non-video pages (playlists, channel pages, /creators) fall through
        # to the default /scrape ladder.
        scrape_kind = "default" if kind == "youtube" else kind

        api_url, params = self._build_request_params(url, attempt, scrape_kind)

        async with semaphore:
            async with session.get(api_url, params=params) as response:
                response_text = await response.text(errors="replace")

                if response.status != 200:
                    raise RuntimeError(
                        f"Unexpected ScrapingDog status {response.status} "
                        f"(attempt {attempt}, kind={kind}): {response_text[:200]}"
                    )

        return self._build_metadata(url=url, html_content=response_text)

    async def _scrape_youtube(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        url: str,
    ) -> Dict[str, Optional[str]]:
        video_id = _extract_youtube_video_id(url)
        if not video_id:
            raise RuntimeError(f"Could not extract YouTube video id from {url}")

        params = {
            "api_key": get_scrapingdog_api_key(),
            "v": video_id,
        }

        async with semaphore:
            async with session.get(self.youtube_api_url, params=params) as response:
                response_text = await response.text(errors="replace")

                if response.status != 200:
                    raise RuntimeError(
                        f"Unexpected ScrapingDog YouTube status {response.status}: "
                        f"{response_text[:200]}"
                    )

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"ScrapingDog YouTube returned non-JSON for {url}: {err}"
            )

        return self._build_youtube_metadata(url=url, payload=payload)

    def _build_request_params(
        self, url: str, attempt: int, kind: str
    ) -> Tuple[str, Dict[str, str]]:
        request_url = url

        if kind == "reddit":
            request_url = rewrite_to_old_reddit(url)

        params: Dict[str, str] = {
            "api_key": get_scrapingdog_api_key(),
            "url": request_url,
        }

        if kind == "reddit":
            # old.reddit is server-rendered, so attempt 1 is plain (cheapest).
            # Escalate to premium proxies, then add JS render with a wait.
            if attempt >= 2:
                params["premium"] = "true"
            if attempt >= 3:
                params["dynamic"] = "true"
                params["wait"] = "3000"
        else:
            params["dynamic"] = "true" if attempt >= 2 else "false"
            if attempt >= 3:
                params["premium"] = "true"

        return self.api_url, params

    def _build_youtube_metadata(
        self, url: str, payload: Dict
    ) -> Dict[str, Optional[str]]:
        video = payload.get("video") if isinstance(payload, dict) else {}
        channel = payload.get("channel") if isinstance(payload, dict) else {}
        video = video if isinstance(video, dict) else {}
        channel = channel if isinstance(channel, dict) else {}

        title = (video.get("title") or "").strip()
        description = (video.get("description") or "").strip()

        text_parts = [title, description]

        for key in ("keywords", "tags"):
            value = video.get(key)
            if isinstance(value, list):
                text_parts.append(" ".join(str(item) for item in value if item))
            elif isinstance(value, str):
                text_parts.append(value)

        for key in ("author", "published_time"):
            value = video.get(key)
            if isinstance(value, str) and value:
                text_parts.append(value)

        channel_name = channel.get("name")
        if isinstance(channel_name, str) and channel_name:
            text_parts.append(channel_name)

        html_text = " ".join(part for part in text_parts if part).strip()

        return {
            "title": title,
            "snippet": description,
            "link": url,
            "html_content": html_text,
            "html_text": html_text,
        }

    def _build_metadata(self, url: str, html_content: str) -> Dict[str, Optional[str]]:
        parser = _ScrapingDogHTMLParser()
        parser.feed(html_content)
        parser.close()

        return {
            "title": " ".join(parser.title_chunks).strip(),
            "snippet": self._extract_description(url=url, parser=parser),
            "link": url,
            "html_content": html_content,
            "html_text": " ".join(parser.text_chunks).strip(),
        }

    def _extract_description(self, url: str, parser: _ScrapingDogHTMLParser) -> str:
        hostname = (urlparse(url).hostname or "").lower()

        if "wikipedia.org" in hostname:
            paragraph = self._first_non_empty(parser.paragraphs)
            if paragraph:
                return self._clean_wikipedia_description(paragraph)

        if "news.ycombinator.com" in hostname:
            comment = self._first_non_empty(parser.hacker_news_comments)
            if comment:
                return comment

        if parser.description:
            return parser.description

        if parser.og_description:
            return parser.og_description

        return self._first_non_empty(parser.paragraphs) or ""

    @staticmethod
    def _first_non_empty(values: List[str]) -> Optional[str]:
        return next((value for value in values if value.strip()), None)

    @staticmethod
    def _clean_wikipedia_description(text: str) -> str:
        cleaned_text = re.sub(r"\[\d+\]", "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text.rstrip(" .")


async def scrape_links_with_retries(
    urls: List[str], max_attempts: int = 3
) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
    if has_scrapingdog_api_key():
        bt.logging.info("Using ScrapingDog for validator web scraping.")
        return await ScrapingDogScraper().scrape_metadata_with_retries(
            urls=urls,
            max_attempts=max_attempts,
        )

    bt.logging.info(
        "SCRAPINGDOG_API_KEY is not set. Falling back to Apify Cheerio "
        "scraper for validator web scraping."
    )

    try:
        from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor
        from neurons.validators.apify.utils import (
            scrape_links_with_retries as scrape_links_with_apify_retries,
        )
    except Exception as exc:
        bt.logging.warning(
            f"Apify fallback is unavailable for validator web scraping: {exc}"
        )
        return [], list(dict.fromkeys(urls))

    request_to_original = {rewrite_to_old_reddit(url): url for url in urls}
    request_urls = list(request_to_original.keys())

    fetched, missing = await scrape_links_with_apify_retries(
        urls=request_urls,
        scraper_actor_class=CheerioScraperActor,
        group_size=100,
        max_attempts=max_attempts,
    )

    for item in fetched:
        link = item.get("link")
        if link in request_to_original:
            item["link"] = request_to_original[link]

    missing = [request_to_original.get(url, url) for url in missing]

    return fetched, missing
