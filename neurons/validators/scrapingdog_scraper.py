import asyncio
import os
import re
import weakref
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import bittensor as bt

SCRAPINGDOG_API_KEY = os.environ.get("SCRAPINGDOG_API_KEY")


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
    request_timeout_seconds = 30
    max_concurrent_requests = 50
    _shared_semaphores = weakref.WeakKeyDictionary()

    @classmethod
    def _get_shared_semaphore(cls) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = cls._shared_semaphores.get(loop)

        if semaphore is None:
            semaphore = asyncio.Semaphore(cls.max_concurrent_requests)
            cls._shared_semaphores[loop] = semaphore

        return semaphore

    async def scrape_metadata(self, urls: List[str]) -> List[Dict[str, Optional[str]]]:
        if not SCRAPINGDOG_API_KEY:
            raise ValueError(
                "Please set the SCRAPINGDOG_API_KEY environment variable. "
                "See here: https://github.com/Desearch-ai/subnet-22/blob/main/docs/env_variables.md."
            )

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
                asyncio.create_task(self._scrape_url(session, semaphore, url))
                for url in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_results: List[Dict[str, Optional[str]]] = []

        failed_urls = []

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failed_urls.append(url)
                continue

            if result:
                scraped_results.append(result)

        if failed_urls:
            bt.logging.warning(f"ScrapingDog failed to fetch links: {failed_urls}")

        return scraped_results

    async def scrape_metadata_with_retries(
        self, urls: List[str], max_attempts: int = 2
    ) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
        fetched_links_with_metadata: List[Dict[str, Optional[str]]] = []
        non_fetched_links = list(dict.fromkeys(urls))
        attempt = 1

        while attempt <= max_attempts and non_fetched_links:
            bt.logging.info(
                "ScrapingDog attempt "
                f"{attempt}/{max_attempts}, processing "
                f"{len(non_fetched_links)} links with concurrency limit "
                f"{self.max_concurrent_requests}."
            )

            fetched_links_with_metadata.extend(
                await self.scrape_metadata(non_fetched_links)
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
    ) -> Dict[str, Optional[str]]:
        async with semaphore:
            async with session.get(
                self.api_url,
                params=self._build_request_params(url),
            ) as response:
                response_text = await response.text(errors="replace")

                if response.status != 200:
                    raise RuntimeError(
                        f"Unexpected ScrapingDog status {response.status}: "
                        f"{response_text[:200]}"
                    )

        return self._build_metadata(url=url, html_content=response_text)

    def _build_request_params(self, url: str) -> Dict[str, str]:
        return {
            "api_key": SCRAPINGDOG_API_KEY or "",
            "url": url,
            "dynamic": "false",
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
    urls: List[str], max_attempts: int = 2
) -> Tuple[List[Dict[str, Optional[str]]], List[str]]:
    return await ScrapingDogScraper().scrape_metadata_with_retries(
        urls=urls,
        max_attempts=max_attempts,
    )
