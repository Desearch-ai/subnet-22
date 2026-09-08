"""Visit each candidate once and decide whether miners can crawl it.

A host qualifies when robots.txt allows our token, it publishes a sitemap we can parse, and its
homepage serves readable English text.
"""

from __future__ import annotations

import asyncio
import zlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from .sources import USER_AGENT

ROBOTS_TOKEN = "DesearchBot"
# One request per second to a host unless its robots.txt asks for longer. Applies to every
# request we make to that host: robots.txt, each sitemap in the tree, and the homepage.
MIN_HOST_INTERVAL = 1.0
SITEMAP_GUESSES = ("/sitemap.xml", "/sitemap_index.xml")
MIN_SITEMAP_URLS = 10
MIN_HOMEPAGE_CHARS = 200
MAX_ROBOTS_BYTES = 512 * 1024
MAX_SITEMAP_BYTES = 8 * 1024 * 1024
MAX_HOMEPAGE_BYTES = 512 * 1024

SITEMAP_DIRECTIVE = re.compile(rb"(?im)^\s*sitemap\s*:\s*(\S+)")
LOC = re.compile(rb"<loc>\s*([^<\s]+)\s*</loc>", re.I)
SITEMAP_INDEX = re.compile(rb"<sitemapindex", re.I)
HTML_LANG = re.compile(r"<html[^>]*\blang=[\"']?([a-zA-Z-]{2,8})", re.I)
SCRIPT_OR_STYLE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
META_CHARSET = re.compile(rb'charset=["\']?([\w-]+)', re.I)
BOT_WALL = re.compile(
    r"(please enable javascript|enable javascript and refresh|you need to enable javascript"
    r"|access denied|are you a robot|checking your browser|attention required)",
    re.I,
)


@dataclass
class Result:
    host: str
    rank: int | None = None
    tld_group: str = ""
    type_hint: str | None = None
    robots_status: int | None = None
    robots_allows: bool | None = None
    crawl_delay: float | None = None
    sitemap_url: str | None = None
    sitemap_source: str | None = None
    sitemap_kind: str | None = None
    sitemap_urls: int = 0
    home_status: int | None = None
    home_chars: int = 0
    declared_lang: str | None = None
    language: str | None = None
    qualified: bool = False
    reject_reason: str | None = None
    error: str | None = None
    checked_at: str = ""
    sample_urls: list[str] = field(default_factory=list)


def _decode(body: bytes) -> str:
    match = META_CHARSET.search(body[:4096])
    encoding = match.group(1).decode("ascii", "ignore") if match else "utf-8"
    try:
        return body.decode(encoding, "replace")
    except LookupError:
        return body.decode("utf-8", "replace")


def _visible_text(html: str) -> str:
    return " ".join(TAG.sub(" ", SCRIPT_OR_STYLE.sub(" ", html)).split())


def _gunzip(body: bytes) -> bytes:
    """Decompress gzip, keeping whatever decoded when the body was cut short by a size cap."""
    if body[:2] != b"\x1f\x8b":
        return body
    decompressor = zlib.decompressobj(wbits=31)
    try:
        return decompressor.decompress(body)
    except zlib.error:
        return body


def robots_allows(text: str, token: str) -> tuple[bool, float | None]:
    """Longest-match rule for our token, falling back to the wildcard group."""
    groups: dict[str, list[tuple[str, str]]] = {}
    delays: dict[str, float] = {}
    current: list[str] = []
    previous_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, _, value = (part.strip() for part in line.partition(":"))
        field_name = field_name.lower()
        if field_name == "user-agent":
            if not previous_was_agent:
                current = []
            current.append(value.lower())
            groups.setdefault(value.lower(), [])
            previous_was_agent = True
            continue
        previous_was_agent = False
        if field_name in ("allow", "disallow"):
            for agent in current:
                groups.setdefault(agent, []).append((field_name, value))
        elif field_name == "crawl-delay":
            for agent in current:
                try:
                    delays[agent] = float(value)
                except ValueError:
                    pass

    agent = token.lower() if token.lower() in groups else "*"
    best_rule, best_length = None, -1
    for rule, path in groups.get(agent, []):
        if not path:
            continue
        prefix = path.rstrip("*")
        if "/".startswith(prefix) or prefix == "/":
            if len(prefix) > best_length or (
                len(prefix) == best_length and rule == "allow"
            ):
                best_rule, best_length = rule, len(prefix)
    allowed = best_rule != "disallow"
    return allowed, delays.get(agent)


def parse_sitemap(body: bytes) -> tuple[str, int, list[str]]:
    body = _gunzip(body)
    locs = LOC.findall(body)
    if not locs:
        return "invalid", 0, []
    kind = "index" if SITEMAP_INDEX.search(body[:4096]) else "urlset"
    urls = [loc.decode("utf-8", "replace") for loc in locs[:5]]
    return kind, len(locs), urls


def _resolver():
    try:
        from aiohttp.resolver import AsyncResolver

        return AsyncResolver()
    except ImportError:
        return None


class Qualifier:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        detect_language,
        timeout: float = 8.0,
        adult: set[str] = frozenset(),
    ):
        self.session = session
        self.detect_language = detect_language
        self.adult = adult
        self.crawl_delay = 0.0
        self._next_request_at = 0.0
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 6.0),
                                              sock_connect=min(timeout, 6.0))

    async def _pace(self) -> None:
        """Wait out the remainder of this host's interval before the next request."""
        wait = self._next_request_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_request_at = time.monotonic() + max(self.crawl_delay, MIN_HOST_INTERVAL)

    async def _get(self, url: str, limit: int) -> tuple[int | None, bytes]:
        await self._pace()
        async with self.session.get(
            url,
            timeout=self.timeout,
            allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            return response.status, await response.content.read(limit)

    async def _try_schemes(self, host: str, path: str, limit: int):
        last = "unknown"
        for scheme in ("https", "http"):
            try:
                return await self._get(f"{scheme}://{host}{path}", limit)
            except Exception as exc:
                last = type(exc).__name__
                if "DNS" in last:
                    break
        raise RuntimeError(last)

    async def run(self, host: str, rank=None, tld_group="", type_hint=None) -> Result:
        result = Result(
            host=host,
            rank=rank,
            tld_group=tld_group,
            type_hint=type_hint,
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if host in self.adult:
            result.reject_reason = "adult_list"
            return result
        self.crawl_delay = 0.0
        self._next_request_at = 0.0
        try:
            result.robots_status, body = await self._try_schemes(
                host, "/robots.txt", MAX_ROBOTS_BYTES
            )
        except Exception as exc:
            result.error, result.reject_reason = str(exc)[:60], "unreachable"
            return result

        candidates: list[tuple[str, str]] = []
        if result.robots_status == 200:
            text = _decode(body)
            result.robots_allows, result.crawl_delay = robots_allows(text, ROBOTS_TOKEN)
            self.crawl_delay = result.crawl_delay or 0.0
            if result.robots_allows is False:
                result.reject_reason = "robots_disallowed"
                return result
            candidates = [
                (m.decode("utf-8", "replace"), "robots")
                for m in SITEMAP_DIRECTIVE.findall(body)
            ][:3]
        elif result.robots_status is not None and 500 <= result.robots_status:
            result.reject_reason = "robots_unavailable"
            return result

        candidates += [(f"https://{host}{path}", "guess") for path in SITEMAP_GUESSES]

        for url, origin in candidates:
            try:
                status, payload = await self._get(url, MAX_SITEMAP_BYTES)
            except Exception:
                continue
            if status != 200 or not payload:
                continue
            kind, count, sample = parse_sitemap(payload)
            if kind == "invalid":
                continue
            result.sitemap_url, result.sitemap_source = url, origin
            result.sitemap_kind, result.sitemap_urls, result.sample_urls = (
                kind,
                count,
                sample,
            )
            break

        if not result.sitemap_url:
            result.reject_reason = "no_sitemap"
            return result
        if result.sitemap_kind == "urlset" and result.sitemap_urls < MIN_SITEMAP_URLS:
            result.reject_reason = "sitemap_too_small"
            return result

        try:
            result.home_status, body = await self._try_schemes(
                host, "/", MAX_HOMEPAGE_BYTES
            )
        except Exception as exc:
            result.error, result.reject_reason = str(exc)[:60], "homepage_unreachable"
            return result
        if result.home_status != 200:
            result.reject_reason = "homepage_error"
            return result

        html = _decode(body)
        match = HTML_LANG.search(html[:4000])
        result.declared_lang = match.group(1).lower().split("-")[0] if match else None
        text = _visible_text(html)
        result.home_chars = len(text)
        if BOT_WALL.search(text[:2000]):
            result.reject_reason = "bot_wall"
            return result
        if len(text) < MIN_HOMEPAGE_CHARS:
            result.reject_reason = "no_homepage_text"
            return result
        result.language = self.detect_language(text[:2000])
        if result.language != "en":
            result.reject_reason = "not_english"
            return result

        result.qualified = True
        return result


async def qualify_hosts(
    hosts, detect_language, concurrency: int = 256, timeout: float = 8.0, on_result=None
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=4,
        ttl_dns_cache=900,
        enable_cleanup_closed=True,
        resolver=_resolver(),
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        qualifier = Qualifier(session, detect_language, timeout)

        async def worker():
            while True:
                entry = await queue.get()
                if entry is None:
                    queue.task_done()
                    return
                try:
                    result = await qualifier.run(*entry)
                except Exception as exc:
                    result = Result(
                        host=entry[0], reject_reason="crashed", error=str(exc)[:60]
                    )
                if on_result:
                    on_result(result)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        for entry in hosts:
            await queue.put(entry)
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)


def write_jsonl(path: Path, results) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
