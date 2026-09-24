from __future__ import annotations

import asyncio
import codecs
import ipaddress
import itertools
import re
import socket
import ssl
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import aiohttp
import charset_normalizer
from aiohttp.abc import AbstractResolver

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)
PROXY_SCHEMES = ("http", "https")
HTML_TYPES = ("text/html", "application/xhtml+xml")
SNIFFED_TYPES = ("", "application/octet-stream", "binary/octet-stream")
RETRYABLE = frozenset({"connect", "timeout", "http_5xx", "blocked"})
# Refusals are per address, so retry through another route.
RETRY_STATUSES = frozenset({403, 408, 429})
FALLBACK_STATUSES = frozenset({401, 403, 407, 408, 429})
FALLBACK_ERRORS = frozenset({"timeout", "dns", "connect", "tls", "blocked", "other"})
SCRAPINGDOG_URL = "https://api.scrapingdog.com/scrape"
SCRAPINGDOG_ATTEMPTS = 2
# ScrapingDog returns 400 when its own fetch failed; a retry usually works.
SCRAPINGDOG_RETRY_STATUSES = frozenset({400, 429})
GZIP_WBITS = 16 + zlib.MAX_WBITS
DECODINGS = {"gzip": GZIP_WBITS, "x-gzip": GZIP_WBITS, "deflate": zlib.MAX_WBITS}
UNDECODED = frozenset(
    {
        "br",
        "zstd",
        "compress",
        "x-compress",
        "aes128gcm",
        "dcb",
        "dcz",
        "exi",
        "pack200-gzip",
    }
)
LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
NAT64 = ipaddress.ip_network("64:ff9b::/96")
MAX_REDIRECTS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
READ_CHUNK = 65536
META_CHARSET = re.compile(rb"""<meta[^>]*?charset\s*=\s*["']?\s*([\w.:-]+)""", re.I)
GUESS_BYTES = 65536
UNSAFE_CODECS = frozenset(
    {"undefined", "idna", "punycode", "unicode-escape", "raw-unicode-escape"}
)

DNS_ERRORS = (aiohttp.ClientConnectorDNSError, socket.gaierror)
TLS_ERRORS = (ssl.SSLError, aiohttp.ClientSSLError, aiohttp.ServerFingerprintMismatch)
CONNECTION_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    aiohttp.ClientResponseError,
    OSError,
)

Resolver = Callable[[str, int], Awaitable[list[str]]]


class ForbiddenAddress(Exception):
    pass


@dataclass(frozen=True)
class FetchSettings:
    concurrency: int = 32
    per_domain: int = 8
    proxy_urls: tuple[str, ...] = ()
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 30.0
    max_bytes: int = 5_000_000
    allow_private: bool = False

    def __post_init__(self):
        for index, proxy in enumerate(self.proxy_urls):
            _check_proxy(index, proxy)


@dataclass
class Fetched:
    url: str
    final_url: str
    fetched_at: datetime
    status: int = 0
    error: str | None = None
    content_type: str = ""
    charset: str | None = None
    body: bytes | None = None
    elapsed_ms: int = 0
    attempts: int = 1


@dataclass
class Route:
    proxy: str | None
    session: aiohttp.ClientSession | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        if self.proxy is None:
            return "direct"
        return urlsplit(self.proxy).netloc.rpartition("@")[2]


@dataclass
class DomainSlot:
    semaphore: asyncio.Semaphore
    users: int = 0


class Fetcher:
    def __init__(self, settings: FetchSettings, resolve: Resolver | None = None):
        self.timeout = settings.timeout
        self.max_bytes = settings.max_bytes
        self.per_domain = settings.per_domain
        self.concurrency = settings.concurrency
        self.slots = asyncio.Semaphore(settings.concurrency)
        self.domain_slots: dict[str, DomainSlot] = {}
        self.headers = browser_headers(settings.user_agent)
        self.guarded = not settings.allow_private
        self.resolve = resolve or resolve_host
        self.routes = [Route(proxy) for proxy in settings.proxy_urls or (None,)]
        self.rotation = itertools.cycle(self.routes)

    def next_route(self) -> Route:
        return next(self.rotation)

    async def fetch(self, url: str, deadline: float | None = None) -> Fetched:
        async with self._domain_slot(_domain(url)), self.slots:
            fetched = await self._attempt(self.next_route(), url, deadline)
            if retryable(fetched):
                fetched = await self._attempt(self.next_route(), url, deadline)
                fetched.attempts = 2
        return fetched

    async def aclose(self) -> None:
        for route in self.routes:
            if route.session is not None:
                await route.session.close()

    @asynccontextmanager
    async def _domain_slot(self, domain: str):
        slot = self.domain_slots.get(domain)
        if slot is None:
            slot = self.domain_slots[domain] = DomainSlot(
                asyncio.Semaphore(self.per_domain)
            )
        slot.users += 1
        try:
            async with slot.semaphore:
                yield
        finally:
            slot.users -= 1
            if not slot.users:
                del self.domain_slots[domain]

    def _session(self, route: Route) -> aiohttp.ClientSession:
        if route.session is None:
            # Direct fetches resolve through the guard; a proxy resolves its own targets.
            resolver = (
                GuardedResolver(self.resolve)
                if self.guarded and route.proxy is None
                else None
            )
            route.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=self.concurrency, resolver=resolver
                ),
                headers=self.headers,
                cookie_jar=aiohttp.DummyCookieJar(),
                auto_decompress=False,
                trust_env=False,
                timeout=aiohttp.ClientTimeout(
                    sock_connect=self.timeout, sock_read=self.timeout
                ),
            )
        return route.session

    async def _attempt(self, route: Route, url: str, deadline: float | None) -> Fetched:
        fetched = Fetched(url=url, final_url=url, fetched_at=datetime.now(timezone.utc))
        allowed = self.timeout
        if deadline is not None:
            allowed = min(allowed, deadline - time.time())
        if allowed <= 0:
            fetched.error = "timeout"
            return fetched

        started = time.monotonic()
        try:
            await asyncio.wait_for(self._follow_redirects(route, fetched), allowed)
        except Exception as exc:
            fetched.error = classify_error(exc)
        fetched.elapsed_ms = int((time.monotonic() - started) * 1000)
        return fetched

    async def _follow_redirects(self, route: Route, fetched: Fetched) -> None:
        session, url = self._session(route), fetched.url
        # Kept for this redirect chain only, per host, like a fresh browser tab.
        cookies: dict[str, dict[str, str]] = defaultdict(dict)
        for _ in range(MAX_REDIRECTS + 1):
            host = urlsplit(url).hostname or ""
            if self.guarded:
                refuse_local(host)
            async with session.get(
                url,
                proxy=route.proxy,
                allow_redirects=False,
                cookies=cookies.get(host) or None,
            ) as response:
                cookies[host].update(
                    {name: morsel.value for name, morsel in response.cookies.items()}
                )
                location = response.headers.get("location")
                if response.status in REDIRECT_STATUSES and location:
                    url = urljoin(str(response.url), location)
                    continue
                await self._read_body(response, fetched)
                return
        fetched.error = "redirect_loop"

    async def _read_body(
        self, response: aiohttp.ClientResponse, fetched: Fetched
    ) -> None:
        fetched.status = response.status
        fetched.final_url = str(response.url)
        fetched.content_type = response.headers.get("content-type", "")
        fetched.charset = response.charset
        failed = status_error(response.status)

        declared_html = html_type(fetched.content_type)
        if declared_html is False:
            fetched.error = failed or "not_html"
            return
        if (response.content_length or 0) > self.max_bytes:
            fetched.error = failed or "too_large"
            return
        decompressor = decompressor_for(
            ", ".join(response.headers.getall("content-encoding", ()))
        )
        if decompressor is None:
            fetched.error = failed or "other"
            return

        body = bytearray()
        async for chunk in response.content.iter_chunked(READ_CHUNK):
            if not decompressor.feed(chunk, body, self.max_bytes):
                fetched.error = failed or "too_large"
                return
            if decompressor.finished:
                break

        if declared_html is None and not sniffs_html(body):
            fetched.error = failed or "not_html"
            return
        fetched.body = bytes(body)
        fetched.error = failed


class ScrapingDog:
    def __init__(
        self,
        api_key: str,
        concurrency: int = 8,
        timeout: float = 60.0,
        max_bytes: int = 5_000_000,
        endpoint: str = SCRAPINGDOG_URL,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_bytes = max_bytes
        self.slots = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: aiohttp.ClientSession | None = None
        self.requests: Counter[str] = Counter()

    async def fetch(
        self, url: str, rendered: bool = False, deadline: float | None = None
    ) -> Fetched:
        async with self.slots:
            for attempt in range(1, SCRAPINGDOG_ATTEMPTS + 1):
                fetched = await self._attempt(url, rendered, deadline)
                fetched.attempts = attempt
                again = fetched.error == "timeout" or (
                    fetched.status >= 500
                    or fetched.status in SCRAPINGDOG_RETRY_STATUSES
                )
                if not again:
                    break
            return fetched

    async def aclose(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def __aenter__(self) -> ScrapingDog:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    async def _attempt(
        self, url: str, rendered: bool, deadline: float | None
    ) -> Fetched:
        if self.session is None:
            # No brotli: a stale brotli library breaks aiohttp's decoder.
            self.session = aiohttp.ClientSession(
                timeout=self.timeout, headers={"Accept-Encoding": "gzip, deflate"}
            )
        fetched = Fetched(url=url, final_url=url, fetched_at=datetime.now(timezone.utc))
        params = {
            "api_key": self.api_key,
            "url": url,
            "dynamic": "true" if rendered else "false",
        }
        allowed = self.timeout.total
        if deadline is not None:
            allowed = min(allowed, deadline - time.time())
        if allowed <= 0:
            fetched.error = "timeout"
            return fetched
        self.requests["rendered" if rendered else "plain"] += 1
        started = time.monotonic()
        try:
            async with self.session.get(
                self.endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=allowed),
            ) as response:
                fetched.status = response.status
                fetched.content_type = response.headers.get("content-type", "")
                fetched.charset = response.charset
                if response.status == 200:
                    fetched.body = await _read_capped(response, self.max_bytes)
                    if fetched.body is None:
                        fetched.error = "too_large"
                else:
                    fetched.error = status_error(response.status)
        except asyncio.TimeoutError:
            fetched.error = "timeout"
        except aiohttp.ClientError:
            fetched.error = "connect"
        fetched.elapsed_ms = int((time.monotonic() - started) * 1000)
        return fetched


class Decompressor:
    def __init__(self, wbits: int | None):
        self.stream = zlib.decompressobj(wbits) if wbits else None
        self.retry_raw = wbits == zlib.MAX_WBITS

    @property
    def finished(self) -> bool:
        return self.stream is not None and self.stream.eof

    def feed(self, data: bytes, body: bytearray, max_bytes: int) -> bool:
        if self.stream is None:
            body += data
            return len(body) <= max_bytes
        while data and not self.stream.eof:
            body += self._inflate(data, max_bytes + 1 - len(body))
            if len(body) > max_bytes:
                return False
            data = self.stream.unconsumed_tail
        return True

    def _inflate(self, data: bytes, limit: int) -> bytes:
        retry_raw, self.retry_raw = self.retry_raw, False
        try:
            return self.stream.decompress(data, limit)
        except zlib.error:
            if not retry_raw:
                raise
        self.stream = zlib.decompressobj(-zlib.MAX_WBITS)
        return self.stream.decompress(data, limit)


class GuardedResolver(AbstractResolver):
    def __init__(self, resolve: Resolver):
        self._resolve = resolve

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        refuse_local(host)
        addresses = list(dict.fromkeys(await self._resolve(host, port)))
        if not addresses or not all(map(is_public, addresses)):
            raise ForbiddenAddress(host)
        return [
            {
                "hostname": host,
                "host": address,
                "port": port,
                "family": socket.AF_INET6 if ":" in address else socket.AF_INET,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
            for address in addresses
        ]

    async def close(self) -> None:
        pass


async def resolve_host(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def refuse_local(host: str) -> None:
    name = host.rstrip(".").lower()
    if _ip(name) is not None:
        local = not is_public(name)
    else:
        tld = name.rpartition(".")[2]
        local = (
            "." not in name
            or name.endswith(LOCAL_SUFFIXES)
            or tld.isdigit()
            or tld.startswith("0x")
        )
    if local:
        raise ForbiddenAddress(host)


def is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in IPV4_COMPATIBLE:
            return False
        if ip in NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        else:
            ip = ip.ipv4_mapped or ip.sixtofour or ip
    return ip.is_global and not ip.is_multicast


def browser_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Upgrade-Insecure-Requests": "1",
    }


def needs_fallback(error: str | None, status: int) -> bool:
    """Only failures another address could fix; a 404 or a PDF stays what it is."""
    return status >= 500 or status in FALLBACK_STATUSES or error in FALLBACK_ERRORS


async def _read_capped(response: aiohttp.ClientResponse, cap: int) -> bytes | None:
    body = bytearray()
    async for chunk in response.content.iter_chunked(READ_CHUNK):
        body += chunk
        if len(body) > cap:
            return None
    return bytes(body)


def retryable(fetched: Fetched) -> bool:
    if fetched.error == "http_4xx":
        return fetched.status in RETRY_STATUSES
    return fetched.error in RETRYABLE


def status_error(status: int) -> str | None:
    if 200 <= status < 300:
        return None
    if 400 <= status < 500:
        return "http_4xx"
    if status >= 500:
        return "http_5xx"
    return "other"


def html_type(content_type: str) -> bool | None:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in SNIFFED_TYPES:
        return None
    return mime in HTML_TYPES


def sniffs_html(body: bytes | bytearray) -> bool:
    head = bytes(body[:2048]).lower()
    return b"<html" in head or b"<!doctype html" in head


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout"
    causes = _causes(exc)
    if any(isinstance(cause, DNS_ERRORS) for cause in causes):
        return "dns"
    if any(isinstance(cause, TLS_ERRORS) for cause in causes):
        return "tls"
    if isinstance(exc, CONNECTION_ERRORS):
        return "connect"
    return "other"


def _causes(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def decompressor_for(content_encoding: str) -> Decompressor | None:
    codings = [coding.strip().lower() for coding in content_encoding.split(",")]
    codings = [coding for coding in codings if coding not in ("", "identity")]
    if len(codings) > 1 or (codings and codings[0] in UNDECODED):
        return None
    return Decompressor(DECODINGS.get(codings[0]) if codings else None)


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _domain(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def decode_html(body: bytes, charset: str | None) -> str:
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", "replace")
    for candidate in (charset, _meta_charset(body)):
        codec = _text_codec(candidate)
        if codec is None:
            continue
        try:
            return body.decode(codec, "replace")
        except UnicodeError:
            continue

    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        guess = charset_normalizer.from_bytes(body[:GUESS_BYTES]).best()
    return body.decode(guess.encoding if guess else "utf-8", "replace")


def _meta_charset(body: bytes) -> str | None:
    found = META_CHARSET.search(body[:4096])
    return found.group(1).decode("ascii", "ignore") if found else None


def _text_codec(name: str | None) -> str | None:
    if not name:
        return None
    try:
        info = codecs.lookup(name)
    except (LookupError, ValueError):
        return None
    if not getattr(info, "_is_text_encoding", True) or info.name in UNSAFE_CODECS:
        return None
    return info.name


def _check_proxy(index: int, proxy: str) -> None:
    host = None
    try:
        parts = urlsplit(proxy)
        host = parts.hostname
        valid = (
            parts.scheme in PROXY_SCHEMES
            and bool(host)
            and bool(parts.port)
            and parts.path in ("", "/")
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            f"proxy entry {index} (host {host or '?'}) is not"
            " http(s)://user:pass@host:port"
        ) from None
