from __future__ import annotations

import asyncio
import gzip
import io
import logging
import socket
import time
import traceback
import tracemalloc
import zlib
from asyncio.selector_events import BaseSelectorEventLoop
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import pytest

from desearch.extraction.schema import PAGE_SCHEMA
from desearch.fetch import (
    MAX_REDIRECTS,
    Fetched,
    Fetcher,
    ForbiddenAddress,
    is_public,
    refuse_local,
)
from neurons.miners import miner as miner_script
from neurons.miners.config import Settings
from neurons.miners.miner import Miner
from neurons.miners.rows import (
    ROW_GROUP_ROWS,
    UploadWriter,
    build_row,
    decode_html,
    write_parquet,
)
from tests.synthetic import CHALLENGE

CAP = 1_000_000
GZIP, ZLIB, RAW = 16 + zlib.MAX_WBITS, zlib.MAX_WBITS, -zlib.MAX_WBITS
HTML = (
    "<!doctype html><html><head><title>Café au lait</title></head><body><p>"
    + "The quick brown fox jumps over the lazy dog. " * 200
    + "</p></body></html>"
).encode()
PUBLIC, CDN = "93.184.216.34", "151.101.1.1"
DNS = {
    "news.example": [PUBLIC],
    "cdn.example": [CDN],
    "intranet.example": ["10.1.2.3"],
    "split.example": [PUBLIC, "127.0.0.1"],
    "nat64.example": ["64:ff9b::a9fe:a9fe"],
    "compat.example": ["::5db8:d822"],
    "mixed.example": [PUBLIC, "2002:a00:1::1"],
    "sixtofour.example": ["2002:5db8:d822::1"],
}
SECRET = "s3cr3tpw"
CLAIM = "/v1/tasks/claim"


def compress(data: bytes, wbits: int) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, wbits)
    return packer.compress(data) + packer.flush()


def bomb(megabytes: int, wbits: int) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, wbits)
    block = bytes(1 << 20)
    return b"".join(
        [packer.compress(block) for _ in range(megabytes)] + [packer.flush()]
    )


def page(body: bytes, charset: str | None = None, url: str = "https://site.example/"):
    return Fetched(
        url=url,
        final_url=url,
        fetched_at=datetime.now(timezone.utc),
        status=200,
        content_type="text/html",
        charset=charset,
        body=body,
    )


class Scripted:
    def __init__(self, reply):
        self.reply = reply
        self.requests: list[str] = []
        self.open: set[asyncio.Task] = set()

    async def __aenter__(self) -> Scripted:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        return self

    async def __aexit__(self, *_) -> None:
        for task in self.open:
            task.cancel()
        self.server.close()
        await self.server.wait_closed()

    async def finalized(self) -> None:
        deadline = time.monotonic() + 5
        while self.open and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not self.open, "the client left the connection open"

    async def _handle(self, reader, writer) -> None:
        self.open.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            self.requests.append(head.split(b"\r\n", 1)[0].decode())
            answer = self.reply(head)
            if isinstance(answer, bytes):
                writer.write(answer)
            else:
                async for piece in answer:
                    writer.write(piece)
                    await writer.drain()
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            self.open.discard(asyncio.current_task())


def head(status: int, headers: list[tuple[str, str]]) -> bytes:
    lines = [f"HTTP/1.1 {status} Scripted", *(f"{k}: {v}" for k, v in headers)]
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def chunked(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def response(
    payload: bytes,
    encoding: str = "",
    chunk: int | None = None,
    status: int = 200,
    headers: list | None = None,
):
    headers = list(headers or [("content-type", "text/html; charset=utf-8")])
    if encoding:
        headers.append(("content-encoding", encoding))
    if chunk is None:
        whole = head(status, [*headers, ("content-length", str(len(payload)))])
        return lambda _: whole + payload

    async def pieces():
        yield head(status, [*headers, ("transfer-encoding", "chunked")])
        for start in range(0, len(payload), chunk):
            yield chunked(payload[start : start + chunk])
        yield b"0\r\n\r\n"

    return lambda _: pieces()


def streamed(headers: list[tuple[str, str]], blocks):
    async def pieces():
        yield head(200, [*headers, ("transfer-encoding", "chunked")])
        while True:
            yield chunked(blocks())

    return lambda _: pieces()


def headers_only(*headers: tuple[str, str]):
    """Headers, then a body that never comes: reading it would time out."""

    async def pieces():
        yield head(200, list(headers))
        await asyncio.Event().wait()

    return lambda _: pieces()


def read(reply, max_bytes: int = CAP, finalize: bool = True) -> tuple[Fetched, int]:
    async def go() -> tuple[Fetched, int]:
        async with Scripted(reply) as server:
            fetcher = Fetcher(
                Settings(max_bytes=max_bytes, allow_private=True, timeout=5)
            )
            tracemalloc.start()
            try:
                fetched = await fetcher._attempt(fetcher.next_route(), server.url, None)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
                await fetcher.aclose()
            if finalize:
                await server.finalized()
            return fetched, peak

    return asyncio.run(go())


@pytest.mark.parametrize(
    "charset",
    [
        "undefined",
        "idna",
        "punycode",
        "unicode_escape",
        "raw-unicode-escape",
        "rot13",
        "base64",
        "hex",
        "zlib",
        "bz2",
        "uu",
        "quopri",
        "no-such-codec",
        "utf\x00",
    ],
)
def test_unsafe_charsets_fall_back_to_utf8(charset):
    assert decode_html(HTML, charset) == HTML.decode()


@pytest.mark.parametrize("charset", ["undefined", "idna"])
def test_hostile_charset_in_header_or_meta_still_builds_a_row(charset):
    meta = HTML.replace(b"<head>", f'<head><meta charset="{charset}">'.encode())
    equiv = HTML.replace(
        b"<head>",
        f'<head><meta http-equiv="Content-Type" content="text/html; charset={charset}">'.encode(),
    )
    for fetched in (page(HTML, charset), page(meta), page(equiv)):
        row = build_row(fetched, CAP)
        assert row["error"] is None
        assert row["title"] == "Café au lait"


async def crawled(miner: Miner, urls: list[str]) -> list[dict]:
    pages = UploadWriter("t", "hk")
    await miner.crawl(urls, None, pages)
    return pq.read_table(io.BytesIO(await pages.finish())).to_pylist()


class StaticFetcher:
    def __init__(self, pages: dict[str, Fetched]):
        self.pages = pages

    async def fetch(self, url: str, deadline: float | None = None) -> Fetched:
        return self.pages[url]

    async def aclose(self) -> None:
        pass


class ClaimApi:
    hotkey = "5HardeningTestHotkey"

    def __init__(self):
        self.calls: list[str] = []

    async def post(self, path: str, body: dict | None = None) -> dict:
        self.calls.append(path)
        return {
            "task": {"task_id": "t-r", "urls": [], "upload": {}},
            "receipt": {"body": {"task_id": "t-r", "outcome": "issued"}},
        }

    async def aclose(self) -> None:
        pass


def test_a_row_that_fails_to_build_becomes_an_error_row(monkeypatch):
    urls = ["https://site.example/good", "https://site.example/poison"]
    pages = {url: page(HTML, url=url) for url in urls}
    real = miner_script.build_row

    def build(fetched: Fetched, max_bytes: int) -> dict:
        if fetched.url.endswith("/poison"):
            raise UnicodeError("undefined encoding")
        return real(fetched, max_bytes)

    monkeypatch.setattr(miner_script, "build_row", build)
    miner = Miner(Settings(), api=ClaimApi(), fetcher=StaticFetcher(pages))

    async def go() -> list[dict]:
        try:
            return await crawled(miner, urls)
        finally:
            await miner.aclose()

    rows = sorted(asyncio.run(go()), key=lambda row: urls.index(row["url"]))

    assert [row["url"] for row in rows] == urls
    assert [row["error"] for row in rows] == [None, "other"]
    assert rows[1]["html"] is None and rows[1]["status"] == 200
    table = pq.read_table(io.BytesIO(write_parquet(rows, "t-poison", "hk")))
    assert table.schema.equals(PAGE_SCHEMA) and table.num_rows == 2


def test_a_receipt_that_cannot_be_written_does_not_drop_the_claim(tmp_path, caplog):
    miner = Miner(
        Settings(receipts_file=str(tmp_path)), api=ClaimApi(), fetcher=StaticFetcher({})
    )
    worked: list[str] = []

    async def work(task: dict) -> None:
        worked.append(task["task_id"])

    miner.process_task = work

    async def go() -> float:
        try:
            delay = await miner.poll()
            await asyncio.gather(*miner.in_flight)
            return delay
        finally:
            await miner.aclose()

    with caplog.at_level(logging.WARNING, logger="miner"):
        delay = asyncio.run(go())

    assert delay == 0.0 and worked == ["t-r"]
    assert "could not keep receipt" in caplog.text


@pytest.mark.parametrize(
    ("encoding", "wbits"),
    [("gzip", GZIP), ("x-gzip", GZIP), ("deflate", ZLIB), ("deflate", RAW)],
)
def test_compression_bombs_stop_at_the_cap(encoding, wbits):
    payload = bomb(128, wbits)
    assert len(payload) < CAP // 4

    fetched, peak = read(response(payload, encoding))

    assert fetched.error == "too_large" and fetched.body is None
    assert peak < 3 * CAP


def test_an_endless_compressed_stream_is_cut_off_at_the_cap():
    block = bytes(1 << 20)
    packer = zlib.compressobj(9, zlib.DEFLATED, GZIP)

    def gzipped() -> bytes:
        return packer.compress(block) + packer.flush(zlib.Z_SYNC_FLUSH)

    headers = [("content-type", "text/html"), ("content-encoding", "gzip")]
    fetched, peak = read(streamed(headers, gzipped))

    assert fetched.error == "too_large"
    assert peak < 3 * CAP


def test_a_bomb_behind_an_error_status_keeps_the_status_error():
    fetched, _ = read(response(bomb(64, GZIP), "gzip", status=500))
    assert fetched.error == "http_5xx" and fetched.status == 500


@pytest.mark.parametrize(
    ("encoding", "payload"),
    [
        ("", HTML),
        ("identity", HTML),
        ("gzip", gzip.compress(HTML)),
        ("x-gzip", gzip.compress(HTML)),
        ("GZip", gzip.compress(HTML)),
        ("deflate", compress(HTML, ZLIB)),
        ("deflate", compress(HTML, RAW)),
    ],
)
@pytest.mark.parametrize("chunk", [None, 7])
def test_supported_encodings_decode_to_the_original(encoding, payload, chunk):
    fetched, _ = read(response(payload, encoding, chunk))
    assert fetched.error is None and fetched.body == HTML


def test_a_body_exactly_at_the_cap_passes_and_one_byte_more_fails():
    exact = HTML.ljust(CAP, b" ")
    for body, error in ((exact, None), (exact + b" ", "too_large")):
        for encoding, payload in (("", body), ("gzip", gzip.compress(body))):
            fetched, _ = read(response(payload, encoding, chunk=65536))
            assert fetched.error == error, encoding
            assert fetched.body == (body if error is None else None)


@pytest.mark.parametrize(
    "headers",
    [
        [("content-encoding", "gzip, gzip")],
        [("content-encoding", "gzip"), ("content-encoding", "gzip")],
        [("content-encoding", "br")],
        [("content-encoding", "zstd")],
        [("content-encoding", "compress")],
        [("content-encoding", "gzip, deflate")],
        [("content-encoding", "x-compress")],
        [("content-encoding", "BR")],
        [("content-encoding", "Zstd")],
        [("content-encoding", "gzip, utf-8")],
        [("content-encoding", "utf-8, utf-8")],
    ],
)
def test_stacked_or_unadvertised_encodings_are_refused(headers):
    stacked = gzip.compress(gzip.compress(HTML))
    fetched, _ = read(
        response(stacked, headers=[("content-type", "text/html"), *headers])
    )
    assert fetched.error == "other" and fetched.body is None


def test_corrupt_compressed_bodies_are_other():
    fetched, _ = read(response(b"\x1f\x8b\x08\x00garbage-not-gzip" * 10, "gzip"))
    assert fetched.error == "other" and fetched.body is None


def test_prechecks_still_run_before_any_decoding():
    too_long = headers_only(
        ("content-type", "text/html"),
        ("content-encoding", "gzip"),
        ("content-length", str(CAP + 1)),
    )
    pdf = headers_only(
        ("content-type", "application/pdf"),
        ("content-encoding", "br"),
        ("content-length", "10"),
    )
    assert read(too_long, finalize=False)[0].error == "too_large"
    assert read(pdf, finalize=False)[0].error == "not_html"


class Network:
    """Sends connections for public addresses to loopback servers; the rest never answer."""

    def __init__(self):
        self.routes: dict[str, int] = {}
        self.refused: set[str] = set()
        self.connected: list[str] = []


@pytest.fixture
def network(monkeypatch) -> Network:
    made = Network()
    real = BaseSelectorEventLoop.sock_connect

    async def sock_connect(loop, sock, address):
        host = address[0]
        if host in ("127.0.0.1", "::1"):
            return await real(loop, sock, address)
        made.connected.append(host)
        if host in made.refused:
            raise ConnectionRefusedError(61, "Connection refused")
        if host not in made.routes:
            await asyncio.sleep(3600)
        loopback = "::ffff:127.0.0.1" if sock.family == socket.AF_INET6 else "127.0.0.1"
        return await real(loop, sock, (loopback, made.routes[host], *address[2:]))

    monkeypatch.setattr(BaseSelectorEventLoop, "sock_connect", sock_connect)
    return made


def resolver(calls: list[str]):
    async def resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        if host not in DNS:
            raise socket.gaierror(
                socket.EAI_NONAME, "nodename nor servname provided, or not known"
            )
        return DNS[host]

    return resolve


def redirect(location: str) -> bytes:
    return (
        f"HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n"
    ).encode()


def ok(body: bytes = HTML) -> bytes:
    head = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n\r\n"
    return head.encode() + body


def direct(
    network: Network,
    url: str,
    replies: dict[str, bytes] | None = None,
    resolve=None,
    timeout: float = 5,
):
    calls: list[str] = []

    async def go() -> Fetched:
        async with AsyncExitStack() as stack:
            for address, reply in (replies or {}).items():
                served = await stack.enter_async_context(Scripted(lambda _, r=reply: r))
                network.routes[address] = served.port
            fetcher = Fetcher(
                Settings(timeout=timeout), resolve=resolve or resolver(calls)
            )
            try:
                return await fetcher.fetch(url)
            finally:
                await fetcher.aclose()

    return asyncio.run(go()), calls, network.connected


@pytest.mark.parametrize(
    ("url", "resolved"),
    [
        ("http://127.0.0.1:8080/", []),
        ("http://[::1]/", []),
        ("http://[::ffff:127.0.0.1]/", []),
        ("http://169.254.169.254/latest/meta-data/", []),
        ("http://0.0.0.0/", []),
        ("http://127.1/", []),
        ("http://localhost:3000/", []),
        ("http://metadata.google.internal/", []),
        ("http://intranet.example/", ["intranet.example"]),
        ("http://split.example/", ["split.example"]),
        ("http://nat64.example/", ["nat64.example"]),
        ("http://compat.example/", ["compat.example"]),
        ("http://mixed.example/", ["mixed.example"]),
        ("http://[::127.0.0.1]/", []),
        ("http://[64:ff9b::127.0.0.1]/", []),
        ("http://[2002:7f00:1::]/", []),
    ],
)
def test_private_targets_are_refused_before_connecting(network, url, resolved):
    fetched, calls, connected = direct(network, url)
    assert fetched.error == "other"
    assert calls == resolved and connected == []


@pytest.mark.parametrize(
    "location",
    [
        "http://intranet.example/admin",
        "http://127.0.0.1:2375/containers/json",
        "http://[::ffff:127.0.0.1]/",
        "http://10.0.0.1/",
        "http://localhost/",
    ],
)
def test_redirects_into_private_space_are_refused(network, location):
    fetched, _, connected = direct(
        network, "http://news.example/a", {PUBLIC: redirect(location)}
    )
    assert fetched.error == "other" and fetched.body is None
    assert connected == [PUBLIC]


def test_public_redirects_connect_to_the_checked_address(network):
    replies = {PUBLIC: redirect("http://cdn.example/a"), CDN: ok()}
    fetched, calls, connected = direct(network, "http://news.example/a", replies)

    assert fetched.error is None and fetched.body == HTML
    assert fetched.final_url == "http://cdn.example/a"
    assert calls == ["news.example", "cdn.example"]
    assert connected == [PUBLIC, CDN]


def test_unresolvable_hosts_still_report_dns(network):
    fetched, calls, connected = direct(network, "http://nowhere.example/")
    assert fetched.error == "dns"
    assert calls == ["nowhere.example"] and connected == []


def test_a_refused_address_falls_through_to_the_next_one(network):
    v6 = "2606:4700:4700::1111"
    network.refused.add(v6)

    async def resolve(host: str, port: int) -> list[str]:
        return [v6, PUBLIC, PUBLIC]

    fetched, _, connected = direct(
        network, "http://news.example/", {PUBLIC: ok()}, resolve=resolve
    )

    assert fetched.error is None and fetched.body == HTML
    assert connected == [v6, PUBLIC]


def test_proxied_routes_refuse_local_targets_without_resolving():
    calls: list[str] = []
    urls = [
        "http://127.0.0.1/",
        "http://[fe80::1]/",
        "http://localhost:8080/",
        "http://svc.internal/",
        "http://printer.local/",
        "http://db/",
    ]

    async def go() -> list[Fetched]:
        settings = Settings(timeout=5, proxy_urls=("http://proxy.example:3128",))
        fetcher = Fetcher(settings, resolve=resolver(calls))
        try:
            return [await fetcher.fetch(url) for url in urls]
        finally:
            await fetcher.aclose()

    assert [fetched.error for fetched in asyncio.run(go())] == ["other"] * len(urls)
    assert calls == []


@pytest.mark.parametrize(
    ("location", "error"),
    [("http://10.0.0.5/", "other"), ("http://cdn.example/a", None)],
)
def test_proxied_redirects_are_checked_on_every_hop(location, error):
    def proxy(request: bytes) -> bytes:
        target = request.split(b" ", 2)[1].decode()
        return (
            redirect(location) if urlsplit(target).hostname == "news.example" else ok()
        )

    async def go() -> tuple[Fetched, list[str]]:
        async with Scripted(proxy) as server:
            settings = Settings(timeout=5, proxy_urls=(server.url.rstrip("/"),))
            fetcher = Fetcher(settings)
            try:
                fetched = await fetcher.fetch("http://news.example/a")
            finally:
                await fetcher.aclose()
            hosts = [urlsplit(line.split()[1]).hostname for line in server.requests]
            return fetched, hosts

    fetched, seen = asyncio.run(go())
    assert fetched.error == error
    assert seen == (["news.example"] if error else ["news.example", "cdn.example"])


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.0.0.8",
        "172.17.0.1",
        "192.168.1.1",
        "100.64.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:10.0.0.1",
        "::10.0.0.1",
        "64:ff9b::a00:1",
        "2002:a9fe:a9fe::",
        "localhost",
        "LOCALHOST.",
        "app.localhost",
        "printer.local",
        "metadata.google.internal",
        "router.home.arpa",
        "db",
        "127.1",
        "0x7f.1",
        "2130706433",
        "",
    ],
)
def test_local_hosts_are_refused(host):
    with pytest.raises(ForbiddenAddress):
        refuse_local(host)


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "news.example.",
        PUBLIC,
        "2606:4700:4700::1111",
        "xn--p1ai.xn--p1ai",
    ],
)
def test_public_hosts_pass_the_name_check(host):
    refuse_local(host)


def test_relative_redirects_are_followed_and_loops_are_cut_off():
    def hops(request: bytes) -> bytes:
        path = request.split(b" ", 2)[1]
        if path == b"/start":
            return redirect("next")
        return ok() if path == b"/next" else redirect("/loop")

    async def go() -> tuple[list[Fetched], int]:
        async with Scripted(hops) as server:
            fetcher = Fetcher(Settings(timeout=5, allow_private=True))
            try:
                followed = await fetcher.fetch(server.url + "start")
                looped = await fetcher.fetch(server.url + "loop")
            finally:
                await fetcher.aclose()
            return [followed, looped], len(server.requests)

    (followed, looped), requests = asyncio.run(go())

    assert followed.error is None and followed.final_url.endswith("/next")
    assert looped.error == "redirect_loop" and looped.body is None
    assert requests == 2 + MAX_REDIRECTS + 1


def test_allow_private_turns_the_guard_off():
    async def go() -> tuple[list[Fetched], int]:
        async with Scripted(lambda _: ok()) as server:
            fetched = []
            for allow_private in (False, True):
                fetcher = Fetcher(Settings(timeout=5, allow_private=allow_private))
                try:
                    fetched.append(await fetcher.fetch(server.url))
                finally:
                    await fetcher.aclose()
            return fetched, len(server.requests)

    (guarded, open_), requests = asyncio.run(go())

    assert guarded.error == "other" and guarded.body is None
    assert open_.error is None and open_.body == HTML
    assert requests == 1


@pytest.mark.parametrize(
    "entry",
    [
        f"user:{SECRET}@1.2.3.4:8080",
        f"1.2.3.4:8080:user:{SECRET}",
        f"http://user:pa#{SECRET}@proxy.example:8080",
        f"http://user:1234/{SECRET}@proxy.example:1",
        f"socks5://user:{SECRET}@proxy.example:1080",
        f"http://user:{SECRET}@proxy.example",
        f"http://user:{SECRET}@:8080",
        f"http://user:{SECRET}@[::1:8080",
        f"http://user:{SECRET}@proxy.example:99999",
    ],
)
def test_bad_proxy_entries_fail_without_leaking_the_password(entry):
    with pytest.raises(ValueError) as caught:
        Settings.from_env({"PROXY_URLS": f"http://ok.example:1,{entry}"})

    error = caught.value
    assert "proxy entry 1" in str(error)
    assert SECRET not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None and error.__suppress_context__


def test_good_proxy_entries_are_kept():
    proxies = "http://user:p%40ss@proxy.example:3128, https://10.0.0.1:443/"
    settings = Settings.from_env({"PROXY_URLS": proxies})
    assert settings.proxy_urls == (
        "http://user:p%40ss@proxy.example:3128",
        "https://10.0.0.1:443/",
    )


@pytest.mark.parametrize(
    "encoding",
    [
        "utf-8",
        "UTF-8",
        "none",
        "text",
        "binary",
        "8bit",
        "plain",
        "Identity",
        " IDENTITY ",
    ],
)
@pytest.mark.parametrize("chunk", [None, 7])
def test_unknown_single_encodings_are_read_as_identity(encoding, chunk):
    fetched, _ = read(response(HTML, encoding, chunk))
    assert fetched.error is None and fetched.body == HTML


@pytest.mark.parametrize("encoding", ["utf-8", "none", "binary"])
def test_identity_like_encodings_stay_capped(encoding):
    headers = [("content-type", "text/html"), ("content-encoding", encoding)]
    fetched, peak = read(streamed(headers, lambda: bytes(1 << 16)))

    assert fetched.error == "too_large" and fetched.body is None
    assert peak < 3 * CAP


def test_a_blackholed_first_address_does_not_eat_the_fetch_budget(network):
    v6 = "2606:4700:4700::1111"

    async def resolve(host: str, port: int) -> list[str]:
        return [v6, PUBLIC]

    fetched, _, connected = direct(
        network, "http://news.example/", {PUBLIC: ok()}, resolve=resolve, timeout=5
    )

    assert fetched.error is None and fetched.body == HTML
    assert connected == [v6, PUBLIC]
    assert fetched.attempts == 1 and fetched.elapsed_ms < 1000


def test_when_every_address_is_dead_the_fetch_times_out(network):
    async def resolve(host: str, port: int) -> list[str]:
        return [PUBLIC, CDN]

    fetched, _, connected = direct(
        network, "http://news.example/", resolve=resolve, timeout=0.5
    )

    assert fetched.error == "timeout" and fetched.attempts == 2
    assert connected[0] == PUBLIC and set(connected) == {PUBLIC, CDN}


class ScriptedApi:
    hotkey = "5HardeningTestHotkey"

    def __init__(self, *answers, then=None):
        self.answers = list(answers)
        self.then = then or {"refusal": {"code": "QUEUE_EMPTY", "inputs": {}}}
        self.calls: list[str] = []

    async def post(self, path: str, body: dict | None = None) -> dict:
        self.calls.append(path)
        if path != CLAIM:
            return {}
        answer = self.answers.pop(0) if self.answers else self.then
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def aclose(self) -> None:
        pass


def _recording_miner(api: ScriptedApi, settings: Settings, pause: float = 0.0):
    miner = Miner(settings, api=api, fetcher=StaticFetcher({}))
    finalized: list[str] = []

    async def work(task: dict) -> None:
        await asyncio.sleep(pause)
        finalized.append(task["task_id"])

    miner.process_task = work
    return miner, finalized


def test_an_unexpected_poll_error_backs_off_and_keeps_leasing(monkeypatch, caplog):
    monkeypatch.setattr(miner_script, "ERROR_BACKOFF", 0.01)
    monkeypatch.setitem(miner_script.BACKOFF, "QUEUE_EMPTY", 0.01)
    api = ScriptedApi(
        RuntimeError("claim socket went away"),
        ["not", "an", "answer"],
        {"task": {"task_id": "t-late", "urls": [], "upload": {}}},
    )
    miner, finalized = _recording_miner(api, Settings(idle_exit=1))

    with caplog.at_level(logging.ERROR, logger="miner"):
        asyncio.run(asyncio.wait_for(miner.run(), 5))

    assert finalized == ["t-late"] and api.answers == []
    assert caplog.text.count("claim poll failed") == 2


@pytest.mark.parametrize(
    "refusal, wait",
    [
        ({"code": "NO_CAPACITY", "inputs": {"retry_after": 5.0}}, 5.0),
        ({"code": "LOCKED_OUT", "inputs": {"retry_after": 43_000.0}}, 3600.0),
        (
            {"code": "KIND_CLOSED", "inputs": {"kind": "embed", "retry_after": 3600.0}},
            3600.0,
        ),
        ({"code": "RATE_LIMITED", "inputs": {"retry_after": "soon"}}, 2.0),
        ({"code": "QUEUE_EMPTY", "inputs": {}}, 2.0),
    ],
)
def test_a_refused_miner_waits_as_long_as_the_api_asks(refusal, wait):
    miner = Miner(
        Settings(), api=ScriptedApi({"refusal": refusal}), fetcher=StaticFetcher({})
    )

    assert asyncio.run(miner.poll()) == wait


def test_in_flight_work_is_finalized_when_every_later_poll_breaks(monkeypatch):
    monkeypatch.setattr(miner_script, "ERROR_BACKOFF", 0.01)
    api = ScriptedApi(
        {"task": {"task_id": "t-held", "urls": [], "upload": {}}},
        then=RuntimeError("claim endpoint is down"),
    )
    miner, finalized = _recording_miner(api, Settings(shutdown_grace=5), pause=0.2)

    async def scenario() -> None:
        running = asyncio.create_task(miner.run())
        while api.calls.count(CLAIM) < 5 and not running.done():
            await asyncio.sleep(0.01)
        assert not finalized
        miner.stop()
        await asyncio.wait_for(running, 5)

    asyncio.run(scenario())

    assert finalized == ["t-held"]


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("::7f00:1", False),
        ("::a00:1", False),
        ("::5db8:d822", False),
        ("64:ff9b::7f00:1", False),
        ("64:ff9b::a9fe:a9fe", False),
        ("64:ff9b::5db8:d822", True),
        ("64:ff9b:1::5db8:d822", False),
        ("2002:7f00:1::", False),
        ("2002:a00:1::1", False),
        ("2002:5db8:d822::1", True),
        ("::ffff:7f00:1", False),
        ("::ffff:5db8:d822", True),
        ("2606:4700:4700::1111", True),
        (PUBLIC, True),
    ],
)
def test_embedded_ipv4_addresses_are_judged_by_the_ipv4(address, public):
    assert is_public(address) is public


def test_a_public_6to4_address_is_still_reachable(network):
    fetched, calls, connected = direct(
        network, "http://sixtofour.example/", {"2002:5db8:d822::1": ok()}
    )
    assert fetched.error is None and fetched.body == HTML
    assert connected == ["2002:5db8:d822::1"]


@pytest.mark.parametrize(
    "error, status, again",
    [
        ("http_4xx", 403, True),
        ("http_4xx", 429, True),
        ("http_4xx", 408, True),
        ("http_4xx", 404, False),
        ("http_4xx", 410, False),
        ("blocked", 200, True),
        ("timeout", 0, True),
        ("connect", 0, True),
        ("http_5xx", 502, True),
        ("not_html", 200, False),
        ("too_large", 200, False),
        (None, 200, False),
    ],
)
def test_only_failures_about_our_address_are_worth_another_route(error, status, again):
    from desearch.fetch import retryable

    fetched = Fetched(
        url="https://site.example/a",
        final_url="https://site.example/a",
        fetched_at=datetime.now(timezone.utc),
        status=status,
        error=error,
    )
    assert retryable(fetched) is again


class FakeScrapingDog:
    def __init__(self):
        self.calls: list[str] = []

    async def fetch(self, url: str, rendered: bool = False, deadline=None) -> Fetched:
        self.calls.append(url)
        return page(HTML, url=url)

    async def aclose(self) -> None:
        pass


def refused(url: str, status: int) -> Fetched:
    return Fetched(
        url=url,
        final_url=url,
        fetched_at=datetime.now(timezone.utc),
        status=status,
        error="http_4xx",
    )


def test_scrapingdog_is_tried_only_for_what_another_address_could_fix():
    pages = {
        "https://a.example/refused": refused("https://a.example/refused", 403),
        "https://a.example/gone": refused("https://a.example/gone", 404),
        "https://a.example/challenge": page(
            CHALLENGE, url="https://a.example/challenge"
        ),
        "https://a.example/fine": page(HTML, url="https://a.example/fine"),
    }
    miner = Miner(Settings(), api=ClaimApi(), fetcher=StaticFetcher(pages))
    miner.scrapingdog = dog = FakeScrapingDog()

    async def go() -> list[dict]:
        try:
            return await crawled(miner, list(pages))
        finally:
            await miner.aclose()

    rows = {row["url"]: row for row in asyncio.run(go())}

    assert sorted(dog.calls) == [
        "https://a.example/challenge",
        "https://a.example/refused",
    ]
    assert rows["https://a.example/refused"]["error"] is None
    assert rows["https://a.example/challenge"]["error"] is None
    assert rows["https://a.example/gone"]["error"] == "http_4xx"


def test_a_slow_scrapingdog_call_does_not_hold_up_our_own_fetches():
    refused_url, fine = "https://a.example/refused", "https://b.example/fine"
    pages = {refused_url: refused(refused_url, 403), fine: page(HTML, url=fine)}
    fetched_ourselves = asyncio.Event()

    class OwnIp(StaticFetcher):
        async def fetch(self, url: str, deadline: float | None = None) -> Fetched:
            if url == fine:
                fetched_ourselves.set()
            return await super().fetch(url, deadline)

    class SlowDog(FakeScrapingDog):
        async def fetch(self, url: str, rendered: bool = False, deadline=None):
            await fetched_ourselves.wait()
            return await super().fetch(url, rendered, deadline)

    miner = Miner(Settings(concurrency=1), api=ClaimApi(), fetcher=OwnIp(pages))
    miner.scrapingdog = SlowDog()

    async def go() -> list[dict]:
        try:
            return await asyncio.wait_for(crawled(miner, [refused_url, fine]), 5)
        finally:
            await miner.aclose()

    assert all(row["error"] is None for row in asyncio.run(go()))


def test_without_a_key_the_miner_never_calls_scrapingdog():
    assert (
        Miner(Settings(), api=ClaimApi(), fetcher=StaticFetcher({})).scrapingdog is None
    )
    keyed = Miner(
        Settings(scrapingdog_api_key="k"), api=ClaimApi(), fetcher=StaticFetcher({})
    )
    assert keyed.scrapingdog is not None
    asyncio.run(keyed.aclose())


def test_the_upload_is_written_in_small_row_groups_as_rows_arrive():
    rows = [
        build_row(page(HTML, url=f"https://site.example/{n}"), CAP) for n in range(120)
    ]

    async def write() -> bytes:
        pages = UploadWriter("t-stream", "hk")
        for row in rows:
            await pages.add(row)
            assert len(pages.pending) < ROW_GROUP_ROWS
        return await pages.finish()

    upload = pq.ParquetFile(io.BytesIO(asyncio.run(write())))

    assert upload.num_row_groups == 3
    assert upload.schema_arrow.equals(PAGE_SCHEMA)
    assert [r["url"] for r in upload.read().to_pylist()] == [r["url"] for r in rows]


def test_a_cookie_gate_that_redirects_to_itself_loads_within_one_chain():
    from aiohttp import web

    from tests.local_http import serving

    seen: dict[str, list] = {"gate": [], "elsewhere": []}

    async def gate(request: web.Request) -> web.Response:
        if request.path == "/elsewhere":
            seen["elsewhere"].append(dict(request.cookies))
            return web.Response(text=HTML.decode(), content_type="text/html")
        seen["gate"].append(dict(request.cookies))
        if request.cookies.get("consent") != "yes":
            response = web.HTTPFound(request.path)
            response.set_cookie("consent", "yes")
            raise response
        if request.path == "/hop":
            raise web.HTTPFound(f"http://localhost:{request.url.port}/elsewhere")
        return web.Response(text=HTML.decode(), content_type="text/html")

    async def go() -> tuple[Fetched, Fetched]:
        async with serving(gate) as base:
            fetcher = Fetcher(Settings(timeout=5, allow_private=True))
            try:
                return await fetcher.fetch(base + "story"), await fetcher.fetch(
                    base + "hop"
                )
            finally:
                await fetcher.aclose()

    story, hop = asyncio.run(go())

    assert story.error is None and story.body == HTML
    assert hop.error is None and seen["elsewhere"] == [{}]


def test_a_gzip_cut_short_is_truncated_not_a_page():
    fetched, _ = read(response(gzip.compress(HTML)[:-40], "gzip"))
    assert fetched.error == "truncated" and fetched.body is None


def test_a_concatenated_gzip_decodes_every_member():
    half = len(HTML) // 2
    payload = gzip.compress(HTML[:half]) + gzip.compress(HTML[half:])
    fetched, _ = read(response(payload, "gzip", chunk=7))
    assert fetched.error is None and fetched.body == HTML


def test_trailing_junk_after_a_complete_gzip_is_ignored():
    fetched, _ = read(response(gzip.compress(HTML) + b"\r\n", "gzip"))
    assert fetched.error is None and fetched.body == HTML


def test_a_proxy_route_opens_a_fresh_connection_per_request():
    async def go():
        proxied = Fetcher(Settings(proxy_urls=("http://user:pw@proxy.example:3128",)))
        direct = Fetcher(Settings())
        try:
            return (
                proxied._session(proxied.routes[0]).connector.force_close,
                direct._session(direct.routes[0]).connector.force_close,
            )
        finally:
            await proxied.aclose()
            await direct.aclose()

    assert asyncio.run(go()) == (True, False)
