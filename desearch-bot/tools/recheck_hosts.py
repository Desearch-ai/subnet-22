"""Rechecks hosts as the bot reads them, robots.txt, the homepage and one sitemap or feed, at one request a second a host: walled hosts stay out, ok ones may lift a misfired out."""

import argparse
import csv
import gzip
import http.client
import ipaddress
import re
import socket
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

USER_AGENT = (
    "Mozilla/5.0 (compatible; DesearchBot/1.0; +https://www.desearch.ai/crawler)"
)
TOKEN = "desearchbot"
MAX_REDIRECTS = 5
MAX_BYTES = 512 * 1024
MAX_LISTING_BYTES = 2 * 1024 * 1024
TIMEOUT = 20
# The bot's limits: a whole request within a minute, and Crawl-delay up to a minute.
MAX_SECONDS = 60
MAX_CRAWL_DELAY = 60
MAX_PARALLEL = 20
MIN_CHARS = 200
SAMPLE = 2000
LANGUAGE_SAMPLE = 8000
WALL_STATUSES = {401, 402, 403, 429}
REDIRECTS = {301, 302, 303, 307, 308}
SITEMAP_GUESSES = ("/sitemap.xml", "/sitemap_index.xml")
FEED_GUESSES = ("/feed", "/rss.xml")
HTML_LANG = re.compile(r"""(?i)<html[^>]*\blang=["']?([a-zA-Z-]{2,8})""")
SCRIPT_OR_STYLE = re.compile(
    r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>|<svg[^>]*>.*?</svg>"
)
TAG = re.compile(r"<[^>]+>")
CHARSET = re.compile(rb"""(?i)charset=["']?([\w-]+)""")
BOT_WALL = re.compile(
    "(?i)(please enable javascript|enable javascript and refresh|you need to enable javascript|access denied|are you a robot|checking your browser|attention required)"
)
# Challenge pages the bot would read as thin text or as a page; here they are a wall.
CHALLENGE = re.compile(
    r"(?i)(<title>just a moment|cf-chl-|captcha-delivery\.com|px-captcha|_incapsula_resource|verify you are human)"
)
SITEMAP_LINE = re.compile(
    r"(?im)^[\s\x1c-\x1f]*sitemap[\s\x1c-\x1f]*:[\s\x1c-\x1f]*([^\s\x1c-\x1f]+)"
)
FIELDS = (
    "host",
    "verdict",
    "why",
    "robots_status",
    "crawl_delay",
    "home_status",
    "home_url",
    "chars",
    "declared",
    "language",
    "listing",
    "listing_status",
    "checked_at",
)


@dataclass
class Answer:
    status: int
    url: str
    body: bytes


@dataclass
class Check:
    host: str
    verdict: str = ""
    why: str = ""
    robots_status: str = ""
    crawl_delay: str = ""
    home_status: str = ""
    home_url: str = ""
    chars: str = ""
    declared: str = ""
    language: str = ""
    listing: str = ""
    listing_status: str = ""
    checked_at: str = ""
    rules: tuple = field(default=((), ()), repr=False)

    def row(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}


class Pacer:
    """One request a second to each host, its www name included, or its Crawl-delay once robots.txt names one."""

    def __init__(self, gap: float = 1.0):
        self.gap, self.last, self.delays, self.lock = gap, {}, {}, threading.Lock()

    def wait(self, host: str) -> None:
        host = host[4:] if host.startswith("www.") else host
        with self.lock:
            now = time.monotonic()
            gap = max(self.gap, self.delays.get(host, 0.0))
            at = max(now, self.last.get(host, -gap) + gap)
            self.last[host] = at
        time.sleep(at - now)

    def slow_to(self, host: str, delay: float | None) -> None:
        if delay:
            with self.lock:
                self.delays[host[4:] if host.startswith("www.") else host] = min(
                    delay, MAX_CRAWL_DELAY
                )


class PrivateAddress(OSError):
    pass


class Deadline:
    """The sockets one fetch opened, shut when its time is up so no slow answer holds a worker."""

    def __init__(self):
        self.sockets, self.passed = [], False

    def expire(self) -> None:
        self.passed = True
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


FETCH = threading.local()


def public_address(address: str) -> bool:
    return ipaddress.ip_address(address.split("%")[0]).is_global


def public_connection(
    address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **kwargs
):
    """socket.create_connection to the host's public addresses only, checked where the connection is made."""
    host, port = address
    deadline = getattr(FETCH, "deadline", None)
    last: OSError = PrivateAddress(host)
    for family, kind, proto, _, target in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        if not public_address(target[0]):
            continue
        if deadline and deadline.passed:
            raise TimeoutError("slow answer")
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if deadline:
                deadline.sockets.append(sock)
            sock.connect(target)
            return sock
        except OSError as error:
            last = error
            sock.close()
    raise last


class PublicHTTP(http.client.HTTPConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class PublicHTTPS(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(PublicHTTP, req)


class HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(PublicHTTPS, req, context=self._context)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), HTTPHandler, HTTPSHandler, NoRedirect
)


def gunzip(body: bytes, limit: int) -> bytes:
    if not body.startswith(b"\x1f\x8b"):
        return body
    try:
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, limit)
    except zlib.error:
        return body


def http_fetch(url: str, limit: int, pacer: Pacer) -> Answer:
    """GET over http(s) only, with redirects followed by hand, each hop paced for the host it goes to, all within MAX_SECONDS."""
    deadline = FETCH.deadline = Deadline()
    timer = threading.Timer(MAX_SECONDS, deadline.expire)
    timer.daemon = True
    timer.start()
    try:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https"):
                raise OSError(f"{parts.scheme} redirect")
            pacer.wait(parts.hostname or "")
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
            )
            try:
                with OPENER.open(request, timeout=TIMEOUT) as response:
                    body = response.read(limit)
                    if deadline.passed:
                        raise TimeoutError("slow answer")
                    return Answer(response.status, url, gunzip(body, limit))
            except urllib.error.HTTPError as error:
                location = error.headers.get("Location")
                if error.code in REDIRECTS and location:
                    url = urljoin(url, location)
                    continue
                try:
                    body = error.read(limit)
                except Exception:
                    body = b""
                return Answer(error.code, url, gunzip(body, limit))
            finally:
                if deadline.passed:
                    raise TimeoutError("slow answer")
        raise OSError("too many redirects")
    finally:
        timer.cancel()
        FETCH.deadline = None


def error_name(error: Exception) -> str:
    reason = getattr(error, "reason", None)
    if isinstance(reason, socket.gaierror):
        return "DNS"
    return type(reason if isinstance(reason, Exception) else error).__name__


def first_answer(host: str, path: str, limit: int, fetch) -> tuple[Answer | None, str]:
    last = "unknown"
    for scheme in ("https", "http"):
        try:
            return fetch(f"{scheme}://{host}{path}", limit), ""
        except Exception as error:
            last = error_name(error)
        if last == "DNS":
            break
    return None, last


def decode(body: bytes) -> str:
    found = CHARSET.search(body[:4096])
    for label in ((found.group(1).decode("ascii", "ignore"),) if found else ()) + (
        "utf-8",
    ):
        try:
            return body.decode(label.replace("_", "-"), errors="replace")
        except (LookupError, UnicodeError, ValueError):
            continue
    return body.decode("utf-8", errors="replace")


def matches(rule: str, target: str) -> bool:
    """A rule matches from the start of the path; * stands for any run of characters and a final $ for the end."""
    anchored = rule.endswith("$")
    pieces = (rule[:-1] if anchored else rule).split("*")
    if not target.startswith(pieces[0]):
        return False
    rest = target[len(pieces[0]) :]
    for at, piece in enumerate(pieces[1:]):
        if anchored and at == len(pieces) - 2:
            return rest.endswith(piece)
        found = rest.find(piece)
        if found < 0:
            return False
        rest = rest[found + len(piece) :]
    return not anchored or not rest


def rules(
    robots: str, token: str = TOKEN
) -> tuple[tuple[list[str], list[str]], float | None]:
    """The Allow and Disallow paths of the group our token follows, else *, and its Crawl-delay."""
    groups, delays, current, previous_was_agent = {}, {}, [], False
    for raw in robots.splitlines():
        line = raw.split("#", 1)[0].strip()
        name, colon, value = line.partition(":")
        if not colon:
            continue
        name, value = name.strip().lower(), value.strip()
        if name == "user-agent":
            if not previous_was_agent:
                current = []
            agent = value.lower()
            groups.setdefault(agent, ([], []))
            current.append(agent)
            previous_was_agent = True
            continue
        previous_was_agent = False
        if name in ("allow", "disallow") and value:
            path = value.rstrip("*") or "*"
            for agent in current:
                groups.setdefault(agent, ([], []))[0 if name == "allow" else 1].append(
                    path
                )
        elif name == "crawl-delay":
            try:
                for agent in current:
                    delays[agent] = float(value)
            except ValueError:
                pass
    agent = token if token in groups else "*"
    return groups.get(agent, ([], [])), delays.get(agent)


def allows(paths: tuple[list[str], list[str]], target: str) -> bool:
    if target == "/robots.txt":
        return True
    allow = max((len(rule) for rule in paths[0] if matches(rule, target)), default=None)
    disallow = max(
        (len(rule) for rule in paths[1] if matches(rule, target)), default=None
    )
    return disallow is None or (allow is not None and allow >= disallow)


def detector():
    try:
        import py3langid
    except ImportError:
        sys.exit(
            "py3langid is missing: run this with the bot's venv, whose model the bot's own check ports"
        )

    def detect(text: str) -> str | None:
        if len(text) < 20:
            return None
        if text.isupper():
            text = text.lower()
        return py3langid.classify(unicodedata.normalize("NFC", text))[0]

    return detect


def read_homepage(body: bytes, detect) -> tuple[int, str, str | None, str | None]:
    """Characters, declared lang, detected language and the problem, as homepage.rs reads a page but stricter: a page that declares another language isn't English."""
    html = decode(body)
    declared = HTML_LANG.search(html[:4000])
    declared = declared.group(1).lower().split("-")[0] if declared else None
    page = " ".join(TAG.sub(" ", SCRIPT_OR_STYLE.sub(" ", html)).split())
    if BOT_WALL.search(page[:SAMPLE]) or CHALLENGE.search(html[:20000]):
        return len(page), declared, None, "challenge"
    if len(page) < MIN_CHARS:
        return len(page), declared, None, "no_text"
    language = detect(page[:LANGUAGE_SAMPLE])
    english = (language == "en" and declared in (None, "en")) or (
        language is None and declared == "en"
    )
    return len(page), declared, language, None if english else "not_english"


def moved(host: str, url: str, own_host: bool) -> str | None:
    """The host an answer moved to, by the bot's rule: a subdomain host stays itself only on its own host."""
    final = (urlsplit(url).hostname or "").lower()
    bare = final[4:] if final.startswith("www.") else final
    if not bare or bare == host or (not own_host and bare.endswith("." + host)):
        return None
    return bare


def listing_kind(body: bytes) -> str:
    head = body[:MAX_LISTING_BYTES].lower()
    if b"<urlset" in head or b"<sitemapindex" in head:
        return f"sitemap:{head.count(b'<loc')}"
    if b"<rss" in head or b"<feed" in head or b"<rdf:rdf" in head:
        return f"feed:{head.count(b'<item') + head.count(b'<entry')}"
    return ""


def check(
    host: str, fetch, detect, own_host: bool = False, pacer: Pacer | None = None
) -> Check:
    found = Check(
        host, checked_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    robots, error = first_answer(host, "/robots.txt", MAX_BYTES, fetch)
    if robots is None:
        found.verdict, found.why = "unreachable", f"robots:{error}"
        return found
    found.robots_status = str(robots.status)
    if target := moved(host, robots.url, own_host):
        found.verdict, found.why = "moved", target
        return found
    if robots.status >= 500 or robots.status == 429:
        found.verdict, found.why = (
            "walled" if robots.status == 429 else "unreachable",
            f"robots_{robots.status}",
        )
        return found
    text = decode(robots.body) if robots.status == 200 else ""
    found.rules, delay = rules(text)
    found.crawl_delay = "" if delay is None else f"{delay:g}"
    if pacer:
        pacer.slow_to(host, delay)
    if not allows(found.rules, "/"):
        found.verdict, found.why = "walled", "robots_disallow"
        return found
    home, error = first_answer(host, "/", MAX_BYTES, fetch)
    if home is None:
        found.verdict, found.why = "unreachable", f"homepage:{error}"
        return found
    found.home_status, found.home_url = str(home.status), home.url
    if target := moved(host, home.url, own_host):
        found.verdict, found.why = "moved", target
        return found
    if home.status in WALL_STATUSES:
        found.verdict, found.why = "walled", f"homepage_{home.status}"
        return found
    if CHALLENGE.search(decode(home.body[:20000])):
        found.verdict, found.why = "walled", "challenge"
        return found
    if home.status >= 500:
        found.verdict, found.why = "unreachable", f"homepage_{home.status}"
        return found
    if home.status != 200:
        found.verdict, found.why = "homepage_error", f"homepage_{home.status}"
        return found
    chars, declared, language, problem = read_homepage(home.body, detect)
    found.chars, found.declared, found.language = (
        str(chars),
        declared or "",
        language or "",
    )
    named = SITEMAP_LINE.findall(text)[:1]
    candidates = named or [path for path in SITEMAP_GUESSES + FEED_GUESSES]
    for url in candidates:
        try:
            url = urljoin(f"https://{host}/", url)
            if not allows(found.rules, urlsplit(url).path or "/"):
                continue
            answer = fetch(url, MAX_LISTING_BYTES)
        except Exception as error:
            found.listing_status = error_name(error)
            continue
        found.listing_status = str(answer.status)
        kind = listing_kind(answer.body) if answer.status == 200 else ""
        if kind:
            found.listing = f"{kind} {url}"
            break
        if answer.status in WALL_STATUSES:
            break
    found.verdict, found.why = (
        ("ok", "")
        if problem is None
        else (
            ("walled", problem) if problem == "challenge" else (problem, language or "")
        )
    )
    return found


def checked(host: str, fetch, detect, own_host: bool, pacer: Pacer) -> Check:
    """One host's check; a failure on our side reads as an error for that host alone."""
    try:
        return check(host, fetch, detect, own_host, pacer)
    except Exception as error:
        return Check(
            host,
            "error",
            type(error).__name__,
            checked_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


def read_hosts(path: str) -> list[str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as source:
        lines = source.read().splitlines()
    if lines and "\t" in lines[0] and lines[0].split("\t")[0] == "host":
        lines = [line.split("\t")[0] for line in lines[1:]]
    seen, hosts = set(), []
    for line in lines:
        host = line.strip().lower()
        if host and not host.startswith("#") and host not in seen:
            seen.add(host)
            hosts.append(host)
    return hosts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        required=True,
        help="hosts to check, one a line, or a TSV whose first column is host",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="the TSV to write: host, verdict (ok, walled, moved, not_english, no_text, homepage_error, unreachable, error), why and what was read",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=MAX_PARALLEL,
        help=f"hosts checked at once, at most {MAX_PARALLEL}",
    )
    parser.add_argument(
        "--own-host",
        action="store_true",
        help="the hosts are subdomain hosts, which stay themselves only on their own host",
    )
    args = parser.parse_args(argv)
    hosts = read_hosts(args.hosts)
    if not hosts:
        sys.exit(f"{args.hosts} lists no hosts")
    detect = detector()
    pacer = Pacer()

    def fetch(url: str, limit: int) -> Answer:
        return http_fetch(url, limit, pacer)

    temporary = Path(f"{args.out}.new")
    with open(temporary, "w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        with ThreadPoolExecutor(
            max_workers=max(1, min(args.parallel, MAX_PARALLEL))
        ) as pool:
            for done, found in enumerate(
                pool.map(
                    lambda host: checked(host, fetch, detect, args.own_host, pacer),
                    hosts,
                ),
                1,
            ):
                writer.writerow(found.row())
                if done % 100 == 0:
                    target.flush()
                    print(f"{done}/{len(hosts)} checked", file=sys.stderr, flush=True)
    temporary.replace(args.out)
    with open(args.out, newline="") as source:
        verdicts = [row["verdict"] for row in csv.DictReader(source, delimiter="\t")]
    print(
        f"{len(verdicts)} hosts checked into {args.out}: "
        + ", ".join(
            f"{verdict} {verdicts.count(verdict)}" for verdict in sorted(set(verdicts))
        )
    )


if __name__ == "__main__":
    main()
