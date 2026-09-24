"""Reject sources a miner could have served itself."""

import asyncio
import ipaddress
import re
from typing import Iterable, Optional
from urllib.parse import urlparse

import bittensor as bt

WILDCARD_DNS_SUFFIXES = (
    "sslip.io",
    "nip.io",
    "xip.io",
    "traefik.me",
    "localtest.me",
    "lvh.me",
    "vcap.me",
    "1u.ms",
)

TUNNEL_SUFFIXES = (
    "ngrok.io",
    "ngrok.app",
    "ngrok.dev",
    "ngrok-free.app",
    "trycloudflare.com",
    "localhost.run",
    "lhr.life",
    "serveo.net",
    "loca.lt",
    "pagekite.me",
    "telebit.io",
    "tunnelto.dev",
    "bore.pub",
    "zrok.io",
    "jprq.io",
    "expose.sh",
)

PLATFORM_SUFFIXES = (
    "vercel.app",
    "netlify.app",
    "netlify.com",
    "herokuapp.com",
    "pages.dev",
    "workers.dev",
    "r2.dev",
    "github.io",
    "gitlab.io",
    "surge.sh",
    "fly.dev",
    "onrender.com",
    "railway.app",
    "koyeb.app",
    "glitch.me",
    "repl.co",
    "replit.app",
    "deno.dev",
    "val.run",
    "cyclic.app",
    "azurewebsites.net",
    "appspot.com",
    "firebaseapp.com",
    "web.app",
    "s3.amazonaws.com",
    "s3-website.amazonaws.com",
    "storage.googleapis.com",
    "blob.core.windows.net",
    "digitaloceanspaces.com",
    "ondigitalocean.app",
)

SHORTENER_HOSTS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "buff.ly",
        "is.gd",
        "rb.gy",
        "cutt.ly",
        "shorturl.at",
        "rebrand.ly",
        "s.id",
        "tiny.cc",
        "shorte.st",
        "t.ly",
    }
)

PRIVATE_TLDS = (
    ".local",
    ".internal",
    ".test",
    ".localhost",
    ".onion",
    ".i2p",
    ".invalid",
)

DNS_TIMEOUT_SECONDS = 2.0

_EMBEDDED_IP = re.compile(
    r"(?:^|[.-])(\d{1,3})[-.](\d{1,3})[-.](\d{1,3})[-.](\d{1,3})(?:[.-]|$)"
)
_ALL_DIGITS = re.compile(r"^\d+$")
_HEX_HOST = re.compile(r"^0x[0-9a-f]+$")

_resolved: dict[str, Optional[str]] = {}


def host_of(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def _embedded_ip(host: str) -> Optional[str]:
    match = _EMBEDDED_IP.search(host)
    if not match:
        return None
    try:
        return str(ipaddress.IPv4Address(".".join(match.groups())))
    except ValueError:
        return None


def _is_ip_literal(host: str) -> bool:
    """Dotted, IPv6, and the decimal/hex spellings a browser also accepts."""
    if _ALL_DIGITS.match(host) or _HEX_HOST.match(host):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _matches(host: str, suffixes: Iterable[str]) -> bool:
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


async def resolve(host: str) -> Optional[str]:
    """First A record for the host, or None; failures resolve to None."""
    if host in _resolved:
        return _resolved[host]

    try:
        infos = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, None, family=2, type=1),
            timeout=DNS_TIMEOUT_SECONDS,
        )
        address = infos[0][4][0] if infos else None
    except Exception:
        address = None

    _resolved[host] = address

    return address


async def rejection_reason(url: str, miner_ip: Optional[str] = None) -> Optional[str]:
    if (url or "").strip().lower().startswith("http://"):
        return "http"

    host = host_of(url)
    if not host:
        return "no-host"
    if _is_ip_literal(host):
        return "ip-host"
    if "." not in host or host.endswith(PRIVATE_TLDS):
        return "not-public"
    if _matches(host, WILDCARD_DNS_SUFFIXES) or _matches(host, TUNNEL_SUFFIXES):
        return "wildcard-dns"
    if _matches(host, PLATFORM_SUFFIXES):
        return "hosting-platform"
    if host in SHORTENER_HOSTS:
        return "url-shortener"

    embedded = _embedded_ip(host)
    if embedded:
        return "self-host" if embedded == miner_ip else "ip-host"

    if miner_ip and await resolve(host) == miner_ip:
        return "self-host"

    return None


async def rejected_links(
    links: Iterable[str], miner_ip: Optional[str] = None
) -> dict[str, str]:
    """Map each unusable link to why it cannot be treated as a source."""
    unique = list(dict.fromkeys(link for link in links if link))
    reasons = await asyncio.gather(
        *[rejection_reason(link, miner_ip) for link in unique]
    )
    rejected = {link: reason for link, reason in zip(unique, reasons) if reason}

    if rejected:
        bt.logging.debug(
            f"[Provenance] rejected {len(rejected)}/{len(unique)} sources: "
            f"{sorted(set(rejected.values()))}"
        )

    return rejected


def miner_ip_of(response) -> Optional[str]:
    return getattr(getattr(response, "axon", None), "ip", None)
