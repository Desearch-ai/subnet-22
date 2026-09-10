"""The crawler's HTTP session, and the rule that keeps it off private networks."""

from __future__ import annotations

import ipaddress
import socket

import aiohttp
from aiohttp.resolver import AsyncResolver

RESOLVER = "127.0.0.1"
RESOLVER_PORT = 5335


def public_host(host: str) -> bool:
    """False for an address literal inside a private network; names are checked on resolving."""
    try:
        return ipaddress.ip_address(host.strip("[]")).is_global
    except ValueError:
        return True


def public_only(host: str, answers: list[dict]) -> list[dict]:
    """Drop answers that point into a private network, so no sitemap can aim us at one."""
    public = [
        answer for answer in answers if ipaddress.ip_address(answer["host"]).is_global
    ]
    if not public:
        raise OSError(f"{host} resolves only to non-public addresses")
    return public


class PublicResolver(AsyncResolver):
    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[dict]:
        return public_only(host, await super().resolve(host, port, family))


def session(concurrency: int) -> aiohttp.ClientSession:
    """One session for the loop; the local resolver does the DNS caching, so aiohttp does not."""
    connector = aiohttp.TCPConnector(
        limit=concurrency * 2,
        limit_per_host=4,
        use_dns_cache=False,
        resolver=PublicResolver(
            nameservers=[RESOLVER], udp_port=RESOLVER_PORT, tcp_port=RESOLVER_PORT
        ),
    )
    return aiohttp.ClientSession(connector=connector)
