"""The abuse blocklist desearch-bot keeps in data/blocklist.json, read with the bot's rules so the feeder hands miners none of it."""

from __future__ import annotations

import ipaddress
import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

BLOCKLIST = (
    Path(__file__).resolve().parents[2] / "desearch-bot" / "data" / "blocklist.json"
)


class Blocklist:
    def __init__(self, raw: dict):
        self.operators = list(raw["operators"])
        self.not_blocked = set(raw["not_blocked"])
        self.domains = list(raw["domains"])
        self.networks = [ipaddress.ip_network(n, strict=False) for n in raw["networks"]]

    def ip(self, address: str) -> bool:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return any(ip.version == net.version and ip in net for net in self.networks)

    def host(self, host: str) -> bool:
        host = host.strip("[]").rstrip(".").lower()
        try:
            return self.ip(host.split("%", 1)[0])
        except ValueError:
            pass

        def under(domain: str) -> bool:
            return host == domain or host.endswith("." + domain)

        if any(under(d) for d in self.domains):
            return True
        if any(under(d) for d in self.not_blocked):
            return False
        return any(
            name in label for label in host.split(".")[:-1] for name in self.operators
        )

    def url(self, url: str) -> bool:
        host = urlsplit(url).hostname
        return bool(host) and self.host(host)


@lru_cache(maxsize=1)
def load(path: str = str(BLOCKLIST)) -> Blocklist:
    return Blocklist(json.loads(Path(path).read_text()))
