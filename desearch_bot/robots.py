"""What a site's robots.txt allows us to fetch, and which sitemaps it names."""

from __future__ import annotations

import math
import re

TOKEN = "DesearchBot"
SITEMAP = re.compile(r"(?im)^\s*sitemap\s*:\s*(\S+)")


def rules(text: str, token: str = TOKEN) -> tuple[bool, float | None]:
    """Whether our token may fetch the site's root, and any crawl delay it is asked to keep."""
    groups: dict[str, list[tuple[str, str]]] = {}
    delays: dict[str, float] = {}
    current: list[str] = []
    previous_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, _, value = (part.strip() for part in line.partition(":"))
        name = name.lower()
        if name == "user-agent":
            if not previous_was_agent:
                current = []
            current.append(value.lower())
            groups.setdefault(value.lower(), [])
            previous_was_agent = True
            continue
        previous_was_agent = False
        if name in ("allow", "disallow"):
            for agent in current:
                groups.setdefault(agent, []).append((name, value))
        elif name == "crawl-delay":
            delay = _seconds(value)
            if delay is not None:
                for agent in current:
                    delays[agent] = delay

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
    return best_rule != "disallow", delays.get(agent)


def sitemaps(text: str) -> list[str]:
    return SITEMAP.findall(text)


def _seconds(value: str) -> float | None:
    try:
        delay = float(value)
    except ValueError:
        return None
    return delay if math.isfinite(delay) and delay >= 0 else None
