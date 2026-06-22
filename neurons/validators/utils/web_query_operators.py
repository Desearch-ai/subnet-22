import re
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse

_SITE_RE = re.compile(r"(?i)\bsite:(\S+)")


@dataclass
class WebQueryOperators:
    """Operators parsed out of a raw web query, plus the query text with them stripped."""

    text: str
    sites: List[str] = field(default_factory=list)

    def host_allowed(self, url: str) -> bool:
        """True when no site: filter applies, or the URL host is one of the sites (or a subdomain)."""
        if not self.sites:
            return True

        host = (urlparse(url).hostname or "").lower().rstrip(".")
        if not host:
            return False

        return any(host == site or host.endswith("." + site) for site in self.sites)


def _normalize_domain(raw: str) -> str:
    domain = raw.strip().strip("/").lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/", 1)[0]
    return domain.rstrip(".")


def normalize_domains(raw: List[str]) -> List[str]:
    seen = []
    for item in raw or []:
        domain = _normalize_domain(item)
        if domain and domain not in seen:
            seen.append(domain)
    return seen


def host_in_domains(url: str, domains: List[str]) -> bool:
    if not domains:
        return False

    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False

    return any(host == domain or host.endswith("." + domain) for domain in domains)


def parse_web_query(query: str) -> WebQueryOperators:
    """Extract supported operators (only ``site:`` today) and return the cleaned query text."""
    if not query:
        return WebQueryOperators(text="")

    sites = [
        domain for raw in _SITE_RE.findall(query) if (domain := _normalize_domain(raw))
    ]

    stripped = re.sub(r"\s+", " ", _SITE_RE.sub("", query)).strip()

    return WebQueryOperators(text=stripped, sites=sites)
