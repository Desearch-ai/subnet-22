"""What a domain is about, and whether that rules it out.

Labels come from the UT1 blacklists and the adult blocklists, both of which are plain files, so
labelling six million domains costs no requests. Cloudflare returns richer labels but its
threat-intelligence quota is 100 calls a month below Enterprise, so it can only enrich the head
of the list; see `radar.py`.

Matching is on the exact host. UT1 lists subdomains such as `<name>.wordpress.com`, so collapsing
its entries to their registrable domain would mislabel the parent.
"""

from __future__ import annotations

from pathlib import Path

from . import adult as adult_lists
from . import sources

# UT1 category -> the label we store. Anything not listed here we do not record.
UT1_LABELS = {
    "adult": "adult",
    "mixed_adult": "mixed_adult",
    "lingerie": "lingerie",
    "dating": "dating",
    "gambling": "gambling",
    "bank": "bank",
    "financial": "finance",
    "bitcoin": "crypto",
    "press": "news",
    "blog": "blog",
    "shopping": "shopping",
    "jobsearch": "jobs",
    "games": "games",
    "sports": "sports",
    "cooking": "food",
    "astrology": "astrology",
    "celebrity": "celebrity",
    "manga": "manga",
    "audio-video": "media",
    "radio": "media",
    "liste_bu": "education",
    "sexual_education": "education",
    "educational_games": "education",
    "ai": "ai",
    "social_networks": "social",
    "forums": "forums",
    "chat": "chat",
    "webmail": "webmail",
    "vpn": "vpn",
    "translation": "translation",
    "download": "download",
    "filehosting": "filehosting",
    "webhosting": "hosting",
    "mobile-phone": "mobile",
    "remote-control": "remote_access",
    "sect": "sect",
    "fakenews": "fakenews",
    "drogue": "drugs",
    "agressif": "violence",
    "dangerous_material": "dangerous",
    "publicite": "ads",
    "marketingware": "marketingware",
    "shortener": "shortener",
    "redirector": "redirector",
    "dynamic-dns": "dynamic_dns",
    "doh": "doh",
    "residential-proxies": "proxy",
    "stalkerware": "stalkerware",
    "hacking": "hacking",
    "warez": "warez",
    "ddos": "ddos",
    "cryptojacking": "cryptojacking",
    "malware": "malware",
    "phishing": "phishing",
    "tricheur": "cheating",
    "dialer": "dialer",
}

# A domain carrying any of these never reaches a miner.
EXCLUDE = frozenset(
    {
        "adult",
        "gambling",
        # Bank portals are login screens with nothing to index, and their intrusion detection
        # treats a robots.txt fetch followed by two sitemap probes as a scan.
        "bank",
        "malware",
        "phishing",
        "cryptojacking",
        "stalkerware",
        "ddos",
        "hacking",
        "warez",
        "dangerous",
        "dialer",
        "cheating",
        "shortener",
        "redirector",
        "ads",
        "marketingware",
        "dynamic_dns",
        "doh",
        "proxy",
        "social",
        "forums",
        "chat",
        "webmail",
        "filehosting",
    }
)

# Preferred when a domain carries several labels, most specific first.
PRIORITY = (
    "adult",
    "gambling",
    "malware",
    "phishing",
    "cryptojacking",
    "stalkerware",
    "warez",
    "hacking",
    "ddos",
    "bank",
    "news",
    "blog",
    "shopping",
    "jobs",
    "games",
    "sports",
    "education",
    "finance",
    "crypto",
    "media",
    "food",
    "dating",
    "social",
    "forums",
)


class Catalogue:
    """Every label we can assign offline, indexed by host."""

    def __init__(self, by_label: dict[str, set[str]]):
        self.by_label = by_label

    @classmethod
    def load(cls, data_dir: Path, refresh: bool = False) -> "Catalogue":
        data_dir = Path(data_dir)
        ut1 = sources.read_categories(data_dir / "ut1.tar.gz", set(UT1_LABELS))
        by_label: dict[str, set[str]] = {}
        for name, hosts in ut1.items():
            by_label.setdefault(UT1_LABELS[name], set()).update(hosts)
        by_label.setdefault("adult", set()).update(
            adult_lists.load(data_dir, refresh=refresh)
        )
        return cls(by_label)

    def labels(self, host: str) -> list[str]:
        found = [label for label, hosts in self.by_label.items() if host in hosts]
        return sorted(found, key=lambda label: (PRIORITY.index(label)
                                                if label in PRIORITY else len(PRIORITY), label))

    def primary(self, labels: list[str]) -> str | None:
        return labels[0] if labels else None

    def excluded(self, labels) -> str | None:
        return next((label for label in labels if label in EXCLUDE), None)


def stats(catalogue: Catalogue) -> dict[str, int]:
    return {label: len(hosts) for label, hosts in sorted(catalogue.by_label.items())}
