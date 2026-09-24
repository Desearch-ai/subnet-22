"""Registrable-domain extraction and TLD grouping."""

from __future__ import annotations

from pathlib import Path

ENGLISH_MARKET = {
    "uk",
    "co.uk",
    "org.uk",
    "ac.uk",
    "me.uk",
    "gov.uk",
    "ie",
    "au",
    "com.au",
    "net.au",
    "org.au",
    "edu.au",
    "gov.au",
    "nz",
    "co.nz",
    "org.nz",
    "net.nz",
    "ac.nz",
    "govt.nz",
    "ca",
    "gc.ca",
    "za",
    "co.za",
    "org.za",
    "ac.za",
    "in",
    "co.in",
    "ac.in",
    "gov.in",
    "nic.in",
    "sg",
    "com.sg",
    "edu.sg",
    "gov.sg",
    "ph",
    "com.ph",
    "gov.ph",
    "ng",
    "com.ng",
    "gov.ng",
    "ke",
    "co.ke",
    "gh",
    "com.gh",
    "pk",
    "com.pk",
    "edu.pk",
    "lk",
    "bd",
    "com.bd",
    "my",
    "com.my",
    "edu.my",
    "hk",
    "com.hk",
    "edu.hk",
    "jm",
    "tt",
    "bb",
    "bs",
    "mt",
    "com.mt",
    "cy",
    "com.cy",
    "ug",
    "co.ug",
    "tz",
    "co.tz",
    "zw",
    "co.zw",
    "zm",
    "co.zm",
    "bw",
    "co.bw",
    "na",
    "com.na",
    "mu",
    "fj",
    "com.fj",
    "pg",
}

GENERIC = {
    "com",
    "org",
    "net",
    "info",
    "biz",
    "io",
    "ai",
    "co",
    "dev",
    "app",
    "tech",
    "xyz",
    "online",
    "site",
    "cloud",
    "digital",
    "news",
    "blog",
    "me",
    "tv",
    "cc",
    "us",
    "media",
    "world",
    "life",
    "today",
    "space",
    "store",
    "shop",
    "agency",
    "studio",
    "solutions",
    "services",
    "systems",
    "network",
    "group",
    "team",
    "works",
    "wiki",
    "guide",
    "review",
    "zone",
    "live",
    "press",
    "software",
    "design",
    "email",
    "expert",
    "consulting",
    "capital",
    "finance",
    "fund",
    "health",
    "care",
    "legal",
    "law",
    "academy",
    "school",
    "education",
    "institute",
    "foundation",
    "ngo",
    "int",
    "gov",
    "edu",
    "mil",
    "pro",
    "name",
    "mobi",
    "asia",
    "global",
    "one",
    "page",
    "link",
    "fyi",
    "plus",
    "pub",
    "run",
    "sh",
    "so",
    "gg",
    "st",
    "is",
    "im",
    "eu",
    "art",
    "photography",
    "travel",
    "science",
    "engineering",
    "energy",
    "money",
    "market",
    "company",
    "center",
    "community",
    "events",
    "games",
    "tools",
    "video",
    "audio",
    "training",
    "coach",
    "fitness",
    "family",
    "house",
    "garden",
    "kitchen",
    "coffee",
    "bar",
    "restaurant",
    "cafe",
    "city",
    "earth",
    "club",
    "social",
    "codes",
    "computer",
    "vip",
    "best",
    "cool",
    "fun",
}

BIG_GENERIC = {"com", "net", "org"}


class PublicSuffixList:
    def __init__(self, path: Path):
        self.rules: set[str] = set()
        self.exceptions: set[str] = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("!"):
                self.exceptions.add(line[1:])
            else:
                self.rules.add(line)

    def registrable(self, host: str) -> str | None:
        host = host.lower().strip().strip(".")
        if not host or any(c in host for c in " /:@"):
            return None
        labels = host.split(".")
        if len(labels) < 2 or not all(labels):
            return None
        for i in range(len(labels)):
            candidate = ".".join(labels[i:])
            if candidate in self.exceptions:
                return candidate
            if (
                candidate in self.rules
                or f"*.{'.'.join(labels[i + 1 :])}" in self.rules
            ):
                return ".".join(labels[i - 1 :]) if i else None
        return ".".join(labels[-2:])


def suffix(domain: str) -> str:
    return domain.split(".", 1)[1] if "." in domain else ""


def tld_group(domain: str) -> str:
    s = suffix(domain)
    if s in BIG_GENERIC:
        return "big_generic"
    if s in ENGLISH_MARKET:
        return "en_cctld"
    if s in GENERIC:
        return "new_generic"
    return "other_cctld"
