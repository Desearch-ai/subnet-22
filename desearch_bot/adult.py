"""Adult domains, from public blocklists.

Four lists are merged because each misses domains the others catch, and none of them flags a
mainstream site: measured against 98,309 qualified domains they agreed on 576 and disagreed on
278, and none flagged any of a 35-site control set. The union is used rather than a majority vote
because dropping a good domain costs one row out of a hundred thousand, while keeping an adult one
costs credibility.

The `blhk` list ships its own allowlist of domains it knows it over-blocks (a university, a
mainstream magazine); that is subtracted.
"""

from __future__ import annotations

import re
from pathlib import Path

from .sources import download

# name -> (url, format)
LISTS = {
    "bon_appetit": (
        "https://raw.githubusercontent.com/blhk0532/adult-domains/main/"
        "block.af28d2e460.egz2f0.txt",
        "plain",
    ),
    "oisd": ("https://nsfw.oisd.nl/domainswild", "wildcard"),
    "hagezi": (
        "https://raw.githubusercontent.com/hagezi/dns-blocklists/main/adblock/nsfw.txt",
        "adblock",
    ),
    "sinfonietta": (
        "https://raw.githubusercontent.com/Sinfonietta/hostfiles/master/pornography-hosts",
        "hosts",
    ),
}
ALLOWLIST = (
    "https://raw.githubusercontent.com/blhk0532/adult-domains/main/"
    "allow.61f9c283e3.mk6jrl.txt",
    "plain",
)

ADBLOCK_RULE = re.compile(r"\|\|([^\^/]+)\^")


def _parse(path: Path, fmt: str) -> set[str]:
    domains: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line[0] in "#![":
                continue
            if fmt == "hosts":
                parts = line.split()
                if len(parts) < 2:
                    continue
                domain = parts[1]
            elif fmt == "adblock":
                match = ADBLOCK_RULE.match(line)
                if not match:
                    continue
                domain = match.group(1)
            elif fmt == "wildcard":
                domain = line.lstrip("*.")
            else:
                domain = line
            domain = domain.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if "." in domain and " " not in domain:
                domains.add(domain)
    return domains


def load(data_dir: Path, refresh: bool = True) -> set[str]:
    """Download (or reuse) the lists and return the merged set of adult domains."""
    data_dir = Path(data_dir)
    blocked: set[str] = set()
    for name, (url, fmt) in LISTS.items():
        path = data_dir / f"adult_{name}.txt"
        if refresh or not path.exists():
            download(url, path)
        blocked |= _parse(path, fmt)

    allow_url, allow_fmt = ALLOWLIST
    allow_path = data_dir / "adult_allow.txt"
    if refresh or not allow_path.exists():
        download(allow_url, allow_path)
    return blocked - _parse(allow_path, allow_fmt)
