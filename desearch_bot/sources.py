"""Public domain lists and category blacklists."""

from __future__ import annotations

import csv
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .suffixes import PublicSuffixList

USER_AGENT = (
    "Mozilla/5.0 (compatible; DesearchBot/1.0; +https://www.desearch.ai/crawler)"
)

# domain column, rank column, has header
RANKED = {
    "tranco": ("https://tranco-list.eu/top-1m.csv.zip", 1, 0, False),
    "majestic": ("https://downloads.majestic.com/majestic_million.csv", 2, 0, True),
    "opr": ("https://www.domcop.com/files/top/top10milliondomains.csv.zip", 1, 0, True),
    "builtwith": ("https://builtwith.com/dl/builtwith-top1m.zip", 1, 0, False),
    "umbrella": (
        "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
        1,
        0,
        False,
    ),
}
# Only Tranco, Majestic and Open PageRank rank by traffic or links; the other two are unordered.
TRAFFIC_RANKED = ("tranco", "majestic", "opr")

UT1_BLACKLISTS = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"
PUBLIC_SUFFIX_LIST = "https://publicsuffix.org/list/public_suffix_list.dat"


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(
        Request(url, headers={"User-Agent": USER_AGENT}), timeout=300
    ) as response:
        with open(dest, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    return dest


def fetch_all(data_dir: Path) -> dict[str, Path]:
    paths = {
        name: download(url, data_dir / f"{name}.bin")
        for name, (url, *_) in RANKED.items()
    }
    paths["ut1"] = download(UT1_BLACKLISTS, data_dir / "ut1.tar.gz")
    paths["psl"] = download(PUBLIC_SUFFIX_LIST, data_dir / "public_suffix_list.dat")
    return paths


def _csv_path(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        extracted = path.with_name(f"{path.stem}.csv")
        if not extracted.exists():
            extracted.write_bytes(archive.read(name))
        return extracted


def read_ranked(
    path: Path,
    psl: PublicSuffixList,
    domain_column: int,
    rank_column: int,
    header: bool,
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    with open(
        _csv_path(path), encoding="utf-8", errors="replace", newline=""
    ) as handle:
        rows = csv.reader(handle)
        if header:
            next(rows, None)
        for row in rows:
            if len(row) <= max(domain_column, rank_column):
                continue
            domain = psl.registrable(row[domain_column].strip('"'))
            if not domain:
                continue
            try:
                rank = int(row[rank_column].strip('"'))
            except ValueError:
                continue
            if rank < ranks.get(domain, 1 << 62):
                ranks[domain] = rank
    return ranks


def read_categories(path: Path, wanted: set[str]) -> dict[str, set[str]]:
    categories: dict[str, set[str]] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/")
            if (
                len(parts) == 3
                and parts[2] == "domains"
                and parts[1] in wanted
                and member.isfile()
            ):
                text = archive.extractfile(member).read().decode("utf-8", "replace")
                categories[parts[1]] = {
                    line.strip().lower()
                    for line in text.splitlines()
                    if line.strip() and not line.startswith("#")
                }
    return categories
