"""Splits the v1 sites with pages to send again into waves for `crawl_list.py --resend-hosts`: benchmark sites, established news, whole sites, then the rest, biggest first."""

import argparse
import csv
import gzip
import json
from pathlib import Path

WAVE_SITES = 15_000
WAVE_PAGES = 50_000_000
# Backing that isn't a benchmark: a site with any other why word is a benchmark site.
BACKING = {
    "academic",
    "public",
    "news",
    "wikidata",
    "docs",
    "news-list",
    "news-major",
    "judge",
    "benchmark",
    "benchmark-extra",
    "evergreen",
}
GROUPS = ("benchmark", "established news", "whole", "the rest")


def group(site: dict) -> int:
    if any(word not in BACKING for word in site.get("why", [])):
        return 0
    if site["rule"] == "news" and not site.get("since"):
        return 1
    return 2 if site["rule"] == "whole" else 3


def waves(
    sites: list[dict],
    pages: dict[str, int],
    most_sites: int = WAVE_SITES,
    most_pages: int = WAVE_PAGES,
) -> list[list[tuple[str, int, int]]]:
    """Waves of (host, pages, group), each at most `most_sites` sites and `most_pages` pages unless one site holds more."""
    taken = sorted(
        (
            (group(site), -pages.get(site["host"], 0), site["host"])
            for site in sites
            if site["rule"] in ("news", "whole") and pages.get(site["host"], 0) > 0
        ),
    )
    out: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    for kind, minus, host in taken:
        count = -minus
        if current and (
            len(current) >= most_sites
            or sum(page for _, page, _ in current) + count > most_pages
        ):
            out.append(current)
            current = []
        current.append((host, count, kind))
    if current:
        out.append(current)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, help="v1-set.json (.gz allowed)")
    parser.add_argument(
        "--pages",
        required=True,
        help="TSV of host and pages: each set site's pages last sent before the cutoff",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="the directory for resend-wave-NN.txt and waves.tsv",
    )
    args = parser.parse_args(argv)
    opener = gzip.open if args.set.endswith(".gz") else open
    with opener(args.set, "rt") as source:
        sites = json.load(source)["sites"]
    with open(args.pages, newline="") as source:
        pages = {
            row["host"]: int(row["pages"])
            for row in csv.DictReader(source, delimiter="\t")
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = waves(sites, pages)
    with open(out / "waves.tsv", "w") as summary:
        summary.write("wave\tsites\tpages\t" + "\t".join(GROUPS) + "\n")
        for number, wave in enumerate(found, 1):
            (out / f"resend-wave-{number:02d}.txt").write_text(
                "".join(f"{host}\n" for host, _, _ in wave)
            )
            kinds = [
                sum(1 for _, _, kind in wave if kind == index)
                for index in range(len(GROUPS))
            ]
            summary.write(
                f"{number}\t{len(wave)}\t{sum(page for _, page, _ in wave)}\t"
                + "\t".join(map(str, kinds))
                + "\n"
            )
    print(f"{len(found)} waves in {out}")


if __name__ == "__main__":
    main()
