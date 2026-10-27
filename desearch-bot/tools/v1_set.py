"""Builds v1-set.json: every v1 host with its rule, which the crawl list builder, the planner and the indexer all read."""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SINCE = "2020-01-01"
# A news site this big with no benchmark citation and off the news list stays out until the judge reads it.
NEWS_GUARD = 1_000_000
JUDGED_MIN = 10
PASSED_SHARE = 0.8
WHOLE_TYPES = {"academic", "public", "docs"}
# A site fits whole when its stored pages fit under three times its judged cap, the crawl's share.
CRAWL_PER_INDEX = 3
FIELDS = ("host", "rule", "since", "paths", "source", "type", "why")
PATH_DATE = re.compile(r"/((?:19|20)[0-9]{2})[/-](0[1-9]|1[0-2])(?:[/-]|$)")


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, newline="")


def tsv(path: str | None) -> list[dict]:
    if not path:
        return []
    with open_text(path) as source:
        return list(csv.DictReader(source, delimiter="\t"))


def bare(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def row_entry(row: dict) -> dict:
    entry = {
        field: row[field] for field in FIELDS if row.get(field) not in (None, "", [])
    }
    entry["host"] = bare(entry["host"])
    return entry


def read_rows(path: str) -> list[dict]:
    with open(path) as source:
        found = json.load(source)
    return [
        row_entry(row) for row in found.get("sites", []) + found.get("dump_sites", [])
    ]


def hand_entries(rows: list[dict]) -> list[dict]:
    entries = []
    for row in rows:
        row = {**row, "why": row.get("why", "").split()}
        entries.append(row_entry(row))
    return entries


def passed(judged: dict | None) -> bool:
    if not judged or int(judged["judged"]) < JUDGED_MIN:
        return False
    return int(judged["substantive"]) / int(judged["judged"]) >= PASSED_SHARE


def listed_entries(
    decisions,
    backing: dict[str, dict],
    judged: dict[str, dict],
    counts: dict[str, dict],
    since: str,
) -> tuple[list[dict], list[dict]]:
    """The judged list's v1 types as set entries, and the sites too big to take whole, which wait for named paths."""
    entries, too_big = [], []
    for row in decisions:
        if row["listed"] != "1":
            continue
        host, site_type = row["host"], row["type"]
        back = backing.get(host, {})
        cited = int(back.get("benchmark_citations") or 0) > 0
        news_list = back.get("news_list") == "1"
        count = counts.get(host, {})
        stored = int(count.get("stored") or 0)
        if site_type == "news":
            if (
                not (cited or news_list)
                and int(count.get("dated_2020_on") or 0) > NEWS_GUARD
            ):
                entries.append({"host": host, "rule": "out", "why": ["size-guard"]})
                continue
            entry = {
                "host": host,
                "rule": "news",
                "since": since,
                "type": "news",
                "why": ["news-list" if news_list else "news"],
            }
        elif site_type in WHOLE_TYPES or (
            site_type == "reference" and passed(judged.get(host))
        ):
            if stored > int(row["cap"]) * CRAWL_PER_INDEX:
                too_big.append(
                    {
                        "host": host,
                        "type": site_type,
                        "stored": stored,
                        "cap": int(row["cap"]),
                    }
                )
                continue
            entry = {
                "host": host,
                "rule": "whole",
                "type": site_type,
                "why": [site_type if site_type != "reference" else "judge"],
            }
        else:
            continue
        if cited:
            entry["why"].append("benchmark")
        if back.get("wikidata_type"):
            entry["why"].append("wikidata")
        entries.append(entry)
    return entries, too_big


def build(
    row_files: list[list[dict]],
    hand: list[dict],
    decisions,
    backing,
    judged,
    counts,
    since: str = SINCE,
) -> tuple[list[dict], list[dict]]:
    listed, too_big = listed_entries(decisions, backing, judged, counts, since)
    sites: dict[str, dict] = {}
    for entry in [row_entry(row) for rows in row_files for row in rows] + hand + listed:
        sites.setdefault(entry["host"], entry)
    return [sites[host] for host in sorted(sites)], too_big


def page_date(url: str, published: str | None) -> date | None:
    if published:
        return date.fromisoformat(published[:10])
    found = PATH_DATE.search(urlsplit(url).path)
    return date(int(found.group(1)), int(found.group(2)), 1) if found else None


def entry_for(sites: dict[str, dict], host: str) -> dict | None:
    # A mobile host (en.m.wikipedia.org) is its site's host.
    host = re.sub(r"^([^.]+)\.m\.", r"\1.", bare(host))
    while host:
        if host in sites:
            return sites[host]
        host = host.partition(".")[2]
    return None


def matches(sites: dict[str, dict], url: str, published: str | None) -> bool:
    """Whether a page belongs to v1: the entry for its host or the nearest listed host above it decides."""
    parts = urlsplit(url)
    entry = entry_for(sites, parts.hostname or "")
    if entry is None or entry["rule"] == "out":
        return False
    if entry["rule"] == "bulk":
        return True
    if entry["rule"] == "whole":
        return not entry.get("paths") or any(
            parts.path.startswith(path) for path in entry["paths"]
        )
    dated = page_date(url, published)
    return dated is not None and (
        not entry.get("since") or dated >= date.fromisoformat(entry["since"])
    )


def summary(sites: list[dict]) -> Counter:
    return Counter((site["rule"], site.get("type", "")) for site in sites)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, help="the set file to write, such as v1-set.json"
    )
    parser.add_argument(
        "--decisions",
        required=True,
        help="the crawl list builder's decisions TSV (.gz allowed)",
    )
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        help="a set rows file (JSON with sites); earlier files win a host",
    )
    parser.add_argument(
        "--hand", help="TSV of host, rule, source, type, why: hosts set by hand"
    )
    parser.add_argument(
        "--backing", help="TSV of host, wikidata_type, benchmark_citations, news_list"
    )
    parser.add_argument("--judged", help="TSV of host, judged, substantive")
    parser.add_argument(
        "--counts",
        help="store counts TSV (.gz allowed) with host, stored and dated_2020_on",
    )
    parser.add_argument(
        "--too-big", help="TSV to write the judged-list sites too big to take whole"
    )
    parser.add_argument(
        "--since", default=SINCE, help="the date floor of the judged list's news sites"
    )
    args = parser.parse_args(argv)
    backing = {r["host"]: r for r in tsv(args.backing)}
    judged = {r["host"]: r for r in tsv(args.judged)}
    counts = {r["host"]: r for r in tsv(args.counts)}
    with open_text(args.decisions) as source:
        sites, too_big = build(
            [read_rows(path) for path in args.rows],
            hand_entries(tsv(args.hand)),
            csv.DictReader(source, delimiter="\t"),
            backing,
            judged,
            counts,
            args.since,
        )
    if not sites:
        sys.exit(f"no sites; {args.out} left as it was")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    temporary = Path(f"{args.out}.new")
    temporary.write_text(
        json.dumps({"built": built, "sites": sites}, separators=(",", ":"))
    )
    temporary.replace(args.out)
    if args.too_big:
        with open(args.too_big, "w", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["host", "type", "stored", "cap"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(sorted(too_big, key=lambda site: -site["stored"]))
    print(
        f"{len(sites)} sites in {args.out}; {len(too_big)} judged-list sites too big to take whole"
    )
    for (rule, site_type), count in sorted(summary(sites).items()):
        print(f"{count:>8}  {rule:6} {site_type}")


if __name__ == "__main__":
    main()
