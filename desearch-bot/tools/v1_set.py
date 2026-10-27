"""Builds v1-set.json: every v1 host with its rule, which the crawl list builder, the planner and the indexer all read."""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

SINCE_B = "2020-01-01"
# Option A takes an unbacked news site's last year only.
RECENT_DAYS = 365
JUDGED_MIN = 10
PASSED_SHARE = 0.8
FAILED_SHARE = 0.4
WHOLE_TYPES = {"academic", "public", "docs"}
# A site fits whole when its stored pages fit under three times its judged cap, the crawl's share.
CRAWL_PER_INDEX = 3
FIELDS = ("host", "rule", "since", "paths", "source", "type", "why")
DASHED = re.compile(r"(?<![0-9])([0-9]{4})-([0-9]{2})-([0-9]{2})(?![0-9])")
COMPACT = re.compile(r"(?<![0-9])([0-9]{4})([0-9]{2})([0-9]{2})(?![0-9])")
# Forecast sites are one template per place, so they are big; a small weather blog or a backed outlet is not one.
WEATHER = re.compile(r"weather|forecast")
WEATHER_TYPES = {"news", "reference"}
WEATHER_MIN = 50_000
PEOPLE_SEARCH = re.compile(r"peoplesearch|people-search|peoplefinder|personsearch|whitepages|truepeople|fastpeople|electoralroll|backgroundcheck|publicrecord|salaries|sponsordata|callsignlookup")
# Unrelated sites that store exactly as many URLs come from one generator; small or round counts collide by chance.
NETWORK_MIN = 1_000_000
NETWORK_MIN_THREE = 500_000
DOCS_HOST = re.compile(r"^(docs|developer|developers|learn)\.")


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
    entry = {field: row[field] for field in FIELDS if row.get(field) not in (None, "", [])}
    entry["host"] = bare(entry["host"])
    if DOCS_HOST.match(entry["host"]) and entry.get("type") in ("company", "organization", "reference"):
        entry["type"] = "docs"
    return entry


def read_rows(path: str) -> list[dict]:
    with open(path) as source:
        found = json.load(source)
    return [row_entry(row) for row in found.get("sites", []) + found.get("dump_sites", [])]


def hand_entries(rows: list[dict]) -> list[dict]:
    return [row_entry({**row, "why": row.get("why", "").split()}) for row in rows]


def share(judged: dict | None) -> float | None:
    if not judged or int(judged["judged"]) < JUDGED_MIN:
        return None
    return int(judged["substantive"]) / int(judged["judged"])


def networks(counts: dict[str, dict]) -> set[str]:
    """Sites in a spam network: an exact stored count shared with an unrelated site, large and not round."""
    by_count = defaultdict(list)
    for host, count in counts.items():
        stored = int(count.get("stored") or 0)
        if stored >= NETWORK_MIN_THREE and stored % 1000:
            by_count[stored].append(host)
    found = set()
    for stored, hosts in by_count.items():
        brands = {host.split(".")[0] for host in hosts}
        if len(brands) >= 3 or (len(brands) >= 2 and stored >= NETWORK_MIN):
            found.update(hosts)
    return found


def backed_news(host: str, backing: dict[str, dict], news_sources: dict[str, str]) -> list[str]:
    """What backs a news site as an established outlet: Wikidata, the 897 list or a benchmark citation."""
    back, sources = backing.get(host, {}), news_sources.get(host, "")
    why = []
    if "wikidata" in sources or back.get("wikidata_type") == "News":
        why.append("wikidata")
    if "list897" in sources or back.get("news_list") == "1":
        why.append("news-list")
    if int(back.get("benchmark_citations") or 0) > 0:
        why.append("benchmark")
    return why


def listed_entries(decisions, backing, news_sources, judged, counts, news: str, today: date) -> tuple[list[dict], list[dict]]:
    """The judged list's v1 types as set entries, and the sites too big to take whole, which wait for named paths."""
    recent = (today - timedelta(days=RECENT_DAYS)).isoformat()
    spam = networks(counts)
    entries, too_big = [], []
    for row in decisions:
        if row["listed"] != "1":
            continue
        host, site_type = row["host"], row["type"]
        if site_type not in WHOLE_TYPES | {"news", "reference"}:
            continue
        verdict = share(judged.get(host))
        if host in spam:
            entries.append({"host": host, "rule": "out", "why": ["spam"]})
            continue
        if site_type != "news" and PEOPLE_SEARCH.search(host):
            entries.append({"host": host, "rule": "out", "why": ["people-search"]})
            continue
        stored = int(counts.get(host, {}).get("stored") or 0)
        if site_type in WEATHER_TYPES and WEATHER.search(host) and stored >= WEATHER_MIN and not backed_news(host, backing, news_sources):
            entries.append({"host": host, "rule": "out", "why": ["template"]})
            continue
        if site_type == "news":
            why = backed_news(host, backing, news_sources)
            if not why and verdict is not None and verdict <= FAILED_SHARE:
                entries.append({"host": host, "rule": "out", "why": ["judge"]})
                continue
            entry = {"host": host, "rule": "news", "type": "news", "why": why or ["news"]}
            if news == "B":
                entry["since"] = SINCE_B
            elif not why:
                entry["since"] = recent
        elif site_type in WHOLE_TYPES or (verdict is not None and verdict >= PASSED_SHARE):
            if stored > int(row["cap"]) * CRAWL_PER_INDEX:
                too_big.append({"host": host, "type": site_type, "stored": stored, "cap": int(row["cap"])})
                continue
            entry = {"host": host, "rule": "whole", "type": site_type, "why": [site_type if site_type != "reference" else "judge"]}
            if backing.get(host, {}).get("wikidata_type"):
                entry["why"].append("wikidata")
        else:
            continue
        entries.append(entry)
    return entries, too_big


def build(row_files, hand, decisions, backing, news_sources, judged, counts, news: str = "A", today: date | None = None) -> tuple[list[dict], list[dict]]:
    """Hand entries first (the lead's outs), then the teammates' rows in order, then the judged list; the first to name a host wins."""
    listed, too_big = listed_entries(decisions, backing, news_sources, judged, counts, news, today or datetime.now(timezone.utc).date())
    sites: dict[str, dict] = {}
    for entry in hand + [row_entry(row) for rows in row_files for row in rows] + listed:
        sites.setdefault(entry["host"], entry)
    return [sites[host] for host in sorted(sites)], too_big


def calendar_date(year: str, month: str, day: str) -> date | None:
    try:
        found = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return found if 1990 <= found.year <= 2099 else None


def path_date(path: str) -> date | None:
    """`/YYYY/MM/DD/`, `/YYYY/MM/` (the first of the month), or `YYYY-MM-DD` or `YYYYMMDD` standing apart, as the planner reads them."""
    parts = path.split("/")
    for at, part in enumerate(parts):
        month = parts[at + 1] if at + 1 < len(parts) else ""
        if len(part) == 4 and 1 <= len(month) <= 2:
            day = parts[at + 2] if at + 2 < len(parts) and 1 <= len(parts[at + 2]) <= 2 else "1"
            found = calendar_date(part, month, day) or calendar_date(part, month, "1")
            if found:
                return found
        for pattern in (DASHED, COMPACT):
            for match in pattern.finditer(part):
                if found := calendar_date(*match.groups()):
                    return found
    return None


def page_date(url: str, published: str | None) -> date | None:
    if published and re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", published):
        return date.fromisoformat(published[:10])
    return path_date(urlsplit(url).path)


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
    return Counter((site["rule"], site.get("type", ""), "since" in site) for site in sites)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the set file to write, such as v1-set.json")
    parser.add_argument("--decisions", required=True, help="the crawl list builder's decisions TSV (.gz allowed)")
    parser.add_argument("--news", choices=("A", "B"), default="A", help="A: established outlets all years, other news sites their last year; B: every news site from 2020")
    parser.add_argument("--rows", action="append", default=[], help="a set rows file (JSON with sites); earlier files win a host")
    parser.add_argument("--hand", help="TSV of host, rule, type, why: the lead's entries, which win over every other source")
    parser.add_argument("--backing", help="TSV of host, wikidata_type, benchmark_citations, news_list")
    parser.add_argument("--news-domains", help="TSV (.gz allowed) of news hosts and their sources (wikidata, list897, jev, ut1)")
    parser.add_argument("--judged", help="TSV of host, judged, substantive")
    parser.add_argument("--counts", help="store counts TSV (.gz allowed) with host and stored")
    parser.add_argument("--too-big", help="TSV to write the judged-list sites too big to take whole")
    args = parser.parse_args(argv)
    backing = {r["host"]: r for r in tsv(args.backing)}
    news_sources = {r["host"]: r["sources"] for r in tsv(args.news_domains)}
    judged = {r["host"]: r for r in tsv(args.judged)}
    counts = {r["host"]: r for r in tsv(args.counts)}
    with open_text(args.decisions) as source:
        decisions = csv.DictReader(source, delimiter="\t")
        sites, too_big = build([read_rows(path) for path in args.rows], hand_entries(tsv(args.hand)), decisions, backing, news_sources, judged, counts, args.news)
    if not sites:
        sys.exit(f"no sites; {args.out} left as it was")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    temporary = Path(f"{args.out}.new")
    temporary.write_text(json.dumps({"built": built, "news": args.news, "sites": sites}, separators=(",", ":")))
    temporary.replace(args.out)
    if args.too_big:
        with open(args.too_big, "w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["host", "type", "stored", "cap"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(too_big, key=lambda site: -site["stored"]))
    print(f"{len(sites)} sites in {args.out} (news {args.news}); {len(too_big)} judged-list sites too big to take whole")
    for (rule, site_type, dated), count in sorted(summary(sites).items()):
        print(f"{count:>8}  {rule:6} {site_type:13} {'since' if dated else ''}")


if __name__ == "__main__":
    main()
