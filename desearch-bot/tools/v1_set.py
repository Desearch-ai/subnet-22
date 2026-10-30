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
# Option A takes an unbacked news site's last year only, from the first of the month, so the floor moves once a month.
RECENT_DAYS = 365
# More pages dated in a year than an unbacked outlet writes: a generated or re-stamped sitemap.
NEWS_YEAR_MAX = 100_000
JUDGED_MIN = 10
PASSED_SHARE = 0.8
FAILED_SHARE = 0.4
WHOLE_TYPES = {"academic", "public", "docs"}
# A site fits whole when its stored pages fit under three times its judged cap, the crawl's share.
CRAWL_PER_INDEX = 3
FIELDS = ("host", "rule", "since", "paths", "source", "type", "why")
# At each run of digits, left to right: YYYY-MM-DD, then YYYYMMDD, each standing apart from other digits.
DIGITS_DATE = re.compile(r"(?<![0-9])(?:([0-9]{4})-([0-9]{2})-([0-9]{2})(?![0-9])|([0-9]{4})([0-9]{2})([0-9]{2})(?![0-9]))?")
# Forecast sites are one template per place, so they are big, backed or not; a small weather blog is not one.
WEATHER = re.compile(r"weather|forecast")
WEATHER_TYPES = {"news", "reference"}
WEATHER_MIN = 50_000
PEOPLE_SEARCH = re.compile(r"peoplesearch|people-search|peoplefinder|personsearch|whitepages|truepeople|fastpeople|electoralroll|backgroundcheck|publicrecord|salaries|sponsordata|callsignlookup")
# Unrelated sites that store exactly as many URLs come from one generator; small or round counts collide by chance.
NETWORK_MIN = 1_000_000
NETWORK_MIN_THREE = 500_000
DOCS_HOST = re.compile(r"^(docs|developer|developers|learn)\.")
# A site that moved passes its backing and rule along a chain of redirects this long at most.
MAX_HOPS = 5
VERDICTS = {"keep", "out"}
# What an `out` row's `why` may say; a row giving only benchmark names takes its reason from its note.
OUT_REASONS = {
    "mirror", "benchmark-copy", "people-search", "template", "spam", "aggregator", "judge", "terms", "not-english",
    "simple-english-copy", "dictionary", "data-not-pages", "project-infrastructure", "taxonomy-stubs", "media-files",
    "piracy", "no-dump", "shop", "platform", "too-big", "low-value",
}
NOTE_REASONS = (
    ("terms", "terms"), ("web archive", "mirror"), ("same database", "mirror"), ("private people", "people-search"),
    ("registration records", "people-search"), ("no dump", "no-dump"), ("only part worth", "too-big"),
    ("shop", "shop"), ("stock photos", "shop"), ("exchange", "shop"), ("platform", "platform"),
    ("presentations", "platform"), ("generated", "template"), ("listings", "template"), ("catalog", "template"),
    ("statistics", "template"), ("profiles", "template"),
)


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


def out_reason(why: list[str], note: str) -> list[str]:
    reasons = [word for word in why if word in OUT_REASONS]
    if reasons:
        return reasons
    note = note.lower()
    return [next((reason for words, reason in NOTE_REASONS if words in note), "low-value")]


def row_entry(row: dict) -> dict:
    entry = {field: row[field] for field in FIELDS if row.get(field) not in (None, "", [])}
    entry["host"] = bare(entry["host"])
    if DOCS_HOST.match(entry["host"]) and entry.get("type") in ("company", "organization", "reference"):
        entry["type"] = "docs"
    if entry["rule"] == "out":
        entry = {"host": entry["host"], "rule": "out", "why": out_reason(entry.get("why", []), row.get("note", ""))}
    return entry


def read_rows(path: str) -> list[dict]:
    with open(path) as source:
        found = json.load(source)
    return [row_entry(row) for row in found.get("sites", []) + found.get("dump_sites", [])]


def hand_entries(rows: list[dict]) -> list[dict]:
    return [row_entry({**row, "why": row.get("why", "").split(), "paths": (row.get("paths") or "").split()}) for row in rows]


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


def outlet(host: str, outlets: dict[str, str]) -> str | None:
    """The outlet list's word for a host: its own entry, a parent's (africa.espn.com is ESPN), or a news. subdomain's."""
    if host in outlets:
        return outlets[host]
    labels = host.split(".")
    for at in range(1, len(labels) - 1):
        parent = ".".join(labels[at:])
        if parent in outlets:
            return outlets[parent]
    return outlets.get(f"news.{host}")


def backed_news(host: str, backing: dict[str, dict], news_sources: dict[str, str], outlets: dict[str, str], moved_from: dict[str, list[str]] | None = None) -> list[str]:
    """What makes a news site an established outlet: Wikidata, the 897 list, the major-news list or a name added by hand."""
    back, sources = backing.get(host, {}), news_sources.get(host, "")
    why = []
    if "wikidata" in sources or back.get("wikidata_type") == "News":
        why.append("wikidata")
    if "list897" in sources or back.get("news_list") == "1":
        why.append("news-list")
    listed = outlet(host, outlets)
    if listed and listed not in why:
        why.append(listed)
    # A site that moved keeps its backing on the host it moved to.
    for source in [] if why else (moved_from or {}).get(host, []):
        if why := backed_news(source, backing, news_sources, outlets):
            break
    return why


def news_since(backed: bool, news: str, recent: str) -> str | None:
    if news == "B":
        return SINCE_B
    return None if backed else recent


def listed_entries(decisions, backing, news_sources, outlets, judged, counts, news_counts, news: str, recent: str, moved_from=None) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """The judged list's v1 types as set entries, the sites too big to take whole, which wait for named paths, and the decisions of the listed hosts other sites moved to."""
    spam = networks(counts)
    moved_from = moved_from or {}
    entries, too_big, targets = [], [], {}
    for row in decisions:
        if row["listed"] != "1":
            continue
        host, site_type = row["host"], row["type"]
        if host in moved_from:
            targets[host] = row
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
        if site_type in WEATHER_TYPES and WEATHER.search(host) and stored >= WEATHER_MIN:
            entries.append({"host": host, "rule": "out", "why": ["template"]})
            continue
        if site_type == "news":
            why = backed_news(host, backing, news_sources, outlets, moved_from)
            if not why and verdict is not None and verdict <= FAILED_SHARE:
                entries.append({"host": host, "rule": "out", "why": ["judge"]})
                continue
            if not why and int(news_counts.get(host, {}).get("dated_last_year") or 0) > NEWS_YEAR_MAX:
                entries.append({"host": host, "rule": "out", "why": ["spam"]})
                continue
            entry = {"host": host, "rule": "news", "type": "news", "why": why or ["news"]}
            if since := news_since(bool(why), news, recent):
                entry["since"] = since
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
    return entries, too_big, targets


def redirects_of(decisions) -> dict[str, str]:
    """Hosts the crawl list holds out only because they redirect, and the registry's host for where they went."""
    return {row["host"]: bare(row["canonical"]) for row in decisions if row["state"] == "redirects" and row["reason"] == "state:redirects" and row.get("canonical")}


def moved_to(host: str, redirects: dict[str, str]) -> str | None:
    """Where a chain of moves ends; none when it loops or runs past MAX_HOPS."""
    target = host
    for _ in range(MAX_HOPS):
        if target not in redirects:
            break
        target = redirects[target]
    return None if target == host or target in redirects else target


def related(host: str, other: str) -> bool:
    return host == other or host.endswith("." + other) or other.endswith("." + host)


def moves(redirects: dict[str, str]) -> dict[str, str]:
    """Each moved host's final host, unless it is the host's own parent or subdomain: a parent stands for every site under it."""
    found = {}
    for host in redirects:
        target = moved_to(host, redirects)
        if target and not related(host, target):
            found[host] = target
    return found


def carried(first: dict[str, dict], moved: dict[str, str], targets: dict[str, dict], counts: dict[str, dict]) -> list[dict]:
    """A news or whole site that moved takes its rule to its new host when the crawl list takes that host on its own with a type the rule fits, whole only when it fits."""
    fits = {"news": {"news"}, "whole": WHOLE_TYPES | {"reference"}}
    found = []
    for source, target in sorted(moved.items()):
        entry, row = first.get(source), targets.get(target)
        if not entry or not row or entry["rule"] not in fits or row["type"] not in fits[entry["rule"]] or entry.get("paths"):
            continue
        if entry["rule"] == "whole" and int(counts.get(target, {}).get("stored") or 0) > int(row["cap"]) * CRAWL_PER_INDEX:
            continue
        found.append({**{k: v for k, v in entry.items() if k != "since"}, "host": target, "why": entry.get("why", []) + ["redirect"]})
    return found


def read_verdicts(path: str | None) -> dict[str, str]:
    """The site check's TSV of host and verdict, keep or out."""
    verdicts = {}
    for row in tsv(path):
        verdict = (row.get("verdict") or "").strip().lower()
        if verdict not in VERDICTS:
            sys.exit(f"{path}: {row.get('host')} has verdict {verdict!r}, not keep or out; nothing written")
        verdicts[bare(row["host"])] = verdict
    return verdicts


def build(row_files, hand, decisions, backing, news_sources, judged, counts, news_counts=None, news: str = "A", today: date | None = None, outlets=None, verdicts=None, redirects=None) -> tuple[list[dict], list[dict]]:
    """Hand entries first (the lead's outs), then the site check's outs, the teammates' rows in order, the judged list, and the rules of sites that moved; the first to name a host wins."""
    outlets = outlets or {}
    recent = ((today or datetime.now(timezone.utc).date()) - timedelta(days=RECENT_DAYS)).replace(day=1).isoformat()
    if redirects is None:
        decisions = list(decisions)
        redirects = redirects_of(decisions)
    moved = moves(redirects)
    moved_from = defaultdict(list)
    for source, target in sorted(moved.items()):
        moved_from[target].append(source)
    listed, too_big, targets = listed_entries(decisions, backing, news_sources, outlets, judged, counts, news_counts or {}, news, recent, moved_from)
    rows = []
    # Every news site takes its window from the same outlet test, the teammates' rows too.
    for entry in (row_entry(row) for row_file in row_files for row in row_file):
        if entry["rule"] == "news":
            entry.pop("since", None)
            if since := news_since(bool(backed_news(entry["host"], backing, news_sources, outlets, moved_from)), news, recent):
                entry["since"] = since
        rows.append(entry)
    quality = [{"host": host, "rule": "out", "why": ["quality"]} for host, verdict in sorted((verdicts or {}).items()) if verdict == "out"]
    sites: dict[str, dict] = {}
    for entry in hand + quality + rows:
        sites.setdefault(entry["host"], entry)
    for entry in carried(sites, moved, targets, counts):
        if entry["rule"] == "news" and (since := news_since(bool(backed_news(entry["host"], backing, news_sources, outlets, moved_from)), news, recent)):
            entry["since"] = since
        listed.append(entry)
    for entry in listed:
        sites.setdefault(entry["host"], entry)
    return [sites[host] for host in sorted(sites)], too_big


def real_date(year: str, month: str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def calendar_date(year: str, month: str, day: str) -> date | None:
    """A date in a URL path: 1990-2099 only, so a numeric ID doesn't read as one."""
    found = real_date(year, month, day)
    return found if found and 1990 <= found.year <= 2099 else None


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
        for at in range(len(part)):
            match = DIGITS_DATE.match(part, at)
            groups = match.groups() if match else (None,) * 6
            dashed, compact = groups[:3], groups[3:]
            if (dashed[0] and (found := calendar_date(*dashed))) or (compact[0] and (found := calendar_date(*compact))):
                return found
    return None


def page_date(url: str, published: str | None) -> date | None:
    found = re.match(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", (published or "").strip())
    return (found and real_date(*found.groups())) or path_date(urlsplit(url).path)


def entry_for(sites: dict[str, dict], host: str) -> dict | None:
    # A mobile host (en.m.wikipedia.org) is its site's host.
    host = re.sub(r"^([^.]+)\.m\.", r"\1.", bare(host))
    while host:
        if host in sites:
            return sites[host]
        host = host.partition(".")[2]
    return None


def matches(sites: dict[str, dict], url: str, published: str | None, page_type: str | None = None, undated: str = "out") -> bool:
    """Whether a page belongs to v1: the entry for its host or the nearest listed host above it decides; with `undated` article, an undated news article is in."""
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
    if dated is None:
        return undated == "article" and page_type == "article"
    return (
        not entry.get("since") or dated >= date.fromisoformat(entry["since"])
    )




def summary(sites: list[dict]) -> Counter:
    return Counter((site["rule"], site.get("type", ""), "since" in site) for site in sites)


def uncounted(sites: list[dict], counts: dict[str, dict], news_counts: dict[str, dict]) -> tuple[int, int]:
    """News sites with no news counts and whole sites with no store counts, which the spam and too-big checks read as empty."""
    news = sum(1 for site in sites if site["rule"] == "news" and site["host"] not in news_counts)
    whole = sum(1 for site in sites if site["rule"] == "whole" and site["host"] not in counts)
    return news, whole


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the set file to write, such as v1-set.json")
    parser.add_argument("--decisions", required=True, help="the crawl list builder's decisions TSV (.gz allowed)")
    parser.add_argument("--news", choices=("A", "B"), default="A", help="A: established outlets all years, other news sites their last year; B: every news site from 2020")
    parser.add_argument("--undated", choices=("out", "article"), default="out", help="article: a news page with no date is in when its page_type is article")
    parser.add_argument("--rows", action="append", default=[], help="a set rows file (JSON with sites); earlier files win a host")
    parser.add_argument("--hand", help="TSV of host, rule, type, why, paths, source (why and paths space-separated): the lead's entries, which win over every other source")
    parser.add_argument("--backing", help="TSV of host, wikidata_type, benchmark_citations, news_list")
    parser.add_argument("--news-domains", help="TSV (.gz allowed) of news hosts and their sources (wikidata, list897, jev, ut1)")
    parser.add_argument("--outlets", help="TSV of host and why: news outlets by Wikidata class and official website, the major-news list and names added by hand")
    parser.add_argument("--judged", help="TSV of host, judged, substantive")
    parser.add_argument("--counts", help="store counts TSV (.gz allowed) with host and stored")
    parser.add_argument("--news-counts", help="news store counts TSV (.gz allowed) with host and dated_last_year")
    parser.add_argument("--too-big", help="TSV to write the judged-list sites too big to take whole")
    parser.add_argument("--site-verdicts", help="the site check's TSV (.gz allowed) of host and verdict, keep or out: out puts a host out for quality, over every source but the hand entries")
    args = parser.parse_args(argv)
    backing = {r["host"]: r for r in tsv(args.backing)}
    news_sources = {r["host"]: r["sources"] for r in tsv(args.news_domains)}
    judged = {r["host"]: r for r in tsv(args.judged)}
    counts = {r["host"]: r for r in tsv(args.counts)}
    news_counts = {r["host"]: r for r in tsv(args.news_counts)}
    verdicts = read_verdicts(args.site_verdicts)
    outlets = {}
    for row in tsv(args.outlets):
        outlets.setdefault(row["host"], row["why"])
    with open_text(args.decisions) as source:
        redirects = redirects_of(csv.DictReader(source, delimiter="\t"))
    with open_text(args.decisions) as source:
        decisions = csv.DictReader(source, delimiter="\t")
        sites, too_big = build([read_rows(path) for path in args.rows], hand_entries(tsv(args.hand)), decisions, backing, news_sources, judged, counts, news_counts, args.news, outlets=outlets, verdicts=verdicts, redirects=redirects)
    if not sites:
        sys.exit(f"no sites; {args.out} left as it was")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    temporary = Path(f"{args.out}.new")
    temporary.write_text(json.dumps({"built": built, "news": args.news, "undated": args.undated, "sites": sites}, separators=(",", ":")))
    temporary.replace(args.out)
    if args.too_big:
        with open(args.too_big, "w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["host", "type", "stored", "cap"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(too_big, key=lambda site: -site["stored"]))
    print(f"{len(sites)} sites in {args.out} (news {args.news}); {len(too_big)} judged-list sites too big to take whole")
    quality = sum(1 for site in sites if site.get("why") == ["quality"])
    moved = sum(1 for site in sites if "redirect" in site.get("why", []))
    print(f"{quality} out by the site check ({sum(v == 'out' for v in verdicts.values())} out verdicts, {sum(v == 'keep' for v in verdicts.values())} keep); {moved} taken from sites that moved ({len(redirects)} redirects in the registry)")
    news_missing, whole_missing = uncounted(sites, counts, news_counts)
    print(f"{news_missing} news sites without news counts and {whole_missing} whole sites without store counts: the spam and too-big checks read them as empty")
    for (rule, site_type, dated), count in sorted(summary(sites).items()):
        print(f"{count:>8}  {rule:6} {site_type:13} {'since' if dated else ''}")


if __name__ == "__main__":
    main()
