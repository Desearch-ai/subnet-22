"""Builds v1-set.json: every v1 host with its rule, which the crawl list builder, the planner and the indexer all read."""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SINCE = "2020-01-01"
# A news site this big with no benchmark citation and off the news list waits for the judge, held at this many pages.
NEWS_GUARD = 1_000_000
EXTRA_MIN_QUESTIONS = 3
JUDGED_MIN = 10
PASSED_SHARE = 0.8
WHOLE_TYPES = {"academic", "public", "docs"}
BENCHMARKS = {
    "Ours": "ours",
    "SQA": "simpleqa",
    "SQAV": "simpleqa-verified",
    "Seal": "sealqa",
    "Fresh": "freshqa",
    "WW": "webwalkerqa",
    "BCP": "browsecomp-plus",
}
EVERGREEN_TYPES = {
    "technical docs": "docs",
    "government and statistics": "public",
    "encyclopedia": "reference",
    "reference database": "reference",
    "dictionary": "reference",
}
TAKE_CAPS = {
    "relevant sections, up to 20,000 pages": 20_000,
    "template database: judge first, up to 1,000,000 pages": 1_000_000,
}
CITED_HOSTS = "cited hosts:"
BULK_FIX = "use a bulk source"
FANDOM = "Fandom wiki XML dumps"
GAP_FIX = "accept the gap"
# A hand row that only names the bulk source of a benchmark site.
SOURCE = "source"


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


def why_benchmarks(unlocks: str) -> list[str]:
    names = []
    for part in unlocks.split(";"):
        label = part.strip().rsplit(" ", 1)[0]
        if label in BENCHMARKS:
            names.append(BENCHMARKS[label])
    return names


def benchmark_type(row: dict) -> str:
    return EVERGREEN_TYPES.get(row["type"], row["type"])


def benchmark_hosts(row: dict) -> list[str]:
    take = row["take"]
    if take.startswith(CITED_HOSTS):
        return [bare(h) for h in take[len(CITED_HOSTS) :].split()]
    return [bare(h) for h in (row["hosts"] or row["site"]).split()]


def chosen(row: dict, extra_min: int) -> bool:
    if row["fix"] == GAP_FIX:
        return False
    if row["tier"] == "benchmark":
        return True
    if row["tier"] == "benchmark (extra)":
        return int(row["questions"] or 0) >= extra_min
    return row["tier"] == "evergreen"


def benchmark_entries(
    rows: list[dict], sources: dict[str, str], extra_min: int
) -> list[dict]:
    entries = []
    for row in rows:
        if not chosen(row, extra_min):
            continue
        if row["tier"] == "evergreen":
            why = ["evergreen"]
        elif row["tier"] == "benchmark (extra)":
            why = ["benchmark-extra"]
        else:
            why = why_benchmarks(row["unlocks"]) or ["benchmark"]
        site_type = benchmark_type(row)
        for host in benchmark_hosts(row):
            if row["fix"] == BULK_FIX:
                source = sources.get(row["site"]) or (
                    "fandom" if FANDOM in row["why"] else row["site"]
                )
                entries.append(
                    {
                        "host": host,
                        "rule": "bulk",
                        "source": source,
                        "type": site_type,
                        "why": why,
                    }
                )
                continue
            entry = {"host": host, "rule": "whole", "type": site_type, "why": why}
            if row["take"] in TAKE_CAPS:
                entry["cap"] = TAKE_CAPS[row["take"]]
            entries.append(entry)
    return entries


def passed(judged: dict | None) -> bool:
    if not judged or int(judged["judged"]) < JUDGED_MIN:
        return False
    return int(judged["substantive"]) / int(judged["judged"]) >= PASSED_SHARE


def listed_entries(
    decisions,
    backing: dict[str, dict],
    judged: dict[str, dict],
    dated: dict[str, int],
    since: str,
) -> list[dict]:
    entries = []
    for row in decisions:
        if row["listed"] != "1":
            continue
        host, site_type = row["host"], row["type"]
        back = backing.get(host, {})
        cited = int(back.get("benchmark_citations") or 0) > 0
        news_list = back.get("news_list") == "1"
        if site_type == "news":
            entry = {
                "host": host,
                "rule": "news",
                "since": since,
                "type": "news",
                "why": ["news-list" if news_list else "news"],
            }
            if not (cited or news_list) and dated.get(host, 0) > NEWS_GUARD:
                entry["cap"] = NEWS_GUARD
                entry["why"].append("guard")
        elif site_type in WHOLE_TYPES:
            entry = {
                "host": host,
                "rule": "whole",
                "cap": int(row["cap"]),
                "type": site_type,
                "why": [site_type],
            }
        elif site_type == "reference" and passed(judged.get(host)):
            entry = {
                "host": host,
                "rule": "whole",
                "cap": int(row["cap"]),
                "type": "reference",
                "why": ["judge"],
            }
        else:
            continue
        if cited:
            entry["why"].append("benchmark")
        if back.get("wikidata_type"):
            entry["why"].append("wikidata")
        entries.append(entry)
    return entries


def hand_entry(row: dict) -> dict:
    entry = {"host": bare(row["host"]), "rule": row["rule"]}
    if row["rule"] == "out":
        return entry
    if row["rule"] == "news":
        entry["since"] = row.get("since") or SINCE
    if row.get("cap"):
        entry["cap"] = int(row["cap"])
    if row["rule"] == "bulk":
        entry["source"] = row["source"]
    entry.update(type=row["type"], why=row["why"].split())
    return entry


def build(
    hand: list[dict],
    benchmark_rows: list[dict],
    decisions,
    backing,
    judged,
    dated,
    since: str = SINCE,
    extra_min: int = EXTRA_MIN_QUESTIONS,
) -> list[dict]:
    sources = {
        bare(r["host"]): r["source"]
        for r in hand
        if r["rule"] in ("bulk", SOURCE) and r.get("source")
    }
    hand_entries = [hand_entry(r) for r in hand if r["rule"] != SOURCE]
    sites: dict[str, dict] = {}
    for entry in (
        hand_entries
        + benchmark_entries(benchmark_rows, sources, extra_min)
        + listed_entries(decisions, backing, judged, dated, since)
    ):
        sites.setdefault(entry["host"], entry)
    return [sites[host] for host in sorted(sites)]


def summary(sites: list[dict]) -> Counter:
    return Counter(
        (site["rule"], site.get("type", ""), "cap" in site) for site in sites
    )


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
        "--benchmarks",
        required=True,
        help="benchmark-sites.csv: benchmark sites, extras and evergreen additions",
    )
    parser.add_argument(
        "--hand",
        required=True,
        help="TSV of host, rule, source, type, why, cap, since: hosts set by hand, and rule 'source' to name a benchmark site's bulk source",
    )
    parser.add_argument(
        "--backing", help="TSV of host, wikidata_type, benchmark_citations, news_list"
    )
    parser.add_argument("--judged", help="TSV of host, judged, substantive")
    parser.add_argument(
        "--dated",
        help="store counts TSV (.gz allowed) with host and dated_2020_on, for the news size guard",
    )
    parser.add_argument("--since", default=SINCE, help="the news date floor")
    parser.add_argument("--extra-min-questions", type=int, default=EXTRA_MIN_QUESTIONS)
    args = parser.parse_args(argv)
    with open(args.benchmarks, newline="") as source:
        benchmark_rows = list(csv.DictReader(source))
    backing = {r["host"]: r for r in tsv(args.backing)}
    judged = {r["host"]: r for r in tsv(args.judged)}
    dated = {r["host"]: int(r["dated_2020_on"]) for r in tsv(args.dated)}
    with open_text(args.decisions) as source:
        decisions = csv.DictReader(source, delimiter="\t")
        sites = build(
            tsv(args.hand),
            benchmark_rows,
            decisions,
            backing,
            judged,
            dated,
            args.since,
            args.extra_min_questions,
        )
    if not sites:
        sys.exit(f"no sites; {args.out} left as it was")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    temporary = Path(f"{args.out}.new")
    temporary.write_text(
        json.dumps({"built": built, "sites": sites}, separators=(",", ":"))
    )
    temporary.replace(args.out)
    print(f"{len(sites)} sites in {args.out}")
    for (rule, site_type, capped), count in sorted(summary(sites).items()):
        print(f"{count:>8}  {rule:6} {site_type:13} {'capped' if capped else ''}")


if __name__ == "__main__":
    main()
