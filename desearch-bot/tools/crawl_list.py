"""Builds the bot's crawl list from the registry: the domains it may queue, each with a page cap and a send weight."""

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

JEV = "jev-1.14/types-2"
QUERY = (
    "COPY (SELECT d.host, d.rank, d.state, d.state_reason, d.url_count, t.outcome, t.site_type, t.confidence"
    " FROM bot.domains d LEFT JOIN bot.site_types t ON t.host = d.host AND t.source = '"
    + JEV
    + "')"
    " TO STDOUT WITH (FORMAT csv)"
)

TIERS = {
    "news": 1,
    "reference": 2,
    "academic": 2,
    "public": 2,
    "docs": 2,
    "media": 3,
    "community": 3,
    "personal": 4,
    "organization": 4,
    "company": 4,
}
WEIGHTS = {1: 4, 2: 3, 3: 2, 4: 1}
OUT_TYPES = {"commerce": "shop", "no_value": "junk"}
WIKIDATA_TYPES = {
    "News": "news",
    "Universities and research": "academic",
    "Government and intergovernmental": "public",
    "Databases, wikis, reference sites": "reference",
    "Libraries, museums, archives": "reference",
}
REFUSALS = {
    "homepage_401",
    "homepage_402",
    "homepage_403",
    "robots_disallow",
    "blocked",
}
CRAWLED_STATES = {"active", "no_sitemap", "failing"}
# Only these list pages from their sitemaps, so only these go in the bot's list; the decisions file has the rest.
SENDING_STATES = {"active", "failing"}
RECHECKED = {"no_text", "homepage_error"}
TOP_RANK = 10_000
LOW_CAP = 10_000
UNSURE_CAP = 100_000
# The crawl takes three pages for each one the index may keep, so admission can choose.
CRAWL_PER_INDEX = 3
HOURLY_TOP = 20_000
HOURLY = 5_000
MIN_DOMAINS = 50_000
# A judged site with most sampled pages substantive earns a bigger cap; one with few keeps that share of its cap, at least the low cap.
JUDGED_MIN = 10
EARNED_SHARE = 0.8
FAILED_SHARE = 0.4
EARNED = 5
EARNING_TYPES = {
    "news",
    "reference",
    "media",
    "community",
    "academic",
    "public",
    "docs",
}
# With a v1 set, its sites send first and the rest of the list fills the room they leave.
V1_NEWS_WEIGHT = 4
V1_WEIGHT = 3
NODE2_WEIGHT = 2
# A v1 site taken whole: every stored page may go, up to this many.
V1_WHOLE_CAP = 5_000_000
# Outs a v1 site lifts: the type model's labels, never a refusal, a registry state or a hand decision.
SOFT_OUTS = {"unlabelled", "shop", "junk"}
# A v1 site that names no sitemap is listed too, so the bot looks for its feed; it sends nothing until it lists pages.
FEED_STATES = {"no_sitemap"}


@dataclass
class Row:
    host: str
    rank: int | None
    state: str
    state_reason: str
    url_count: int
    homepage: str
    site_type: str
    confidence: float | None


@dataclass
class Backing:
    wikidata_type: str = ""
    benchmark_citations: int = 0
    news_list: bool = False

    def __bool__(self) -> bool:
        return bool(self.wikidata_type or self.benchmark_citations or self.news_list)


@dataclass
class Decision:
    listed: bool
    reason: str = ""
    site_type: str = ""
    tier: int = 0
    weight: int = 0
    cap: int = 0
    send_cap: int | None = None
    backlog: int | None = None
    hourly: int = 0
    since: str | None = None


@dataclass
class Hand:
    exact: set[str] = field(default_factory=set)
    parts: list[str] = field(default_factory=list)

    def matches(self, host: str) -> bool:
        return host in self.exact or any(part in host for part in self.parts)


@dataclass
class Judged:
    pages: int
    substantive: int


@dataclass
class V1:
    rule: str
    site_type: str = ""
    since: str | None = None


def out(reason: str) -> Decision:
    return Decision(False, reason)


def judged_cap(cap: int, site_type: str, judged: Judged | None) -> int:
    if judged is None or judged.pages < JUDGED_MIN:
        return cap
    share = judged.substantive / judged.pages
    if share >= EARNED_SHARE and site_type in EARNING_TYPES:
        return cap * EARNED
    if share <= FAILED_SHARE:
        return min(cap, max(LOW_CAP, int(cap * share)))
    return cap


def type_cap(site_type: str, rank: int | None, backed: bool, sure: bool) -> int:
    top = rank is not None and rank <= TOP_RANK
    if site_type == "news":
        return 1_000_000 if top or backed else 500_000
    if site_type == "reference":
        return 1_000_000 if backed or (top and sure) else 100_000
    if site_type in ("academic", "public"):
        return 1_000_000 if backed else 200_000
    if site_type == "media":
        return 500_000 if backed else 100_000
    return {
        "docs": 200_000,
        "community": 200_000,
        "personal": 10_000,
        "organization": 5_000,
        "company": 2_000,
    }[site_type]


def decide(
    row: Row,
    backing: Backing,
    override: str = "",
    hand: Hand | None = None,
    judged: Judged | None = None,
) -> Decision:
    if override.startswith("out:"):
        return out(override[4:])
    if hand is not None and hand.matches(row.host):
        return out("hand")
    if row.state == "excluded":
        return out(row.state_reason or "excluded")
    if row.state == "blocked" or row.state_reason == "bot_wall":
        return out("refuses")
    if row.state_reason == "not_english" or row.homepage == "not_english":
        return out("not_english")
    if row.state == "ineligible" and not (
        row.state_reason in RECHECKED and row.homepage == "ok"
    ):
        return out(f"ineligible:{row.state_reason}")
    if row.state not in CRAWLED_STATES | {"ineligible"}:
        return out(f"state:{row.state}")
    if row.homepage in REFUSALS and override != "keep":
        return out("refuses")
    reviewed = override in TIERS
    site_type = (
        override
        if reviewed
        else row.site_type or WIKIDATA_TYPES.get(backing.wikidata_type, "")
    )
    if site_type in OUT_TYPES:
        return out(OUT_TYPES[site_type])
    if site_type not in TIERS:
        return out("unlabelled")
    confidence = 1.0 if reviewed or row.confidence is None else row.confidence
    cap = type_cap(site_type, row.rank, bool(backing), confidence >= 0.8)
    if not backing:
        if confidence < 0.5:
            cap = min(cap, LOW_CAP)
        elif confidence < 0.8:
            cap = min(cap, UNSURE_CAP)
    cap = judged_cap(cap, site_type, judged)
    tier = TIERS[site_type]
    crawl = cap * CRAWL_PER_INDEX
    top = row.rank is not None and row.rank <= TOP_RANK
    # News keeps getting new articles, so only its first walk is bounded.
    news = site_type == "news"
    return Decision(
        True,
        "",
        site_type,
        tier,
        WEIGHTS[tier],
        cap,
        send_cap=None if news else crawl,
        backlog=crawl if news else None,
        hourly=HOURLY_TOP if top else HOURLY,
    )


def v1_weight(site_type: str) -> int:
    return V1_NEWS_WEIGHT if site_type == "news" else V1_WEIGHT


def apply_v1(row: Row, base: Decision, v1: V1 | None, forced_out: bool) -> Decision:
    if v1 is None:
        return (
            replace(base, weight=min(base.weight, NODE2_WEIGHT))
            if base.listed
            else base
        )
    if v1.rule == "out":
        return out("v1")
    lifted = not base.listed and not forced_out and base.reason in SOFT_OUTS
    if v1.rule not in ("news", "whole") or not (base.listed or lifted):
        return replace(base, weight=v1_weight(base.site_type)) if base.listed else base
    site_type = v1.site_type if v1.site_type in TIERS else base.site_type
    if site_type not in TIERS:
        return base
    hourly = HOURLY_TOP if row.rank is not None and row.rank <= TOP_RANK else HOURLY
    if site_type == "news":
        cap = (
            base.cap
            if base.listed and base.site_type == "news"
            else type_cap("news", row.rank, True, True)
        )
        return Decision(
            True,
            "",
            "news",
            TIERS["news"],
            V1_NEWS_WEIGHT,
            cap,
            backlog=cap * CRAWL_PER_INDEX,
            hourly=hourly,
            since=v1.since if v1.rule == "news" else None,
        )
    return Decision(
        True,
        "",
        site_type,
        TIERS[site_type],
        V1_WEIGHT,
        V1_WHOLE_CAP,
        send_cap=V1_WHOLE_CAP * CRAWL_PER_INDEX,
        hourly=hourly,
    )


def registry_rows(csv_path: str | None) -> csv.reader:
    if csv_path:
        return csv.reader(open_text(csv_path))
    command = [
        "psql",
        os.environ["DESEARCH_DB"],
        "-X",
        "-q",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "SET statement_timeout = 0",
        "-c",
        QUERY,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    return csv.reader(process.stdout)


def parse_row(fields: list[str]) -> Row:
    host, rank, state, state_reason, url_count, homepage, site_type, confidence = fields
    return Row(
        host,
        int(rank) if rank else None,
        state,
        state_reason,
        int(url_count or 0),
        homepage,
        site_type,
        float(confidence) if confidence else None,
    )


def open_text(path: str, mode: str = "rt"):
    return (
        gzip.open(path, mode)
        if path.endswith(".gz")
        else open(path, mode.replace("t", ""), newline="")
    )


def read_backing(path: str | None) -> dict[str, Backing]:
    if not path:
        return {}
    with open_text(path) as source:
        return {
            r["host"]: Backing(
                r["wikidata_type"],
                int(r["benchmark_citations"] or 0),
                r["news_list"] == "1",
            )
            for r in csv.DictReader(source, delimiter="\t")
        }


def read_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open_text(path) as source:
        return {
            r["host"]: r["site_type"] for r in csv.DictReader(source, delimiter="\t")
        }


def read_judged(path: str | None) -> dict[str, Judged]:
    if not path:
        return {}
    with open_text(path) as source:
        return {
            r["host"]: Judged(int(r["judged"]), int(r["substantive"]))
            for r in csv.DictReader(source, delimiter="\t")
        }


def read_v1_set(path: str | None) -> dict[str, V1] | None:
    if not path:
        return None
    with open_text(path) as source:
        sites = json.load(source)["sites"]
    return {
        site["host"]: V1(site["rule"], site.get("type", ""), site.get("since"))
        for site in sites
    }


def read_hand(path: str | None) -> Hand:
    hand = Hand()
    if not path:
        return hand
    for line in Path(path).read_text().splitlines():
        entry = line.strip().lower()
        if entry.startswith("*") and entry.endswith("*") and len(entry) > 2:
            hand.parts.append(entry.strip("*"))
        elif entry:
            hand.exact.add(entry)
    return hand


def build(
    rows,
    backing: dict[str, Backing],
    overrides: dict[str, str],
    hand: Hand,
    decisions=None,
    judged: dict[str, Judged] | None = None,
    v1: dict[str, V1] | None = None,
    resend_before: str | None = None,
    resend_hosts: set[str] | None = None,
    node2_hours: float | None = None,
    node2_min_weight: int = 1,
) -> tuple[list[dict], Counter]:
    ranked, counts = [], Counter()
    for row in rows:
        override = overrides.get(row.host, "")
        base = decide(
            row,
            backing.get(row.host, Backing()),
            override,
            hand,
            (judged or {}).get(row.host),
        )
        entry_v1 = (v1 or {}).get(row.host)
        decision = (
            base
            if v1 is None
            else apply_v1(row, base, entry_v1, override.startswith("out:"))
        )
        if v1 is not None and entry_v1 is None and decision.listed and decision.weight < node2_min_weight:
            decision = out("node2")
        counts[decision.site_type if decision.listed else f"out:{decision.reason}"] += 1
        feed = (
            entry_v1 is not None
            and entry_v1.rule in ("news", "whole")
            and row.state in FEED_STATES
        )
        if decision.listed and (row.state in SENDING_STATES or feed):
            entry = {
                "host": row.host,
                "type": decision.site_type,
                "weight": decision.weight,
                "hourly": decision.hourly,
            }
            if node2_hours and v1 is not None and entry_v1 is None:
                bound = decision.send_cap or decision.backlog or row.url_count
                entry["hourly"] = min(
                    decision.hourly,
                    max(1, math.ceil(min(row.url_count, bound) / node2_hours)),
                )
            if decision.send_cap is not None:
                entry["send_cap"] = decision.send_cap
            if decision.backlog is not None:
                entry["backlog"] = decision.backlog
            if decision.since is not None:
                entry["since"] = decision.since
            if (
                resend_before
                and row.host in (resend_hosts or set())
                and entry_v1 is not None
                and entry_v1.rule in ("news", "whole")
            ):
                entry["resend_before"] = resend_before
            ranked.append(
                (
                    row.rank is None,
                    row.rank or 0,
                    row.host,
                    entry,
                    feed and row.state not in SENDING_STATES,
                )
            )
        # The decisions file keeps the decision before the v1 set, which the set itself is built from.
        if decisions is not None:
            decisions.writerow(
                [
                    row.host,
                    row.rank or "",
                    row.state,
                    int(base.listed),
                    base.reason,
                    base.site_type,
                    base.cap,
                    entry_v1.rule if entry_v1 else "",
                ]
            )
    # A listed subdomain takes its pages from its parent's lists, so one listed only for its feed would leave them unsent.
    hosts = {item[2] for item in ranked}
    ranked = [
        item for item in ranked if not (item[4] and parent_listed(item[2], hosts))
    ]
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked], counts


def parent_listed(host: str, hosts: set[str]) -> bool:
    labels = host.split(".")
    return any(".".join(labels[at:]) in hosts for at in range(1, len(labels) - 1))


def write_list(entries: list[dict], path: str) -> None:
    if len(entries) < MIN_DOMAINS:
        sys.exit(
            f"only {len(entries)} domains listed, fewer than {MIN_DOMAINS}; {path} left as it was"
        )
    temporary = Path(f"{path}.new")
    temporary.write_text(json.dumps(entries, separators=(",", ":")))
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="the crawl list the bot reads, such as domains.json",
    )
    parser.add_argument(
        "--backing", help="TSV of host, wikidata_type, benchmark_citations, news_list"
    )
    parser.add_argument(
        "--overrides",
        help="TSV of host, site_type, note; site_type is a type, keep, or out:<reason>",
    )
    parser.add_argument(
        "--excluded",
        help="the hand list: one host per line, or *part* to match inside names",
    )
    parser.add_argument(
        "--decisions", help="TSV (.gz allowed) of every domain's decision"
    )
    parser.add_argument(
        "--judged",
        help="TSV of host, judged, substantive: sampled pages a judge read, and how many had content of their own",
    )
    parser.add_argument(
        "--v1-set",
        help="v1-set.json: its sites send first under its caps, and the rest of the list at lower weights",
    )
    parser.add_argument(
        "--resend-before",
        help="an ISO date: the pages of the v1 sites in --resend-hosts last sent before it go once more (extraction changed)",
    )
    parser.add_argument(
        "--resend-hosts",
        action="append",
        default=[],
        help="a file of hosts, one a line, that take --resend-before: the resend waves applied so far",
    )
    parser.add_argument(
        "--node2-hours",
        type=float,
        help="with --v1-set, each site outside it sends at most its stored pages under its cap divided by this many hours in an hour, at least one, so the v1 sites keep the rest of the cap",
    )
    parser.add_argument(
        "--node2-min-weight",
        type=int,
        default=1,
        help="with --v1-set, sites outside it weighted below this stay off the list; 2 leaves out companies, organizations and personal sites",
    )
    parser.add_argument(
        "--registry-csv",
        help="read the registry rows from this CSV instead of DESEARCH_DB",
    )
    args = parser.parse_args(argv)
    rows = (parse_row(fields) for fields in registry_rows(args.registry_csv))
    backing, overrides, hand, judged, v1 = (
        read_backing(args.backing),
        read_overrides(args.overrides),
        read_hand(args.excluded),
        read_judged(args.judged),
        read_v1_set(args.v1_set),
    )
    resend_hosts = {
        line.strip()
        for path in args.resend_hosts
        for line in Path(path).read_text().splitlines()
        if line.strip()
    }
    if args.decisions:
        with open_text(args.decisions, "wt") as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["host", "rank", "state", "listed", "reason", "type", "cap", "v1"]
            )
            entries, counts = build(
                rows,
                backing,
                overrides,
                hand,
                writer,
                judged,
                v1,
                args.resend_before,
                resend_hosts,
                args.node2_hours,
                args.node2_min_weight,
            )
    else:
        entries, counts = build(
            rows,
            backing,
            overrides,
            hand,
            judged=judged,
            v1=v1,
            resend_before=args.resend_before,
            resend_hosts=resend_hosts,
            node2_hours=args.node2_hours,
            node2_min_weight=args.node2_min_weight,
        )
    write_list(entries, args.out)
    print(f"{len(entries)} domains listed in {args.out}")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"{count:>10}  {name}")


if __name__ == "__main__":
    main()
