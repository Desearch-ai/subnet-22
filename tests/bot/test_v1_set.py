import gzip
import importlib.util
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "v1_set.py"
SPEC = importlib.util.spec_from_file_location("v1_set", SOURCE)
v1_set = importlib.util.module_from_spec(SPEC)
sys.modules["v1_set"] = v1_set
SPEC.loader.exec_module(v1_set)


def decision(host, site_type, cap=1_000_000, listed="1"):
    return {
        "host": host,
        "rank": "100",
        "state": "active",
        "listed": listed,
        "reason": "",
        "type": site_type,
        "cap": str(cap),
    }


def bench(
    site,
    tier="benchmark",
    fix="none",
    take="whole site",
    site_type="news",
    hosts="",
    questions="5",
    unlocks="SQA 2; BCP 1",
    why="",
):
    return {
        "tier": tier,
        "site": site,
        "hosts": hosts or site,
        "questions": questions,
        "unlocks": unlocks,
        "type": site_type,
        "fix": fix,
        "why": why,
        "take": take,
    }


def build(decisions=(), benchmarks=(), hand=(), backing=None, judged=None, dated=None):
    sites = v1_set.build(
        list(hand),
        list(benchmarks),
        list(decisions),
        backing or {},
        judged or {},
        dated or {},
    )
    return {site["host"]: site for site in sites}


def test_listed_news_sites_take_the_date_floor():
    sites = build([decision("news.example", "news")])
    assert sites["news.example"] == {
        "host": "news.example",
        "rule": "news",
        "since": "2020-01-01",
        "type": "news",
        "why": ["news"],
    }


def test_a_big_unbacked_news_site_is_held_until_judged():
    dated = {
        "big.example": 5_000_000,
        "listed.example": 5_000_000,
        "small.example": 900_000,
    }
    backing = {
        "listed.example": {
            "news_list": "1",
            "benchmark_citations": "0",
            "wikidata_type": "",
        }
    }
    sites = build([decision(h, "news") for h in dated], backing=backing, dated=dated)
    assert sites["big.example"]["cap"] == v1_set.NEWS_GUARD
    assert "cap" not in sites["listed.example"]
    assert "cap" not in sites["small.example"]


def test_evergreen_types_keep_their_judged_cap_and_reference_needs_the_judge():
    judged = {
        "passed.example": {"judged": "20", "substantive": "17"},
        "thin.example": {"judged": "20", "substantive": "9"},
        "few.example": {"judged": "5", "substantive": "5"},
    }
    decisions = [decision("uni.example", "academic", 200_000)] + [
        decision(h, "reference", 500_000)
        for h in ("passed.example", "thin.example", "few.example", "unread.example")
    ]
    sites = build(decisions, judged=judged)
    assert (
        sites["uni.example"]["rule"] == "whole"
        and sites["uni.example"]["cap"] == 200_000
    )
    assert sites["passed.example"]["cap"] == 500_000 and sites["passed.example"][
        "why"
    ] == ["judge"]
    assert {"thin.example", "few.example", "unread.example"}.isdisjoint(sites)


def test_other_types_and_unlisted_sites_stay_out():
    sites = build(
        [
            decision("shop.example", "company"),
            decision("mag.example", "media"),
            decision("gone.example", "news", listed="0"),
        ]
    )
    assert sites == {}


def test_benchmark_sites_go_whole_with_the_take_caps_and_their_hosts():
    benchmarks = [
        bench("paper.example", hosts="paper.example www.paper.example"),
        bench(
            "corp.example",
            take="relevant sections, up to 20,000 pages",
            site_type="company",
        ),
        bench(
            "db.example",
            take="template database: judge first, up to 1,000,000 pages",
            site_type="reference",
        ),
        bench(
            "portal.example",
            take="cited hosts: a.portal.example b.portal.example",
            site_type="public",
        ),
        bench("refuses.example", tier="gap", fix="accept the gap"),
    ]
    sites = build([decision("paper.example", "news")], benchmarks)
    assert sites["paper.example"] == {
        "host": "paper.example",
        "rule": "whole",
        "type": "news",
        "why": ["simpleqa", "browsecomp-plus"],
    }
    assert sites["corp.example"]["cap"] == 20_000
    assert sites["db.example"]["cap"] == 1_000_000
    assert {"a.portal.example", "b.portal.example"} <= set(
        sites
    ) and "portal.example" not in sites
    assert "refuses.example" not in sites


def test_extras_need_enough_questions_and_evergreen_types_are_mapped():
    benchmarks = [
        bench("many.example", tier="benchmark (extra)", questions="3", unlocks=""),
        bench("two.example", tier="benchmark (extra)", questions="2", unlocks=""),
        bench("docs.example", tier="evergreen", site_type="technical docs", unlocks=""),
    ]
    sites = build(benchmarks=benchmarks)
    assert sites["many.example"]["why"] == ["benchmark-extra"]
    assert "two.example" not in sites
    assert sites["docs.example"]["type"] == "docs"


def test_hand_entries_win_and_name_bulk_sources():
    hand = [
        {
            "host": "en.wikipedia.org",
            "rule": "bulk",
            "source": "wikipedia",
            "type": "reference",
            "why": "frames simpleqa",
            "cap": "",
            "since": "",
        },
        {
            "host": "mirror.example",
            "rule": "out",
            "source": "",
            "type": "",
            "why": "",
            "cap": "",
            "since": "",
        },
        {
            "host": "dumps.example",
            "rule": "source",
            "source": "dumps-api",
            "type": "",
            "why": "",
            "cap": "",
            "since": "",
        },
    ]
    benchmarks = [
        bench(
            "dumps.example",
            fix="use a bulk source",
            site_type="reference",
            hosts="dumps.example data.dumps.example",
        ),
        bench(
            "game.fandom.com",
            fix="use a bulk source",
            site_type="reference",
            why="refuses crawling; Fandom wiki XML dumps (CC BY-SA)",
        ),
    ]
    sites = build(
        [decision("mirror.example", "reference")],
        benchmarks,
        hand,
        judged={"mirror.example": {"judged": "20", "substantive": "20"}},
    )
    assert sites["en.wikipedia.org"] == {
        "host": "en.wikipedia.org",
        "rule": "bulk",
        "source": "wikipedia",
        "type": "reference",
        "why": ["frames", "simpleqa"],
    }
    assert sites["mirror.example"] == {"host": "mirror.example", "rule": "out"}
    assert sites["data.dumps.example"]["source"] == "dumps-api"
    assert sites["game.fandom.com"]["source"] == "fandom"
    assert "dumps.example" in sites and sites["dumps.example"]["rule"] == "bulk"


def test_the_set_file_is_written_from_the_inputs(tmp_path):
    decisions = tmp_path / "decisions.tsv.gz"
    with gzip.open(decisions, "wt") as out:
        out.write("host\trank\tstate\tlisted\treason\ttype\tcap\n")
        out.write("news.example\t1\tactive\t1\t\tnews\t1000000\n")
        out.write("shop.example\t2\tactive\t0\tshop\t\t0\n")
    benchmarks = tmp_path / "benchmark-sites.csv"
    benchmarks.write_text(
        "tier,rank,site,hosts,questions,unlocks,gain,type,crawl_state,fix,why,take\nbenchmark,1,www.Docs.example,www.Docs.example,4,Ours 4,1,docs,crawled,none,,whole site\n"
    )
    hand = tmp_path / "hand.tsv"
    hand.write_text(
        "host\trule\tsource\ttype\twhy\tcap\tsince\nwiki.example\tout\t\t\t\t\t\n"
    )
    out = tmp_path / "v1-set.json"
    v1_set.main(
        [
            "--out",
            str(out),
            "--decisions",
            str(decisions),
            "--benchmarks",
            str(benchmarks),
            "--hand",
            str(hand),
        ]
    )
    written = json.loads(out.read_text())
    assert [site["host"] for site in written["sites"]] == [
        "docs.example",
        "news.example",
        "wiki.example",
    ]
    assert written["built"].endswith("Z")
