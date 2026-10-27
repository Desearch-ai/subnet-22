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

GOLDEN = Path(__file__).resolve().parent / "v1_set_golden.json"


def decision(host, site_type, cap=1_000_000, listed="1"):
    return {"host": host, "rank": "100", "state": "active", "listed": listed, "reason": "", "type": site_type, "cap": str(cap)}


def build(decisions=(), rows=(), hand=(), backing=None, judged=None, counts=None):
    sites, too_big = v1_set.build([list(r) for r in rows], list(hand), list(decisions), backing or {}, judged or {}, counts or {})
    return {site["host"]: site for site in sites}, too_big


def test_listed_news_sites_take_the_date_floor():
    sites, _ = build([decision("news.example", "news")])
    assert sites["news.example"] == {"host": "news.example", "rule": "news", "since": "2020-01-01", "type": "news", "why": ["news"]}


def test_a_big_unbacked_news_site_stays_out_until_judged():
    counts = {h: {"stored": "9000000", "dated_2020_on": "5000000"} for h in ("big.example", "listed.example")}
    counts["small.example"] = {"stored": "900000", "dated_2020_on": "900000"}
    backing = {"listed.example": {"news_list": "1", "benchmark_citations": "0", "wikidata_type": ""}}
    sites, _ = build([decision(h, "news") for h in counts], backing=backing, counts=counts)
    assert sites["big.example"] == {"host": "big.example", "rule": "out", "why": ["size-guard"]}
    assert sites["listed.example"]["rule"] == sites["small.example"]["rule"] == "news"


def test_evergreen_sites_go_whole_only_when_they_fit_and_reference_needs_the_judge():
    judged = {
        "passed.example": {"judged": "20", "substantive": "17"},
        "thin.example": {"judged": "20", "substantive": "9"},
        "few.example": {"judged": "5", "substantive": "5"},
    }
    counts = {"uni.example": {"stored": "500000"}, "huge.example": {"stored": "700000"}}
    decisions = [decision("uni.example", "academic", 200_000), decision("huge.example", "public", 200_000)]
    decisions += [decision(h, "reference", 500_000) for h in ("passed.example", "thin.example", "few.example", "unread.example")]
    sites, too_big = build(decisions, judged=judged, counts=counts)
    assert sites["uni.example"] == {"host": "uni.example", "rule": "whole", "type": "academic", "why": ["academic"]}
    assert "huge.example" not in sites and too_big == [{"host": "huge.example", "type": "public", "stored": 700_000, "cap": 200_000}]
    assert sites["passed.example"]["why"] == ["judge"]
    assert {"thin.example", "few.example", "unread.example"}.isdisjoint(sites)


def test_other_types_and_unlisted_sites_stay_out():
    sites, _ = build([decision("shop.example", "company"), decision("mag.example", "media"), decision("gone.example", "news", listed="0")])
    assert sites == {}


def test_row_files_win_in_order_and_keep_only_the_set_fields():
    benchmark = [{"host": "www.Paper.example", "rule": "news", "type": "news", "why": ["simpleqa"], "questions": 9, "tier": "benchmark", "note": "x"}]
    bulk = [{"host": "paper.example", "rule": "bulk", "source": "dump", "type": "news", "why": ["judge"]}, {"host": "wikipedia.org", "rule": "out", "why": ["not-english"]}]
    hand = [{"host": "mirror.example", "rule": "out", "why": ["mirror"]}]
    sites, _ = build([decision("paper.example", "news"), decision("mirror.example", "academic")], [benchmark, bulk], hand)
    assert sites["paper.example"] == {"host": "paper.example", "rule": "news", "type": "news", "why": ["simpleqa"]}, "all years, as the benchmark row says"
    assert sites["wikipedia.org"] == {"host": "wikipedia.org", "rule": "out", "why": ["not-english"]}
    assert sites["mirror.example"] == {"host": "mirror.example", "rule": "out", "why": ["mirror"]}


def test_dates_come_from_published_or_the_url_path_as_the_planner_reads_them():
    page_date = v1_set.page_date
    assert page_date("https://a.example/x", "2024-03-15T10:00:00+00:00").isoformat() == "2024-03-15"
    assert page_date("https://a.example/2023/11/02/story", "March 15, 2024").isoformat() == "2023-11-02"
    assert page_date("https://a.example/news/2021/7/slug", None).isoformat() == "2021-07-01"
    assert page_date("https://a.example/a/story-2022-05-09-slug", None).isoformat() == "2022-05-09"
    assert page_date("https://a.example/20190412/slug.html", None).isoformat() == "2019-04-12"
    for undated in ("https://a.example/a/123456789/slug", "https://a.example/p/2024", "https://a.example/x?d=2024-01-01", "https://a.example/2024/13/x", "https://2024-01-01.example/a"):
        assert page_date(undated, None) is None, undated


def test_the_matcher_answers_the_golden_cases():
    golden = json.loads(GOLDEN.read_text())
    sites = {site["host"]: site for site in golden["sites"]}
    for case in golden["cases"]:
        assert v1_set.matches(sites, case["url"], case["published"]) == case["in"], case


def test_the_set_file_is_written_from_the_inputs(tmp_path):
    decisions = tmp_path / "decisions.tsv.gz"
    with gzip.open(decisions, "wt") as out:
        out.write("host\trank\tstate\tlisted\treason\ttype\tcap\tv1\n")
        out.write("news.example\t1\tactive\t1\t\tnews\t1000000\t\n")
        out.write("shop.example\t2\tactive\t0\tshop\t\t0\t\n")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"built": "2026-10-28", "sites": [{"host": "docs.example", "rule": "whole", "paths": ["/guide/"], "type": "docs", "why": ["exa-webcode"]}], "dump_sites": [{"host": "dump.example", "rule": "bulk", "source": "dump", "type": "reference", "why": ["benchmark"]}]}))
    hand = tmp_path / "hand.tsv"
    hand.write_text("host\trule\ttype\twhy\nwiki.example\tout\t\tmirror\n")
    out = tmp_path / "v1-set.json"
    v1_set.main(["--out", str(out), "--decisions", str(decisions), "--rows", str(rows), "--hand", str(hand), "--too-big", str(tmp_path / "too-big.tsv")])
    written = json.loads(out.read_text())
    assert [site["host"] for site in written["sites"]] == ["docs.example", "dump.example", "news.example", "wiki.example"]
    assert written["sites"][0]["paths"] == ["/guide/"]
    assert written["built"].endswith("Z")
