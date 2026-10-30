import gzip
import importlib.util
from datetime import date
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "v1_set.py"
SPEC = importlib.util.spec_from_file_location("v1_set", SOURCE)
v1_set = importlib.util.module_from_spec(SPEC)
sys.modules["v1_set"] = v1_set
SPEC.loader.exec_module(v1_set)

GOLDEN = Path(__file__).resolve().parent / "v1_set_golden.json"
GOLDEN_UNDATED = Path(__file__).resolve().parent / "v1_set_golden_undated.json"


TODAY = date(2026, 10, 28)


def decision(host, site_type, cap=1_000_000, listed="1"):
    return {"host": host, "rank": "100", "state": "active", "listed": listed, "reason": "", "type": site_type, "cap": str(cap)}


def build(decisions=(), rows=(), hand=(), backing=None, news_sources=None, judged=None, counts=None, news="A", news_counts=None, outlets=None, verdicts=None, landing_rank=None):
    sites, too_big = v1_set.build([list(r) for r in rows], list(hand), list(decisions), backing or {}, news_sources or {}, judged or {}, counts or {}, news_counts or {}, news, TODAY, outlets, verdicts, landing_rank=landing_rank)
    return {site["host"]: site for site in sites}, too_big


def test_option_a_takes_established_outlets_for_all_years_and_other_news_sites_for_a_year():
    news_sources = {"outlet.example": "jev+wikidata", "listed.example": "jev+list897", "label.example": "jev+ut1"}
    backing = {"cited.example": {"benchmark_citations": "2", "news_list": "0", "wikidata_type": ""}}
    outlets = {"espn.example": "wikidata", "abcnews.example": "named", "news.portal.example": "wikidata"}
    hosts = ["outlet.example", "listed.example", "africa.espn.example", "abcnews.example", "portal.example", "cited.example", "label.example"]
    sites, _ = build([decision(h, "news") for h in hosts], backing=backing, news_sources=news_sources, outlets=outlets)
    assert sites["outlet.example"] == {"host": "outlet.example", "rule": "news", "type": "news", "why": ["wikidata"]}
    assert [sites[h].get("since") for h in hosts] == [None, None, None, None, None, "2025-10-01", "2025-10-01"], "a citation is no backing"
    assert (sites["africa.espn.example"]["why"], sites["abcnews.example"]["why"]) == (["wikidata"], ["named"])
    assert sites["label.example"]["why"] == ["news"]


def test_a_rows_news_window_comes_from_the_outlet_test_and_its_outs_give_reasons():
    benchmark = [
        {"host": "unbacked.example", "rule": "news", "type": "news", "why": ["simpleqa"]},
        {"host": "floored.example", "rule": "news", "since": "2025-10-01", "type": "news", "why": ["simpleqa"]},
        {"host": "mapcarta.example", "rule": "out", "why": ["simpleqa"], "note": "pages generated from OpenStreetMap and Wikidata"},
        {"host": "trove.example", "rule": "out", "why": ["simpleqa"], "note": "Trove's terms bar bulk and commercial use"},
        {"host": "odd.example", "rule": "out", "why": ["freshqa"], "note": "investor-relations pages"},
        {"host": "copy.example", "rule": "out", "why": ["mirror"]},
    ]
    sites, _ = build(rows=[benchmark], outlets={"floored.example": "wikidata"})
    assert sites["unbacked.example"]["since"] == "2025-10-01"
    assert "since" not in sites["floored.example"], "an outlet takes all years whatever the row said"
    assert [sites[h]["why"] for h in ("mapcarta.example", "trove.example", "odd.example", "copy.example")] == [["template"], ["terms"], ["low-value"], ["mirror"]]


def test_option_b_takes_every_news_site_from_2020():
    sites, _ = build([decision("outlet.example", "news"), decision("label.example", "news")], news_sources={"outlet.example": "wikidata"}, news="B")
    assert {site["since"] for site in sites.values()} == {"2020-01-01"}


def test_an_unbacked_news_site_the_judge_failed_is_out():
    judged = {"thin.example": {"judged": "20", "substantive": "6"}, "outlet.example": {"judged": "20", "substantive": "6"}}
    sites, _ = build([decision("thin.example", "news"), decision("outlet.example", "news")], news_sources={"outlet.example": "wikidata"}, judged=judged)
    assert sites["thin.example"] == {"host": "thin.example", "rule": "out", "why": ["judge"]}
    assert sites["outlet.example"]["rule"] == "news", "an established outlet stands whatever a sample says"


def test_an_unbacked_news_site_dating_more_pages_in_a_year_than_an_outlet_writes_is_out():
    news_counts = {"generated.example": {"dated_last_year": "900000"}, "busy.example": {"dated_last_year": "900000"}, "local.example": {"dated_last_year": "9000"}}
    sites, _ = build([decision(h, "news") for h in news_counts], news_sources={"busy.example": "wikidata"}, news_counts=news_counts, news="B")
    assert sites["generated.example"] == {"host": "generated.example", "rule": "out", "why": ["spam"]}
    assert sites["busy.example"]["rule"] == sites["local.example"]["rule"] == "news"


def test_sites_the_checks_read_as_empty_are_counted():
    sites = [{"host": "counted.example", "rule": "news"}, {"host": "new.example", "rule": "news"}, {"host": "big.example", "rule": "whole"}, {"host": "bulk.example", "rule": "bulk"}]
    assert v1_set.uncounted(sites, {"counted.example": {}}, {"counted.example": {}}) == (1, 1)


def test_spam_networks_are_sites_storing_exactly_as_many_urls_as_unrelated_ones():
    counts = {
        "craftingfiction.com": 4_459_629, "filmfeels.co.uk": 4_459_629, "fitclubcolumbus.com": 4_459_629,
        "ukcom.biz": 2_211_162, "ypage.uk": 2_211_162,
        "oxfordlearners.com": 1_235_050, "oxfordlearners.co.uk": 1_235_050,
        "gostickers.us": 730_038, "eatsauca.com": 730_038, "daim.co": 730_038,
        "pair-a.com": 730_038 + 1, "pair-b.com": 730_038 + 1,
        "cms-a.com": 2_000_000, "cms-b.com": 2_000_000, "cms-c.com": 2_000_000,
    }
    found = v1_set.networks({host: {"stored": str(n)} for host, n in counts.items()})
    assert found == {"craftingfiction.com", "filmfeels.co.uk", "fitclubcolumbus.com", "ukcom.biz", "ypage.uk", "gostickers.us", "eatsauca.com", "daim.co"}


def test_people_search_and_forecast_sites_are_out_by_rule():
    hosts = ["fastpeoplesearch.example", "thepublicrecord.example", "cityforecast.example", "weatherblog.example", "weatherforddemocrat.example", "weather.gov"]
    types = ["reference", "news", "news", "news", "news", "public"]
    hosts.append("weatherchannel.example")
    types.append("news")
    counts = {"cityforecast.example": {"stored": "180000"}, "weatherblog.example": {"stored": "2000"}, "weatherforddemocrat.example": {"stored": "3470"}, "weather.gov": {"stored": "90000"}, "weatherchannel.example": {"stored": "46000000"}}
    sites, _ = build([decision(h, t) for h, t in zip(hosts, types)], judged={"fastpeoplesearch.example": {"judged": "20", "substantive": "20"}}, counts=counts, news_sources={"weatherforddemocrat.example": "list897+wikidata", "weatherchannel.example": "wikidata"})
    assert sites["weatherchannel.example"]["why"] == ["template"], "a forecast site is a template even when Wikidata knows its channel"
    assert sites["fastpeoplesearch.example"]["why"] == ["people-search"]
    assert sites["thepublicrecord.example"]["rule"] == "news", "a news site named for public records is news"
    assert sites["cityforecast.example"] == {"host": "cityforecast.example", "rule": "out", "why": ["template"]}
    assert sites["weatherblog.example"]["rule"] == "news", "a small weather blog is no template"
    assert sites["weatherforddemocrat.example"]["rule"] == "news", "a newspaper named for its town is too small to be a forecast template"
    assert sites["weather.gov"]["rule"] == "whole", "a public site is not a forecast template by its name"


def test_evergreen_sites_go_whole_only_when_they_fit_and_reference_needs_the_judge():
    judged = {"passed.example": {"judged": "20", "substantive": "17"}, "few.example": {"judged": "5", "substantive": "5"}}
    counts = {"uni.example": {"stored": "500000"}, "huge.example": {"stored": "700000"}}
    decisions = [decision("uni.example", "academic", 200_000), decision("huge.example", "public", 200_000)]
    decisions += [decision(h, "reference", 500_000) for h in ("passed.example", "few.example", "unread.example")]
    sites, too_big = build(decisions, judged=judged, counts=counts)
    assert sites["uni.example"] == {"host": "uni.example", "rule": "whole", "type": "academic", "why": ["academic"]}
    assert "huge.example" not in sites and too_big == [{"host": "huge.example", "type": "public", "stored": 700_000, "cap": 200_000}]
    assert sites["passed.example"]["why"] == ["judge"]
    assert {"few.example", "unread.example"}.isdisjoint(sites)


def test_other_types_and_unlisted_sites_stay_out():
    sites, _ = build([decision("shop.example", "company"), decision("mag.example", "media"), decision("gone.example", "news", listed="0")])
    assert sites == {}


def test_ranked_listed_sites_no_entry_takes_are_landing_sites():
    ranked = [decision(host, kind) for host, kind in [("vendor.example", "company"), ("dev.vendor.example", "company"), ("club.example", "organization"), ("mag.example", "media"), ("failed.example", "company"), ("people-search.example", "company"), ("blog.nature.example", "personal"), ("closed.example", "company"), ("big.example", "academic")]]
    low = {**decision("low.example", "company"), "rank": "200001"}
    unranked = {**decision("unranked.example", "company"), "rank": ""}
    judged = {"failed.example": {"judged": "20", "substantive": "4"}, "club.example": {"judged": "20", "substantive": "12"}}
    rows = [[{"host": "nature.example", "rule": "whole", "type": "academic", "why": ["simpleqa"]}, {"host": "closed.example", "rule": "out", "why": ["terms"]}]]
    counts = {"big.example": {"stored": "9000000"}}
    sites, too_big = build(ranked + [low, unranked], rows=rows, judged=judged, counts=counts, landing_rank=200_000)
    landing = {host for host, site in sites.items() if site["rule"] == "landing"}
    assert landing == {"vendor.example", "dev.vendor.example", "club.example", "mag.example", "big.example"}, "a subdomain of a landing site is one too; a too-big whole site is one"
    assert sites["vendor.example"] == {"host": "vendor.example", "rule": "landing", "type": "company", "why": ["rank"]}
    assert sites["closed.example"]["rule"] == "out" and "blog.nature.example" not in sites, "an entry above takes or shuts it"
    assert [site["host"] for site in too_big] == ["big.example"]
    assert all(site["rule"] != "landing" for site in build(ranked)[0].values()), "no landing sites without a rank"


def test_a_landing_site_gives_its_own_homepage_and_first_level_pages():
    sites = {"vendor.example": {"host": "vendor.example", "rule": "landing", "type": "company"}}
    for url in ("http://vendor.example", "https://www.vendor.example/products/", "https://vendor.example/en_GB/about", "https://vendor.example/EN/", "https://m.vendor.example/"):
        assert v1_set.matches(sites, url, None), url
    for url in ("https://vendor.example/products/gpu", "https://vendor.example/fr-fr/", "https://vendor.example/en-us/a/b", "https://vendor.example/?p=1", "https://shop.vendor.example/", "https://vendor.example/en-us/de-de/x"):
        assert not v1_set.matches(sites, url, "2025-01-01"), url


def test_hand_entries_win_then_row_files_in_order_and_docs_hosts_are_docs():
    benchmark = [
        {"host": "www.Paper.example", "rule": "news", "type": "news", "why": ["simpleqa"], "questions": 9, "tier": "benchmark", "note": "x"},
        {"host": "handwiki.org", "rule": "whole", "type": "reference", "why": ["benchmark-extra"]},
        {"host": "docs.vendor.example", "rule": "whole", "type": "company", "why": ["exa-webcode"]},
    ]
    bulk = [{"host": "paper.example", "rule": "bulk", "source": "dump", "type": "news", "why": ["judge"]}, {"host": "wikipedia.org", "rule": "out", "why": ["not-english"]}]
    hand = [{"host": "handwiki.org", "rule": "out", "why": ["mirror"]}]
    sites, _ = build([decision("paper.example", "news")], [benchmark, bulk], hand, outlets={"paper.example": "wikidata"})
    assert sites["paper.example"] == {"host": "paper.example", "rule": "news", "type": "news", "why": ["simpleqa"]}, "an outlet in a row takes all years under A"
    assert sites["handwiki.org"] == {"host": "handwiki.org", "rule": "out", "why": ["mirror"]}
    assert sites["docs.vendor.example"]["type"] == "docs"
    assert sites["wikipedia.org"]["rule"] == "out"


def moved(host, canonical):
    return {"host": host, "rank": "100", "state": "redirects", "listed": "0", "reason": "state:redirects", "type": "", "cap": "0", "canonical": canonical}


def test_a_site_that_moved_passes_its_backing_to_the_host_it_moved_to():
    rows = [
        {"host": "paper.co.uk", "rule": "news", "type": "news", "why": ["freshqa"]},
        {"host": "old.example", "rule": "news", "type": "news", "why": ["simpleqa"]},
        {"host": "museum.example", "rule": "whole", "type": "reference", "why": ["simpleqa"]},
        {"host": "paper.corp.example", "rule": "news", "type": "news", "why": ["simpleqa"]},
        {"host": "loop-a.example", "rule": "news", "type": "news", "why": ["simpleqa"]},
    ]
    outlets = {h: "wikidata" for h in ("paper.co.uk", "old.example", "paper.corp.example", "loop-a.example")}
    decisions = [moved("paper.co.uk", "paper.com"), decision("paper.com", "news"), moved("old.example", "thin.example"), decision("thin.example", "news")]
    decisions += [moved("museum.example", "www.museum.org"), moved("museum.org", "library.example"), decision("library.example", "reference")]
    decisions += [moved("paper.corp.example", "corp.example"), decision("corp.example", "news")]
    decisions += [moved("loop-a.example", "loop-b.example"), moved("loop-b.example", "loop-a.example"), decision("loop-b.example", "news")]
    judged = {"thin.example": {"judged": "20", "substantive": "2"}}
    sites, _ = build(decisions, rows=[rows], outlets=outlets, judged=judged)
    assert sites["paper.com"] == {"host": "paper.com", "rule": "news", "type": "news", "why": ["wikidata"]}, "the outlet test follows the move: all years"
    assert sites["thin.example"] == {"host": "thin.example", "rule": "out", "why": ["judge"]}, "the judge goes by the host's own backing"
    assert "library.example" not in sites, "a rule doesn't move, only backing"
    assert sites["corp.example"]["since"] == "2025-10-01" and sites["loop-b.example"]["since"] == "2025-10-01", "no backing to a parent or around a loop"
    station = build([moved("station.example", "group.example"), decision("group.example", "news")], outlets={"station.example": "wikidata"})[0]
    assert station["group.example"]["since"] == "2025-10-01", "only a site the set's own entries take passes its backing on"
    assert v1_set.moves({"a.example": "b.example", "b.example": "c.example", "docs.x.example": "x.example"}) == {"a.example": "c.example", "b.example": "c.example"}


def test_the_site_check_puts_hosts_out_over_every_source_but_the_hand_entries():
    hand = [{"host": "lead.example", "rule": "whole", "type": "reference", "why": ["evergreen"]}]
    rows = [{"host": "generated.example", "rule": "whole", "type": "reference", "why": ["judge"]}, {"host": "lead.example", "rule": "out", "why": ["template"]}]
    verdicts = {"generated.example": "out", "lead.example": "out", "fine.example": "keep", "unlisted.example": "out"}
    sites, _ = build([decision("fine.example", "academic"), decision("generated.example", "news")], rows=[rows], hand=hand, verdicts=verdicts)
    assert sites["generated.example"] == {"host": "generated.example", "rule": "out", "why": ["quality"]}
    assert sites["lead.example"]["rule"] == "whole", "the lead's entry wins"
    assert sites["fine.example"]["rule"] == "whole" and sites["unlisted.example"]["rule"] == "out"


def test_site_verdicts_are_keep_or_out(tmp_path):
    good = tmp_path / "verdicts.tsv.gz"
    with gzip.open(good, "wt") as out:
        out.write("host\tverdict\tcopied\nwww.Gen.example\tout\t0.4\nok.example\tkeep\t0\n")
    assert v1_set.read_verdicts(str(good)) == {"gen.example": "out", "ok.example": "keep"}
    bad = tmp_path / "bad.tsv"
    bad.write_text("host\tverdict\nodd.example\tcap\n")
    with pytest.raises(SystemExit):
        v1_set.read_verdicts(str(bad))


def test_dates_come_from_published_or_the_url_path_as_the_planner_reads_them():
    page_date = v1_set.page_date
    assert page_date("https://a.example/x", "2024-03-15T10:00:00+00:00").isoformat() == "2024-03-15"
    assert page_date("https://a.example/2023/11/02/story", "March 15, 2024").isoformat() == "2023-11-02"
    assert page_date("https://a.example/news/2021/7/slug", None).isoformat() == "2021-07-01"
    assert page_date("https://a.example/a/story-2022-05-09-slug", None).isoformat() == "2022-05-09"
    assert page_date("https://a.example/20190412/slug.html", None).isoformat() == "2019-04-12"
    assert page_date("https://a.example/2023/11/02/story", "2024-02-30").isoformat() == "2023-11-02", "a published day that doesn't exist falls back to the URL"
    assert page_date("https://a.example/x", "1985-06-01").isoformat() == "1985-06-01", "a published date of any year"
    assert page_date("https://a.example/a/20190412-2020-05-06", None).isoformat() == "2019-04-12", "left to right within a segment, as set.rs reads"
    assert page_date("https://a.example/2024/02/30/x", None).isoformat() == "2024-02-01", "a URL day that doesn't exist falls back to its month"
    for undated in ("https://a.example/a/123456789/slug", "https://a.example/p/2024", "https://a.example/x?d=2024-01-01", "https://a.example/2024/13/x", "https://2024-01-01.example/a"):
        assert page_date(undated, None) is None, undated


def test_the_matcher_answers_the_golden_cases():
    golden = json.loads(GOLDEN.read_text())
    sites = {site["host"]: site for site in golden["sites"]}
    for case in golden["cases"]:
        assert v1_set.matches(sites, case["url"], case["published"]) == case["in"], case


def test_undated_news_articles_are_in_only_when_the_set_says_so():
    golden = json.loads(GOLDEN_UNDATED.read_text())
    sites = {site["host"]: site for site in golden["sites"]}
    for case in golden["cases"]:
        assert v1_set.matches(sites, case["url"], case["published"], case["page_type"], golden["undated"]) == case["in"], case
    assert not v1_set.matches(sites, "https://www.latimes.com/california/story/council-vote", None, "article"), "undated pages stay out by default"


def test_the_set_file_is_written_from_the_inputs(tmp_path):
    decisions = tmp_path / "decisions.tsv.gz"
    with gzip.open(decisions, "wt") as out:
        out.write("host\trank\tstate\tlisted\treason\ttype\tcap\tv1\n")
        out.write("news.example\t1\tactive\t1\t\tnews\t1000000\t\n")
        out.write("shop.example\t2\tactive\t0\tshop\t\t0\t\n")
        out.write("vendor.example\t3\tno_sitemap\t1\t\tcompany\t2000\t\n")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"built": "2026-10-28", "sites": [{"host": "docs.example", "rule": "whole", "paths": ["/guide/"], "type": "docs", "why": ["exa-webcode"]}], "dump_sites": [{"host": "dump.example", "rule": "bulk", "source": "dump", "type": "reference", "why": ["benchmark"]}]}))
    hand = tmp_path / "hand.tsv"
    hand.write_text("host\trule\ttype\twhy\nwiki.example\tout\t\tmirror\n")
    news_domains = tmp_path / "news-domains.tsv"
    news_domains.write_text("host\tsources\nnews.example\tjev+wikidata\n")
    verdicts = tmp_path / "verdicts.tsv.gz"
    with gzip.open(verdicts, "wt") as target:
        target.write("host\tverdict\ngenerated.example\tout\nnews.example\tkeep\n")
    out = tmp_path / "v1-set.json"
    v1_set.main(["--out", str(out), "--decisions", str(decisions), "--rows", str(rows), "--hand", str(hand), "--news-domains", str(news_domains), "--news", "B", "--too-big", str(tmp_path / "too-big.tsv"), "--site-verdicts", str(verdicts), "--landing-rank", "3"])
    written = json.loads(out.read_text())
    assert [site["host"] for site in written["sites"]] == ["docs.example", "dump.example", "generated.example", "news.example", "vendor.example", "wiki.example"]
    golden = {site["host"]: site for site in json.loads(GOLDEN.read_text())["sites"]}
    assert written["sites"][2] == golden["generated.example"], "the golden cases hold a site the site check put out"
    assert written["news"] == "B" and written["sites"][3]["since"] == "2020-01-01"
    assert written["undated"] == "out" and written["sites"][4]["rule"] == "landing"
    assert written["built"].endswith("Z")
