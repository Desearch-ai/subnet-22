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

GOLDEN = Path(__file__).resolve().parents[2] / "publisher" / "tests" / "v1-set-golden.json"
GOLDEN_UNDATED = Path(__file__).resolve().parents[2] / "publisher" / "tests" / "v1-set-golden-undated.json"


TODAY = date(2026, 10, 28)


def decision(host, site_type, cap=1_000_000, listed="1"):
    return {"host": host, "rank": "100", "state": "active", "listed": listed, "reason": "", "type": site_type, "cap": str(cap)}


def build(decisions=(), rows=(), hand=(), backing=None, news_sources=None, judged=None, counts=None, news="A", news_counts=None, outlets=None, verdicts=None, landing_rank=None, **wider):
    sites, too_big = v1_set.build([list(r) for r in rows], list(hand), list(decisions), backing or {}, news_sources or {}, judged or {}, counts or {}, news_counts or {}, news, TODAY, outlets, verdicts, landing_rank=landing_rank, **wider)
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
    assert not v1_set.matches({"shop.example": {"host": "shop.example", "rule": "landing"}}, "https://shop.m.example/", None), "x.m.y is no mobile host, as set.rs reads it"


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
        assert v1_set.matches(sites, case["url"], case["published"], case["page_type"], golden["undated"], golden["hub_pages"]) == case["in"], case
    assert not v1_set.matches(sites, "https://www.latimes.com/california/story/council-vote", None, "article"), "undated pages stay out by default"
    assert not v1_set.matches(sites, "https://outlet.example/world/", None, "listing", golden["undated"]), "hubs need the file's hub pages"
    assert golden["hub_pages"] == v1_set.HUB_PAGES, "the builder writes the hub pages the golden cases read"


def ranked_at(host, site_type, rank):
    return {**decision(host, site_type), "rank": str(rank)}


def test_news_sites_taken_for_all_years_give_their_hubs_when_asked():
    news_sources = {"outlet.example": "wikidata"}
    rows = [[{"host": "cited.example", "rule": "news", "type": "news", "why": ["simpleqa"]}]]
    decisions = [decision("outlet.example", "news"), decision("local.example", "news")]
    sites, _ = build(decisions, rows=rows, news_sources=news_sources, outlets={"cited.example": "wikidata"}, hubs=True)
    assert sites["outlet.example"]["hubs"] is True and sites["cited.example"]["hubs"] is True
    assert "hubs" not in sites["local.example"], "a news site with a since gives no hubs"
    assert all("hubs" not in site for site in build(decisions, news_sources=news_sources)[0].values()), "none without the switch"


def test_top_landing_sites_go_deeper_or_whole_once_the_site_check_keeps_them():
    decisions = [ranked_at("vendor.example", "company", 9_000), ranked_at("far.example", "company", 10_001), ranked_at("mag.example", "media", 500), ranked_at("club.example", "organization", 800), ranked_at("wiki.example", "reference", 900), ranked_at("blog.example", "personal", 100), ranked_at("unchecked.example", "media", 700)]
    verdicts = {"mag.example": "keep", "club.example": "keep", "wiki.example": "keep", "blog.example": "keep"}
    sites, _ = build(decisions, verdicts=verdicts, landing_rank=200_000, deep_rank=10_000, whole_rank=10_000)
    assert sites["vendor.example"] == {"host": "vendor.example", "rule": "landing", "type": "company", "why": ["rank"], "depth": 3}
    assert "depth" not in sites["far.example"] and sites["far.example"]["rule"] == "landing"
    assert [sites[h]["rule"] for h in ("mag.example", "club.example", "wiki.example")] == ["whole"] * 3
    assert sites["mag.example"] == {"host": "mag.example", "rule": "whole", "type": "media", "why": ["rank"], "own_host": True}
    assert sites["blog.example"]["rule"] == sites["unchecked.example"]["rule"] == "landing", "other types, and sites the check hasn't kept, stay landing sites"
    assert all(site["rule"] == "landing" and "depth" not in site for site in build(decisions, verdicts=verdicts, landing_rank=200_000)[0].values()), "none without the ranks"
    deep = {"deep.example": {"host": "deep.example", "rule": "landing", "depth": 3}}
    assert v1_set.matches(deep, "https://deep.example/a/b/c", None) and not v1_set.matches(deep, "https://deep.example/a/b/c/d", None)


def test_docs_and_news_subdomains_of_sites_taken_for_their_rank_are_taken_as_their_parents_are():
    decisions = [ranked_at("vendor.example", "company", 9_000), ranked_at("mag.example", "media", 500), ranked_at("plain.example", "company", 50_000)]
    decisions += [ranked_at("docs.vendor.example", "company", 300_000), ranked_at("investor.vendor.example", "organization", 300_000), ranked_at("news.mag.example", "media", 300_000), ranked_at("help.plain.example", "company", 300_000)]
    decisions += [ranked_at("shop.vendor.example", "company", 300_000), {**ranked_at("support.vendor.example", "company", 300_000), "listed": "0"}, ranked_at("docs.bank.example", "company", 300_000)]
    rows = [[{"host": "bank.example", "rule": "whole", "type": "public", "why": ["public"]}]]
    sites, _ = build(decisions, rows=rows, verdicts={"mag.example": "keep"}, landing_rank=200_000, deep_rank=10_000, whole_rank=10_000, subdomains=True)
    assert sites["docs.vendor.example"] == {"host": "docs.vendor.example", "rule": "landing", "depth": 3, "type": "docs", "why": ["rank"]}
    assert sites["investor.vendor.example"]["type"] == "company" and sites["news.mag.example"] == {"host": "news.mag.example", "rule": "whole", "own_host": True, "type": "media", "why": ["rank"]}
    assert sites["help.plain.example"] == {"host": "help.plain.example", "rule": "landing", "type": "docs", "why": ["rank"]}
    assert not {"shop.vendor.example", "support.vendor.example", "docs.bank.example"} & set(sites), "another label, an unlisted row, or a parent taken for a backing gives none"
    assert "docs.vendor.example" not in build(decisions, rows=rows, landing_rank=200_000)[0]


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
    assert written["built"].endswith("Z") and "hub_pages" not in written
    with pytest.raises(SystemExit):
        v1_set.main(["--out", str(out), "--decisions", str(decisions), "--landing-rank", "3", "--deep-rank", "3"])
    suffixes = tmp_path / "public_suffix_list.dat"
    suffixes.write_text("// ===BEGIN ICANN DOMAINS===\ncom\n*.ck\n!www.ck\n")
    v1_set.main(["--out", str(out), "--decisions", str(decisions), "--news-domains", str(news_domains), "--landing-rank", "3", "--deep-rank", "3", "--hubs", "--public-suffixes", str(suffixes)])
    written = {site["host"]: site for site in json.loads(out.read_text())["sites"]} | {"": json.loads(out.read_text())}
    assert written["news.example"]["hubs"] is True and written["vendor.example"]["depth"] == 3
    assert written[""]["hub_pages"] == v1_set.HUB_PAGES


def test_public_suffixes_are_never_taken_deeper_or_whole_nor_parents_of_subdomains(tmp_path):
    listing = tmp_path / "public_suffix_list.dat"
    listing.write_text("// ===BEGIN ICANN DOMAINS===\nac.za\n*.ck\n!www.ck\n// ===BEGIN PRIVATE DOMAINS===\nherokuapp.com\nus.com\n")
    suffixes = v1_set.read_suffixes(str(listing))
    assert [v1_set.platform(h, suffixes) for h in ("ac.za", "herokuapp.com", "us.com", "co.ck", "www.ck", "vendor.example")] == [True, True, True, True, False, False]
    decisions = [ranked_at("herokuapp.com", "media", 50), ranked_at("us.com", "company", 60), ranked_at("ac.za", "organization", 70), ranked_at("mag.example", "media", 80)]
    decisions += [ranked_at("news.us.com", "company", 300_000), ranked_at("news.mag.example", "media", 300_000)]
    verdicts = {h: "keep" for h in ("herokuapp.com", "ac.za", "mag.example")}
    sites, _ = build(decisions, verdicts=verdicts, landing_rank=200_000, deep_rank=10_000, whole_rank=10_000, subdomains=True, suffixes=suffixes)
    assert [sites[h]["rule"] for h in ("herokuapp.com", "us.com", "ac.za")] == ["landing"] * 3 and "depth" not in sites["us.com"]
    assert sites["mag.example"]["own_host"] is True and "news.us.com" not in sites and sites["news.mag.example"]["own_host"] is True
    whole = {site["host"]: site for site in sites.values()}
    assert not v1_set.matches(whole, "https://shop.mag.example/x", None), "whole for its rank takes its own host only"


def test_a_bulk_only_hand_entry_is_out_in_the_set_and_bulk_only_in_the_planners_copy(tmp_path, capsys):
    decisions = tmp_path / "decisions.tsv"
    decisions.write_text("host\trank\tstate\tlisted\treason\ttype\tcap\tv1\nuniprot.org\t8483\tactive\t1\t\treference\t1000000\t\nvendor.example\t3\tno_sitemap\t1\t\tcompany\t2000\t\n")
    hand = tmp_path / "hand.tsv"
    hand.write_text("host\trule\ttype\twhy\tpaths\tsource\nuniprot.org\tbulk_only\treference\tbenchmark\t\tuniprot-swissprot\n")
    out = tmp_path / "v1-set.json"
    v1_set.main(["--out", str(out), "--decisions", str(decisions), "--hand", str(hand), "--landing-rank", "3"])
    canonical = json.loads(out.read_text())
    assert {s["host"]: s for s in canonical["sites"]}["uniprot.org"] == {"host": "uniprot.org", "rule": "out", "why": ["bulk-only"]}
    capsys.readouterr()
    v1_set.main(["--planner-from", str(out), "--hand", str(hand)])
    copy = json.loads(capsys.readouterr().out)
    uniprot = {"host": "uniprot.org", "rule": "bulk_only", "source": "uniprot-swissprot", "type": "reference", "why": ["benchmark"]}
    assert copy == {**canonical, "sites": [uniprot if s["host"] == "uniprot.org" else s for s in canonical["sites"]]}
    v1_set.main(["--planner-from", str(out)])
    assert json.loads(capsys.readouterr().out) == canonical
    unlisted = {**canonical, "sites": [s for s in canonical["sites"] if s["host"] != "uniprot.org"]}
    assert v1_set.planner_copy(unlisted, v1_set.hand_entries(v1_set.tsv(str(hand))))["sites"][-1] == uniprot
    with pytest.raises(SystemExit):
        v1_set.main(["--out", str(out)])


def test_hand_depths_and_switches_are_read_as_the_set_holds_them_and_a_bad_one_stops_the_build():
    rows = [{"host": "a.example", "rule": "landing", "depth": "3", "hubs": "", "why": "rank"}, {"host": "b.example", "rule": "news", "hubs": "TRUE", "why": "news"}, {"host": "c.example", "rule": "whole", "own_host": "false", "why": "docs"}]
    entries = {e["host"]: e for e in v1_set.hand_entries(rows)}
    assert (entries["a.example"]["depth"], entries["b.example"]["hubs"], entries["c.example"]["own_host"]) == (3, True, False)
    assert "hubs" not in entries["a.example"]
    for bad in ({"host": "d.example", "rule": "landing", "depth": "deep"}, {"host": "d.example", "rule": "landing", "depth": "-1"}, {"host": "d.example", "rule": "news", "hubs": "yes"}):
        with pytest.raises(SystemExit):
            v1_set.hand_entries([bad])
    assert [v1_set.bad_entry(e) for e in ({"depth": 0}, {"depth": True}, {"depth": "3"}, {"hubs": 1}, {"own_host": "true"}, {"depth": 2, "hubs": True}, {"depth": None})] == ["depth 0", "depth True", "depth '3'", "hubs 1", "own_host 'true'", None, None]


def test_registry_hosts_are_named_as_the_set_readers_look_them_up():
    counts = {"www.nhs.example": {"stored": "1000"}}
    decisions = [decision("www.nhs.example", "public", 200_000), decision("www.paper.example", "news"), decision("en.m.wiki.example", "academic")]
    decisions += [{**decision("www.vendor.example", "company"), "rank": "50"}]
    sites, _ = build(decisions, counts=counts, news_sources={"www.paper.example": "wikidata"}, landing_rank=200_000)
    assert set(sites) == {"nhs.example", "paper.example", "en.wiki.example", "vendor.example"}
    assert sites["paper.example"]["why"] == ["wikidata"], "the registry host's own backing and counts still count"
    for url in ("https://www.nhs.example/conditions/flu/", "https://nhs.example/a/b", "https://en.m.wiki.example/x"):
        assert v1_set.matches(sites, url, None), url
    assert v1_set.matches(sites, "https://www.vendor.example/products", None)
