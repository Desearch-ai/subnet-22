import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "crawl_list.py"
)
SPEC = importlib.util.spec_from_file_location("crawl_list", SOURCE)
crawl_list = importlib.util.module_from_spec(SPEC)
sys.modules["crawl_list"] = crawl_list
SPEC.loader.exec_module(crawl_list)

Backing, Hand, Row, decide = (
    crawl_list.Backing,
    crawl_list.Hand,
    crawl_list.Row,
    crawl_list.decide,
)


def row(
    host="example.com",
    rank=50_000,
    state="active",
    state_reason="",
    homepage="ok",
    site_type="news",
    confidence=0.95,
):
    return Row(host, rank, state, state_reason, 1000, homepage, site_type, confidence)


def test_shops_and_sites_without_content_stay_out():
    assert decide(row(site_type="commerce"), Backing()).reason == "shop"
    assert (
        decide(row(site_type="commerce"), Backing(benchmark_citations=4)).reason
        == "shop"
    )
    assert decide(row(site_type="no_value"), Backing()).reason == "junk"


def test_a_reviewed_type_replaces_the_label():
    found = decide(
        row(site_type="commerce", confidence=0.48), Backing(), override="reference"
    )
    assert (found.listed, found.site_type, found.cap) == (True, "reference", 100_000)


def test_refusals_stay_out_unless_a_recheck_kept_them():
    assert decide(row(homepage="homepage_403"), Backing()).reason == "refuses"
    assert decide(row(homepage="robots_disallow"), Backing()).reason == "refuses"
    assert (
        decide(
            row(state="blocked", state_reason="robots_disallow"),
            Backing(),
            override="keep",
        ).reason
        == "refuses"
    )
    assert (
        decide(row(state="ineligible", state_reason="bot_wall"), Backing()).reason
        == "refuses"
    )
    assert decide(row(homepage="homepage_403"), Backing(), override="keep").listed


def test_a_forced_reason_and_the_hand_list_win():
    assert decide(row(), Backing(), override="out:junk").reason == "junk"
    hand = Hand(exact={"example.com"}, parts=["tgstat"])
    assert decide(row(), Backing(), hand=hand).reason == "hand"
    assert decide(row(host="uk.tgstat.com"), Backing(), hand=hand).reason == "hand"
    assert decide(row(host="other.com"), Backing(), hand=hand).listed


def test_registry_states_and_language():
    assert (
        decide(row(state="excluded", state_reason="shop"), Backing()).reason == "shop"
    )
    assert decide(row(state="unreachable"), Backing()).reason == "state:unreachable"
    assert (
        decide(
            row(state="ineligible", state_reason="not_english", homepage=""), Backing()
        ).reason
        == "not_english"
    )
    assert decide(row(homepage="not_english"), Backing()).reason == "not_english"
    assert decide(
        row(state="ineligible", state_reason="no_text", homepage="ok"), Backing()
    ).listed
    assert (
        decide(
            row(state="ineligible", state_reason="no_text", homepage="blank"), Backing()
        ).reason
        == "ineligible:no_text"
    )


def test_unlabelled_domains_take_their_type_from_wikidata_or_wait():
    found = decide(
        row(site_type="", homepage="", confidence=None), Backing(wikidata_type="News")
    )
    assert (found.site_type, found.cap) == ("news", 1_000_000)
    assert (
        decide(row(site_type="", homepage="", confidence=None), Backing()).reason
        == "unlabelled"
    )
    assert (
        decide(
            row(site_type="", homepage="", confidence=None),
            Backing(wikidata_type="Companies and brands"),
        ).reason
        == "unlabelled"
    )


@pytest.mark.parametrize(
    "site_type, rank, backing, confidence, cap",
    [
        ("news", 5_000, Backing(), 0.95, 1_000_000),
        ("news", 50_000, Backing(), 0.95, 500_000),
        ("news", 50_000, Backing(news_list=True), 0.95, 1_000_000),
        ("reference", 5_000, Backing(), 0.95, 1_000_000),
        ("reference", 5_000, Backing(), 0.6, 100_000),
        ("reference", 50_000, Backing(benchmark_citations=2), 0.4, 1_000_000),
        ("academic", 50_000, Backing(), 0.95, 200_000),
        (
            "public",
            50_000,
            Backing(wikidata_type="Government and intergovernmental"),
            0.95,
            1_000_000,
        ),
        ("media", 50_000, Backing(), 0.95, 100_000),
        ("media", 50_000, Backing(), 0.4, 10_000),
        ("docs", 50_000, Backing(), 0.95, 200_000),
        ("community", 50_000, Backing(), 0.7, 100_000),
        ("personal", 50_000, Backing(), 0.95, 10_000),
        ("organization", 50_000, Backing(), 0.95, 5_000),
        ("company", 50_000, Backing(), 0.95, 2_000),
    ],
)
def test_caps_by_type_backing_and_confidence(site_type, rank, backing, confidence, cap):
    found = decide(row(site_type=site_type, rank=rank, confidence=confidence), backing)
    assert (found.cap, found.send_cap) == (cap, cap * 3 // 2)


def test_weights_follow_tiers():
    weights = {
        t: decide(row(site_type=t), Backing()).weight
        for t in ("news", "reference", "media", "company")
    }
    assert weights == {"news": 4, "reference": 3, "media": 2, "company": 1}


def test_the_list_holds_sending_domains_best_rank_first(tmp_path, monkeypatch):
    rows = [
        row(host="b.com", rank=20),
        row(host="a.com", rank=None),
        row(host="c.com", rank=10, site_type="commerce"),
        row(host="d.com", rank=5, state="no_sitemap"),
        row(host="e.com", rank=1),
    ]
    entries, counts = crawl_list.build(rows, {}, {}, Hand())
    assert [e["host"] for e in entries] == ["e.com", "b.com", "a.com"]
    assert entries[0] == {
        "host": "e.com",
        "type": "news",
        "weight": 4,
        "send_cap": 1_500_000,
    }
    assert counts == {"news": 4, "out:shop": 1}
    monkeypatch.setattr(crawl_list, "MIN_DOMAINS", 2)
    target = tmp_path / "domains.json"
    crawl_list.write_list(entries, str(target))
    assert [e["host"] for e in json.loads(target.read_text())] == [
        "e.com",
        "b.com",
        "a.com",
    ]


def test_a_short_list_leaves_the_old_one(tmp_path):
    target = tmp_path / "domains.json"
    target.write_text('["kept.com"]')
    with pytest.raises(SystemExit):
        crawl_list.write_list([{"host": "a.com"}], str(target))
    assert target.read_text() == '["kept.com"]'


def test_files_are_read_as_written(tmp_path):
    backing = tmp_path / "backing.tsv"
    backing.write_text(
        "host\twikidata_type\tbenchmark_citations\tnews_list\nbbc.com\tNews\t8\t1\nx.com\t\t0\t0\n"
    )
    overrides = tmp_path / "overrides.tsv"
    overrides.write_text(
        "host\tsite_type\tnote\nkbb.com\treference\tcar values\nbay.com\tout:junk\tpiracy\n"
    )
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("*tgstat*\nShop.com\n\n")
    found = crawl_list.read_backing(str(backing))
    assert found["bbc.com"] == Backing("News", 8, True) and not found["x.com"]
    assert crawl_list.read_overrides(str(overrides)) == {
        "kbb.com": "reference",
        "bay.com": "out:junk",
    }
    hand = crawl_list.read_hand(str(excluded))
    assert (hand.exact, hand.parts) == ({"shop.com"}, ["tgstat"])
    registry = tmp_path / "registry.csv"
    registry.write_text("bbc.com,93,active,,6723761,ok,news,0.99\nnew.com,,new,,0,,,\n")
    rows = [crawl_list.parse_row(f) for f in crawl_list.registry_rows(str(registry))]
    assert rows[0] == Row("bbc.com", 93, "active", "", 6723761, "ok", "news", 0.99)
    assert rows[1] == Row("new.com", None, "new", "", 0, "", "", None)
