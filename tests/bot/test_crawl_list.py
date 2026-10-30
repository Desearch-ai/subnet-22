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
    url_count=1000,
    canonical_host="",
):
    return Row(
        host,
        rank,
        state,
        state_reason,
        url_count,
        homepage,
        site_type,
        confidence,
        canonical_host,
    )


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


def test_a_recheck_overrules_only_the_homepage_checks():
    def reason(rechecked=True, **fields):
        found = decide(row(**fields), Backing(), rechecked=rechecked)
        return "listed" if found.listed else found.reason

    assert (
        reason(state="ineligible", state_reason="not_english", homepage="") == "listed"
    )
    assert reason(homepage="not_english", state="no_sitemap") == "listed"
    assert (
        reason(state="ineligible", state_reason="no_text", homepage="blank") == "listed"
    )
    assert (
        reason(
            rechecked=False,
            state="ineligible",
            state_reason="no_text",
            homepage="blank",
        )
        == "ineligible:no_text"
    )
    assert (
        reason(
            state="ineligible", state_reason="homepage_error", homepage="homepage_403"
        )
        == "refuses"
    ), "the type model's own refusal still holds"
    assert reason(state="ineligible", state_reason="bot_wall") == "refuses"
    assert reason(state="blocked", state_reason="robots_disallow") == "refuses"
    assert (
        reason(state="excluded", state_reason="non_english_tld") == "non_english_tld"
    ), "a registry exclusion isn't a homepage check"
    assert reason(state="ineligible", state_reason="adult") == "ineligible:adult"
    assert reason(state="unreachable") == "state:unreachable"


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
    if site_type == "news":
        assert (found.cap, found.send_cap, found.backlog) == (cap, None, 3 * cap)
    else:
        assert (found.cap, found.send_cap, found.backlog) == (cap, 3 * cap, None)


@pytest.mark.parametrize(
    "site_type, judged, cap",
    [
        ("reference", None, 1_000_000),
        ("reference", crawl_list.Judged(20, 17), 5_000_000),
        ("reference", crawl_list.Judged(20, 12), 1_000_000),
        ("reference", crawl_list.Judged(20, 8), 400_000),
        ("news", crawl_list.Judged(20, 5), 250_000),
        ("reference", crawl_list.Judged(20, 1), 50_000),
        ("reference", crawl_list.Judged(6, 6), 1_000_000),
        ("news", crawl_list.Judged(12, 12), 5_000_000),
        ("personal", crawl_list.Judged(20, 20), 10_000),
        ("personal", crawl_list.Judged(20, 2), 10_000),
        ("community", crawl_list.Judged(20, 0), 10_000),
    ],
)
def test_judged_sites_earn_or_lose_cap(site_type, judged, cap):
    found = decide(
        row(site_type=site_type, rank=5_000),
        Backing(news_list=True, benchmark_citations=1),
        judged=judged,
    )
    assert found.cap == cap
    assert (found.backlog if site_type == "news" else found.send_cap) == 3 * cap


def test_top_ranked_sites_take_more_an_hour():
    hourly = [decide(row(rank=r), Backing()).hourly for r in (5_000, 50_000, None)]
    assert hourly == [20_000, 5_000, 5_000]


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
        "hourly": 20_000,
        "backlog": 3_000_000,
    }
    reference = crawl_list.build(
        [row(host="r.com", site_type="reference")], {}, {}, Hand()
    )[0][0]
    assert reference == {
        "host": "r.com",
        "type": "reference",
        "weight": 3,
        "hourly": 5_000,
        "send_cap": 300_000,
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
    judged = tmp_path / "judged.tsv"
    judged.write_text(
        "host\ttype\tjudged\tsubstantive\tthin\nuniprot.org\treference\t20\t19\t1\n"
    )
    assert crawl_list.read_judged(str(judged)) == {
        "uniprot.org": crawl_list.Judged(20, 19)
    }
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "bbc.com,93,active,,6723761,ok,news,0.99,\nnew.com,,new,,0,,,,\nold.com,9,redirects,redirect,40,,,,new.com\n"
    )
    rows = [crawl_list.parse_row(f) for f in crawl_list.registry_rows(str(registry))]
    assert rows[0] == Row("bbc.com", 93, "active", "", 6723761, "ok", "news", 0.99)
    assert rows[1] == Row("new.com", None, "new", "", 0, "", "", None)
    assert rows[2].canonical_host == "new.com"
    assert (
        crawl_list.parse_row(
            ["old.com", "", "active", "", "0", "", "", ""]
        ).canonical_host
        == ""
    )
    rechecked = tmp_path / "recheck.tsv"
    rechecked.write_text(
        "host\tverdict\twhy\ncnn.com\tok\t\nnytimes.com\twalled\thomepage_403\nlater.com\tok\t\nlater.com\twalled\thomepage_403\n"
    )
    assert crawl_list.read_rechecked(str(rechecked)) == {"cnn.com"}


def test_the_registry_password_never_reaches_psqls_command_line(monkeypatch):
    seen = {}

    class Process:
        stdout = iter(["a.com,1,active,,10,ok,news,0.9,\n"])

    def popen(command, env=None, **kwargs):
        seen.update(command=command, env=env)
        return Process()

    monkeypatch.setattr(crawl_list.subprocess, "Popen", popen)
    monkeypatch.setenv(
        "DESEARCH_DB",
        "postgresql://postgres:s%40cret-pw@db.example.co:5432/postgres?sslmode=require",
    )
    assert [r[0] for r in crawl_list.registry_rows(None)] == ["a.com"]
    assert not any("cret-pw" in arg for arg in seen["command"])
    assert "DESEARCH_DB" not in seen["env"]
    assert {
        k: seen["env"][k]
        for k in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE", "PGSSLMODE")
    } == {
        "PGHOST": "db.example.co",
        "PGPORT": "5432",
        "PGUSER": "postgres",
        "PGPASSWORD": "s@cret-pw",
        "PGDATABASE": "postgres",
        "PGSSLMODE": "require",
    }
    assert crawl_list.pg_env("host=db.example.co password='a b' dbname=bot") == {
        "PGHOST": "db.example.co",
        "PGPASSWORD": "a b",
        "PGDATABASE": "bot",
    }
    with pytest.raises(SystemExit):
        crawl_list.pg_env("postgresql://u:p@h/db?krbsrvname=x")


def v1_list(rows, v1, overrides=None):
    entries, counts = crawl_list.build(rows, {}, overrides or {}, Hand(), v1=v1)
    return {e["host"]: e for e in entries}, counts


def test_v1_sites_send_first_and_the_rest_fill_at_lower_weights():
    V1 = crawl_list.V1
    rows = [
        row(host="news.com", site_type="news"),
        row(host="uni.com", site_type="academic"),
        row(host="ref.com", site_type="reference"),
        row(host="mag.com", site_type="media"),
        row(host="corp.com", site_type="company"),
    ]
    v1 = {
        "news.com": V1("news", "news", since="2020-01-01"),
        "uni.com": V1("whole", "academic"),
    }
    listed, _ = v1_list(rows, v1)
    assert listed["news.com"] == {
        "host": "news.com",
        "type": "news",
        "weight": 4,
        "hourly": 5_000,
        "backlog": 1_500_000,
        "since": "2020-01-01",
    }
    assert (listed["uni.com"]["weight"], listed["uni.com"]["send_cap"]) == (
        3,
        crawl_list.V1_WHOLE_CAP * 3,
    )
    assert [listed[h]["weight"] for h in ("ref.com", "mag.com", "corp.com")] == [
        2,
        2,
        1,
    ]


def test_v1_news_and_whole_sites_in_the_waves_so_far_carry_the_resend_cutoff():
    V1 = crawl_list.V1
    rows = [
        row(host="news.com", site_type="news"),
        row(host="uni.com", site_type="academic"),
        row(host="later.com", site_type="academic"),
        row(host="corp.com", site_type="company"),
    ]
    v1 = {
        "news.com": V1("news", "news", since="2025-10-01"),
        "uni.com": V1("whole", "academic"),
        "later.com": V1("whole", "academic"),
    }
    entries, _ = crawl_list.build(
        rows,
        {},
        {},
        Hand(),
        v1=v1,
        resend_before="2026-10-27",
        resend_hosts={"news.com", "uni.com", "corp.com"},
    )
    listed = {e["host"]: e for e in entries}
    assert (
        listed["news.com"]["resend_before"]
        == listed["uni.com"]["resend_before"]
        == "2026-10-27"
    )
    assert "resend_before" not in listed["later.com"], "a site in a later wave waits"
    assert "resend_before" not in listed["corp.com"], "node-2 sites aren't sent again"
    assert all("resend_before" not in e for e in v1_list(rows, v1)[0].values()), (
        "only when asked"
    )


def test_node2_sites_spread_their_stored_pages_over_the_hours_given():
    V1 = crawl_list.V1
    rows = [
        row(host="news.com", site_type="news", url_count=9_000_000),
        row(host="corp.com", site_type="company", url_count=1000),
        row(host="tiny.com", site_type="company", url_count=10),
        row(host="big.com", site_type="media", url_count=90_000_000),
    ]
    v1 = {"news.com": V1("news", "news")}
    entries, _ = crawl_list.build(rows, {}, {}, Hand(), v1=v1, node2_hours=60)
    listed = {e["host"]: e for e in entries}
    assert listed["news.com"]["hourly"] == 5_000, "v1 sites keep their hourly limit"
    assert listed["corp.com"]["hourly"] == 17
    assert listed["tiny.com"]["hourly"] == 1, "at least one an hour"
    assert listed["big.com"]["hourly"] == min(
        5_000, -(-listed["big.com"]["send_cap"] // 60)
    ), "stored pages count up to the cap"
    assert {e["host"]: e["hourly"] for e in v1_list(rows, v1)[0].values()}[
        "corp.com"
    ] == 5_000, "only when asked"


def test_a_minimum_weight_leaves_lighter_node2_sites_off_the_list():
    V1 = crawl_list.V1
    rows = [
        row(host="news.com", site_type="news"),
        row(host="mag.com", site_type="media"),
        row(host="corp.com", site_type="company"),
        row(host="own.com", site_type="company"),
    ]
    v1 = {"news.com": V1("news", "news"), "own.com": V1("whole", "company")}
    entries, counts = crawl_list.build(rows, {}, {}, Hand(), v1=v1, node2_min_weight=2)
    assert {e["host"] for e in entries} == {"news.com", "mag.com", "own.com"}, (
        "a v1 site stays whatever its type"
    )
    assert counts["out:node2"] == 1
    assert "corp.com" in v1_list(rows, v1)[0], "only when asked"


def test_hosts_to_rejoin_take_a_cap_one_up():
    rows = [
        row(host="corp.com", site_type="company"),
        row(host="other.com", site_type="company"),
        row(host="news.com", site_type="news"),
    ]
    plain = {e["host"]: e for e in crawl_list.build(rows, {}, {}, Hand())[0]}
    nudged = {
        e["host"]: e
        for e in crawl_list.build(
            rows, {}, {}, Hand(), rejoin_hosts={"corp.com", "news.com"}
        )[0]
    }
    assert nudged["corp.com"]["send_cap"] == plain["corp.com"]["send_cap"] + 1
    assert nudged["other.com"] == plain["other.com"]
    assert nudged["news.com"] == plain["news.com"], (
        "a site without a cap has none to change"
    )


def test_host_files_that_match_no_listed_site_leave_the_list(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(crawl_list, "MIN_DOMAINS", 1)
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "news.com,10,active,,1000,ok,news,0.99,\ncorp.com,20,active,,1000,ok,company,0.99,\n"
    )
    v1 = tmp_path / "v1-set.json"
    v1.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "host": "news.com",
                        "rule": "news",
                        "since": "2025-10-01",
                        "type": "news",
                    }
                ]
            }
        )
    )
    out = tmp_path / "domains.json"
    out.write_text('["kept.com"]')
    files = {
        name: tmp_path / f"{name}.txt" for name in ("waves", "rejoin", "empty", "stray")
    }
    for name, text in {
        "waves": "news.com\n",
        "rejoin": "corp.com\n",
        "empty": "\n",
        "stray": "gone.com\n",
    }.items():
        files[name].write_text(text)
    base = [
        "--out",
        str(out),
        "--registry-csv",
        str(registry),
        "--v1-set",
        str(v1),
        "--resend-before",
        "2026-10-27",
    ]
    for bad in (
        ["--resend-hosts", files["empty"]],
        ["--resend-hosts", files["stray"]],
        ["--resend-hosts", files["waves"], "--rejoin-hosts", files["stray"]],
    ):
        with pytest.raises(SystemExit):
            crawl_list.main(base + [str(arg) for arg in bad])
        assert out.read_text() == '["kept.com"]'
    crawl_list.main(
        base
        + [
            "--resend-hosts",
            str(files["waves"]),
            "--rejoin-hosts",
            str(files["rejoin"]),
        ]
    )
    assert (
        "1 with resend_before, 1 with since, 1 with send_cap one up to rejoin"
        in capsys.readouterr().out
    )
    assert {e["host"] for e in json.loads(out.read_text())} == {"news.com", "corp.com"}


def test_v1_takes_cited_sites_whole_and_drops_its_outs():
    V1 = crawl_list.V1
    rows = [
        row(host="huge.com", site_type="news"),
        row(host="paper.com", site_type="news", rank=500),
        row(host="corp.com", site_type="company"),
    ]
    v1 = {
        "huge.com": V1("out"),
        "paper.com": V1("news", "news"),
        "corp.com": V1("whole", "company"),
    }
    listed, _ = v1_list(rows, v1)
    assert "huge.com" not in listed, "a site the set holds out stops sending"
    assert (
        "send_cap" not in listed["paper.com"]
        and listed["paper.com"]["hourly"] == 20_000
    )
    assert "since" not in listed["paper.com"], (
        "a news site taken for all years has no date floor"
    )
    assert (listed["corp.com"]["weight"], listed["corp.com"]["send_cap"]) == (
        3,
        crawl_list.V1_WHOLE_CAP * 3,
    )


def test_v1_sites_without_a_sitemap_are_listed_for_their_feed():
    V1 = crawl_list.V1
    rows = [
        row(host="feedless.com", state="no_sitemap", site_type="academic"),
        row(host="plain.com", state="no_sitemap", site_type="academic"),
    ]
    rows += [
        row(host="news.parent.com", state="no_sitemap", site_type="news"),
        row(host="parent.com", site_type="news"),
    ]
    v1 = {
        "feedless.com": V1("whole", "academic"),
        "news.parent.com": V1("news", "news"),
        "parent.com": V1("news", "news"),
    }
    listed, _ = v1_list(rows, v1)
    assert set(listed) == {"feedless.com", "parent.com"}, (
        "a subdomain of a listed site isn't listed for its feed alone: the parent's pages under it would stop"
    )


def test_v1_lifts_only_label_outs():
    V1 = crawl_list.V1
    rows = [
        row(host="label.com", site_type="commerce"),
        row(host="judged.com", site_type="news"),
        row(host="refuses.com", homepage="homepage_403"),
        row(host="foreign.com", state_reason="not_english"),
        row(host="mirror.com", site_type="reference"),
    ]
    v1 = {
        h: V1("whole", "news")
        for h in ("label.com", "judged.com", "refuses.com", "foreign.com")
    }
    v1["mirror.com"] = V1("out")
    listed, counts = v1_list(rows, v1, overrides={"judged.com": "out:junk"})
    assert set(listed) == {"label.com"}
    assert counts["out:v1"] == 1 and counts["out:junk"] == 1


def test_a_rechecked_v1_site_is_listed_once_it_sends_and_its_redirect_is_kept():
    V1 = crawl_list.V1
    rows = [
        row(host="outlet.com", homepage="not_english", site_type=""),
        row(host="feedless.org", state="no_sitemap", homepage="not_english"),
        row(host="waiting.com", state="ineligible", state_reason="not_english"),
        row(host="foreign.com", homepage="not_english", site_type=""),
        row(host="old.com", state="redirects", canonical_host="new.com"),
    ]
    v1 = {
        "outlet.com": V1("news", "news"),
        "feedless.org": V1("whole", "reference"),
        "waiting.com": V1("news", "news"),
        "foreign.com": V1("news", "news"),
    }
    rechecked = {"outlet.com", "feedless.org", "waiting.com"}
    decisions = []

    class Writer:
        writerow = decisions.append

    entries, counts = crawl_list.build(
        rows, {}, {}, Hand(), Writer(), v1=v1, rechecked=rechecked
    )
    assert {e["host"] for e in entries} == {"outlet.com", "feedless.org"}, (
        "an ineligible site sends nothing until the bot reads it again"
    )
    assert counts["out:not_english"] == 1
    by_host = {d[0]: d for d in decisions}
    assert by_host["waiting.com"][3:6] == [1, "", "news"]
    assert by_host["old.com"][-1] == "new.com"


def test_the_v1_set_is_read_from_its_file(tmp_path):
    path = tmp_path / "v1-set.json"
    path.write_text(
        json.dumps(
            {
                "built": "2026-10-28T06:00:00Z",
                "sites": [
                    {
                        "host": "a.com",
                        "rule": "news",
                        "since": "2020-01-01",
                        "type": "news",
                        "why": ["news"],
                    },
                    {
                        "host": "b.com",
                        "rule": "whole",
                        "paths": ["/guide/"],
                        "type": "docs",
                        "why": ["docs"],
                    },
                    {"host": "c.com", "rule": "out"},
                ],
            }
        )
    )
    assert crawl_list.read_v1_set(str(path)) == {
        "a.com": crawl_list.V1("news", "news", since="2020-01-01"),
        "b.com": crawl_list.V1("whole", "docs"),
        "c.com": crawl_list.V1("out"),
    }
    assert crawl_list.read_v1_set(None) is None
