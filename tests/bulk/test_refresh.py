import json
import sys
import threading
import time
from datetime import datetime, timezone

import pyarrow as pa
import pytest

from bulk import refresh, refresh_hidden
from bulk.rows import MANIFEST, Batches
from desearch.extraction.schema import PAGE_SCHEMA

DAY1 = datetime(2026, 11, 9, 2, tzinfo=timezone.utc)
DAY2 = datetime(2026, 11, 16, 2, tzinfo=timezone.utc)
BUILD = refresh.Build("subnets", ())


def blank(field: pa.Field):
    if pa.types.is_list(field.type):
        return []
    if pa.types.is_integer(field.type):
        return 200 if field.name == "status" else 0
    if pa.types.is_timestamp(field.type):
        return DAY1
    if pa.types.is_boolean(field.type):
        return None
    if pa.types.is_large_binary(field.type):
        return None
    return None if field.nullable else ""


def row(url: str, text: str, fetched_at=DAY1) -> dict:
    out = {f.name: blank(f) for f in PAGE_SCHEMA}
    out.update(url=url, final_url=url, text=text, title=url, fetched_at=fetched_at)
    return out


def run_folder(path, pages: dict[str, str], as_of=DAY1, removed=()):
    batches = Batches(path, "subnets", as_of.strftime("%Y%m%d"), as_of)
    for url, text in pages.items():
        batches.add(row(url, text, as_of))
    for url in removed:
        batches.remove(url)
    batches.close({"pages": len(pages)})
    return path


def published(folder) -> list[str]:
    manifest = json.loads((folder / MANIFEST).read_text())
    return [r["url"] for r in refresh.rows_of(folder)] if manifest["files"] else []


def urls(n: int) -> dict[str, str]:
    return {f"https://taostats.io/subnets/{i}": f"subnet {i}" for i in range(n)}


def test_a_seeded_source_publishes_only_new_and_changed_pages(tmp_path):
    con = refresh.open_state(tmp_path / "s.sqlite")
    first = urls(40)
    assert refresh.seed(con, BUILD, [run_folder(tmp_path / "r1", first)]) == 40

    second = {
        **first,
        "https://taostats.io/subnets/3": "subnet 3, new owner",
        "https://taostats.io/subnets/40": "new",
    }
    out = tmp_path / "out"
    counts = refresh.diff(
        con, BUILD, run_folder(tmp_path / "r2", second, DAY2), out, "r2", DAY2
    )

    assert (
        counts.seen,
        counts.new,
        counts.changed,
        counts.unchanged,
        counts.removed,
    ) == (41, 1, 1, 39, 0)
    assert sorted(published(out)) == [
        "https://taostats.io/subnets/3",
        "https://taostats.io/subnets/40",
    ]
    assert not counts.large


def test_pages_gone_from_a_full_list_are_hidden_and_come_back(tmp_path):
    con = refresh.open_state(tmp_path / "s.sqlite")
    refresh.seed(con, BUILD, [run_folder(tmp_path / "r1", urls(40))])

    less = urls(40)
    del less["https://taostats.io/subnets/7"]
    counts = refresh.diff(
        con, BUILD, run_folder(tmp_path / "r2", less, DAY2), tmp_path / "o2", "r2", DAY2
    )
    refresh.keep(con, BUILD, counts, "r2", DAY2)
    assert (
        counts.gone == ["https://taostats.io/subnets/7"]
        and published(tmp_path / "o2") == []
    )
    assert refresh.hidden_urls(con) == ["https://taostats.io/subnets/7"]

    counts = refresh.diff(
        con,
        BUILD,
        run_folder(tmp_path / "r3", urls(40), DAY2),
        tmp_path / "o3",
        "r3",
        DAY2,
    )
    refresh.keep(con, BUILD, counts, "r3", DAY2)
    assert published(tmp_path / "o3") == ["https://taostats.io/subnets/7"]
    assert refresh.hidden_urls(con) == []


def test_a_short_fetch_is_refused(tmp_path):
    con = refresh.open_state(tmp_path / "s.sqlite")
    refresh.seed(con, BUILD, [run_folder(tmp_path / "r1", urls(40))])
    with pytest.raises(refresh.Refused):
        refresh.diff(
            con,
            BUILD,
            run_folder(tmp_path / "r2", urls(37), DAY2),
            tmp_path / "o",
            "r2",
            DAY2,
        )


def test_a_large_change_is_flagged_and_removed_txt_counts(tmp_path):
    con = refresh.open_state(tmp_path / "s.sqlite")
    refresh.seed(con, BUILD, [run_folder(tmp_path / "r1", urls(10))])
    changed = {u: t + " edited" for u, t in urls(10).items()}
    counts = refresh.diff(
        con,
        refresh.Build("subnets", (), full_list=False),
        run_folder(
            tmp_path / "r2", changed, DAY2, removed=["https://taostats.io/subnets/99"]
        ),
        tmp_path / "o",
        "r2",
        DAY2,
    )
    assert counts.large and counts.changed == 10
    assert counts.gone == ["https://taostats.io/subnets/99"]


def test_refresh_runs_the_commands_publishes_and_hides(tmp_path, monkeypatch):
    src = run_folder(tmp_path / "prepared", urls(3))
    copy = (
        "{python}",
        "-c",
        "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2])",
        str(src),
        "{out}",
    )
    source = refresh.Source("test", fetch=(), builds=(refresh.Build("subnets", copy),))
    sent, hidden = [], []
    monkeypatch.setattr(
        refresh, "publish", lambda folder, root: sent.append(published(folder)) or "rc=0"
    )
    monkeypatch.setattr(refresh, "hide", lambda name, u: hidden.append((name, u)))

    report = refresh.refresh(source, tmp_path / "home", now=DAY2)
    assert report["builds"]["subnets"]["new"] == 3 and len(sent) == 1
    assert hidden == [("test", [])]

    again = refresh.refresh(
        source, tmp_path / "home", now=datetime(2026, 11, 23, tzinfo=timezone.utc)
    )
    assert again["builds"]["subnets"]["unchanged"] == 3 and len(sent) == 1


def test_dry_run_publishes_and_records_nothing(tmp_path, monkeypatch):
    src = run_folder(tmp_path / "prepared", urls(3))
    copy = (
        "{python}",
        "-c",
        "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2])",
        str(src),
        "{out}",
    )
    source = refresh.Source("test", fetch=(), builds=(refresh.Build("subnets", copy),))
    monkeypatch.setattr(
        refresh, "publish", lambda folder, root: pytest.fail("published on a dry run")
    )
    monkeypatch.setattr(
        refresh, "hide", lambda name, u: pytest.fail("hid on a dry run")
    )
    assert (
        refresh.refresh(source, tmp_path / "home", dry_run=True, now=DAY2)["builds"][
            "subnets"
        ]["new"]
        == 3
    )
    con = refresh.open_state(tmp_path / "home" / "state" / "test.sqlite")
    assert con.execute("SELECT count(*) FROM pages").fetchone() == (0,)


def test_hidden_section_is_rewritten_whole_and_escaped():
    text = "# old rules\nexample.com/junk/*\n"
    once = refresh_hidden.rewrite(
        text, "bittensor", ["https://taostats.io/subnets/7", "https://a.io/x[1]*", "https://a.io/q?id=1"]
    )
    assert once == (
        "# old rules\nexample.com/junk/*\n# refresh bittensor begin\na.io/x[[]1][*]\n"
        "taostats.io/subnets/7\n# refresh bittensor end\n"
    )
    assert refresh_hidden.rewrite(once, "bittensor", []) == text
    assert refresh_hidden.rule("https://a.io/p#frag") == "a.io/p"


def test_publish_hands_the_run_to_the_outbox_and_reads_its_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "RECEIPT_POLL_S", 0.01)
    folder = run_folder(tmp_path / "work" / "refresh-subnets-20261116-0200", urls(2))
    name = folder.name

    def planner():
        while not (tmp_path / "outbox" / f"{name}.ready").exists():
            time.sleep(0.01)
        (tmp_path / "receipts").mkdir(exist_ok=True)
        (tmp_path / "receipts" / f"{name}.txt").write_text("rc=0\n1 files, 2 pages, bulk files [4700]\n0\n")

    threading.Thread(target=planner).start()
    assert refresh.publish(folder, tmp_path) == "1 files, 2 pages, bulk files [4700]"
    assert (tmp_path / "outbox" / name / MANIFEST).exists() and not folder.exists()


def test_a_failed_publish_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "RECEIPT_POLL_S", 0.01)
    folder = run_folder(tmp_path / "work" / "refresh-subnets-x", urls(1))
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "refresh-subnets-x.txt").write_text("rc=1\nerror: no room\n")
    with pytest.raises(RuntimeError, match="no room"):
        refresh.publish(folder, tmp_path)


def test_an_incremental_source_fetches_from_the_last_published_run(tmp_path, monkeypatch):
    src = run_folder(tmp_path / "prepared", urls(2))
    record = ("{python}", "-c", "import sys; open(sys.argv[1], 'a').write(sys.argv[2] + ' ' + sys.argv[3] + '\\n')", str(tmp_path / "calls"), "{since}", "{until}")
    copy = ("{python}", "-c", "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2])", str(src) + "/../prepared", "{out}")
    source = refresh.Source("inc", fetch=(record,), builds=(refresh.Build("arxiv", copy, full_list=False),))
    monkeypatch.setattr(refresh, "publish", lambda folder, root: "rc=0")
    monkeypatch.setattr(refresh, "hide", lambda name, u: None)
    home = tmp_path / "home"

    with pytest.raises(refresh.Refused):
        refresh.refresh(source, home, now=DAY2)
    report = refresh.refresh(source, home, now=DAY2, since="2026-11-10")
    (home / refresh.STATUS).write_text(json.dumps({**report, "ok": True}) + "\n")
    refresh.refresh(source, home, now=datetime(2026, 11, 17, tzinfo=timezone.utc))
    assert (tmp_path / "calls").read_text() == "2026-11-10 2026-11-16\n2026-11-16 2026-11-17\n"


def test_wildcard_paths_expand_to_the_files_they_match(tmp_path):
    for name in ("b.xml", "a.xml", "c.txt"):
        (tmp_path / name).write_text("")
    assert refresh.expand("{work}/*.xml", {"work": str(tmp_path)}) == [str(tmp_path / "a.xml"), str(tmp_path / "b.xml")]
    assert refresh.expand("{python}", {}) == [sys.executable]


def test_the_discord_line_says_what_was_published_or_why_not():
    ok = {
        "source": "bittensor", "run": "20261116-0200", "ok": True, "hidden": 1, "seconds": 812,
        "builds": {"subnets": {"new": 2, "changed": 5, "removed": 1, "seen": 130, "published": "1 files, 7 pages, bulk files [4701]", "large": False}},
    }
    assert refresh.summary(ok) == (
        "refresh bittensor 20261116-0200: subnets 2 new, 5 changed, 1 removed of 130, "
        "1 files, 7 pages, bulk files [4701]; 1 hidden; 812 s"
    )
    bad = {"source": "arxiv", "ok": False, "refused": True, "error": "Refused('short')", "seconds": 3}
    assert refresh.summary(bad) == "⚠️ refresh arxiv refused after 3 s: Refused('short')"


def test_a_source_runs_in_its_own_folder_with_its_own_python(tmp_path):
    log = tmp_path / "log.txt"
    source = refresh.Source("x", (), (), cwd=str(tmp_path), python=sys.executable, memory_bytes=8 << 30)
    refresh.run_command(("{python}", "-c", "import os; print(os.getcwd())"), {}, log, source)
    assert log.read_text().splitlines()[-1] == str(tmp_path.resolve())
