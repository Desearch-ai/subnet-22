import asyncio

import pyarrow as pa
import pyarrow.parquet as pq
from aiohttp import web

from sandbox import run, urls
from sandbox.chain import FakeChain
from sandbox.local import Ports
from tests.local_http import serving


def test_ports_derive_from_the_api_port():
    ports = Ports(19000)
    assert (ports.redis, ports.minio) == (19001, 19002)
    assert ports.api_url == "http://127.0.0.1:19000"
    assert ports.storage_url == "http://127.0.0.1:19002/sandbox-uploads"


def test_the_sandbox_api_gets_its_task_size_without_touching_the_constant():
    from sandbox import local

    command = local.api_command("python", Ports(19000), 100)
    assert command[:2] == ["python", "-c"]
    assert "rounds.TASK_URLS = 100;" in command[2] and "port=19000" in command[2]


def test_the_site_of_a_url_is_its_host_without_www():
    assert urls.host_of("https://www.TheGuardian.com/world/1") == "theguardian.com"
    assert urls.host_of("http://edition.cnn.com/x?y=1") == "edition.cnn.com"


def test_a_file_of_urls_is_parquet_or_one_url_per_line(tmp_path):
    table = tmp_path / "mine.parquet"
    pq.write_table(
        pa.table({"url": ["https://a.example/1", "https://b.example/2"]}), table
    )
    text = tmp_path / "mine.txt"
    text.write_text("https://a.example/1\n\n  https://b.example/2  \n")

    assert (
        urls.read_urls(table)
        == urls.read_urls(text)
        == [
            "https://a.example/1",
            "https://b.example/2",
        ]
    )
    source = urls.FileUrls(text)
    taken = [
        asyncio.run(source.take(1)),
        asyncio.run(source.take(5)),
        asyncio.run(source.take(5)),
    ]
    assert taken == [["https://a.example/1"], ["https://b.example/2"], []]


def test_the_dataset_is_served_shard_by_shard_and_downloaded_once(
    tmp_path, monkeypatch
):
    shards = {}
    for index in range(2):
        path = tmp_path / f"urls-{index:05d}.parquet"
        pq.write_table(
            pa.table({"url": [f"https://s{index}.example/{n}" for n in range(3)]}), path
        )
        shards[f"data/urls-{index:05d}.parquet"] = path.read_bytes()
    downloads = []

    async def handler(request):
        path = request.path
        if path == "/tree":
            listed = [{"path": name} for name in shards] + [{"path": "data/x.md"}]
            return web.json_response(listed)
        name = path.removeprefix("/resolve/")
        downloads.append(name)
        return web.Response(body=shards[name])

    async def scenario():
        async with serving(handler) as base:
            monkeypatch.setattr(urls, "TREE_URL", base + "tree")
            monkeypatch.setattr(urls, "FILE_URL", base + "resolve/")
            source = urls.DatasetUrls(tmp_path / "cache", seed=1)
            first = await source.take(4)
            second = await source.take(2)
            again = urls.DatasetUrls(tmp_path / "cache", seed=1)
            await again.take(6)
            return first, second

    first, second = asyncio.run(scenario())
    served = first + second
    assert len(served) == 6 and len(set(served)) == 6, "every URL of both shards, once"
    assert {u.split("/")[2] for u in served} == {"s0.example", "s1.example"}
    assert sorted(downloads) == [
        "data/urls-00000.parquet",
        "data/urls-00001.parquet",
    ], "downloaded once, then read from the cache"


def test_a_verdict_line_names_the_pages_that_did_not_match():
    task = {
        "task_id": "t1",
        "miner": "5GYoQ4rynWSzEBVTwKypeiK5AjhuB7KtSZwTbPgydvNqe8EY",
        "verdict": "pass",
        "reason": "ok",
        "credited": 35,
        "returned": 40,
        "matched": 7,
        "mismatched": 1,
        "unverifiable": 2,
    }
    listed = [
        {"url": "https://a/1", "sampled": True, "outcome": "matched"},
        {
            "url": "https://a/2",
            "sampled": True,
            "outcome": "mismatched",
            "why": "text differs",
        },
        {"url": "https://a/3", "sampled": False, "outcome": None},
        {"url": "https://a/5", "sampled": True, "outcome": "errors_confirmed"},
        {"url": "https://a/4", "sampled": True, "outcome": "matched", "rejected": True},
    ]
    lines = run.describe(task, listed).splitlines()
    assert lines[0].startswith("task t1 miner 5GYoQ4ry: PASS (ok), paid 35 of 40 rows")
    assert lines[1:] == [
        "  https://a/2: mismatched, text differs",
        "  https://a/4: matched",
    ]


def test_a_verdict_names_where_the_upload_and_the_full_result_are():
    task = {
        "task_id": "t2",
        "miner": "5GYoQ4rynWSzEBVTwKypeiK5AjhuB7KtSZwTbPgydvNqe8EY",
        "verdict": "fail",
        "reason": "content_mismatch",
        "credited": 0,
        "returned": 1000,
        "matched": 1,
        "mismatched": 7,
        "unverifiable": 0,
    }
    lines = run.describe(
        task, [], upload="http://127.0.0.1:18082/u/t2.parquet", details="/runs/t2.json"
    ).splitlines()
    assert lines[1:] == [
        "  upload:  http://127.0.0.1:18082/u/t2.parquet",
        "  details: /runs/t2.json",
    ]


def test_new_log_lines_are_echoed_once_with_the_process_name(tmp_path, capsys):
    sandbox = run.Sandbox(Ports(19000), tmp_path, source=None, task_size=100)
    log = tmp_path / "logs" / "validator.log"
    log.write_text("task=a verdict=pass\nx | INFO | Blocks left until next epoch: 3\n")
    sandbox.echo_logs()
    with log.open("a") as more:
        more.write("task=b verdict=fail\n\n")
    sandbox.echo_logs()
    sandbox.echo_logs()
    assert capsys.readouterr().out.splitlines() == [
        "validator | task=a verdict=pass",
        "validator | task=b verdict=fail",
    ]


def test_the_summary_rate_counts_only_work_checked_after_the_first_verdict():
    stats = run.MinerStats(first_at=0.0)
    stats.add({"verdict": "pass", "returned": 1000, "credited": 900})
    assert "rate after the next task" in stats.line("5GYoQ4rynWSz", 3, now=1.0)

    stats.add({"verdict": "fail", "returned": 1000, "credited": 0})
    assert stats.line("5GYoQ4rynWSz", budget=3, now=120.0) == (
        "  5GYoQ4ry: 2 tasks checked, 1 passed, 2000 pages returned (500/min),"
        " 900 rows paid (45%), budget 3"
    )


def test_the_fake_chain_ticks_on_a_clock_and_seeds_from_the_block_number():
    chain = FakeChain(["burn", "validator"], genesis=0.0, block_seconds=1.0, tempo=60)

    async def scenario():
        block = await chain.block()
        return (
            block,
            (await chain.block_info(block)).hash,
            await chain.uid("validator", 22),
            await chain.uid("nobody", 22),
            await chain.blocks_until_next_epoch(22),
            (await chain.execute(None, None)).success,
        )

    block, block_hash, uid, nobody, left, ok = asyncio.run(scenario())
    assert block > 1_000_000 and block_hash == f"local:{block}"
    assert (uid, nobody, ok) == (1, None, True) and 0 <= left <= 60


def test_the_sandbox_will_not_start_without_a_scrapingdog_key(tmp_path, monkeypatch):
    from sandbox import __main__ as entry
    from sandbox import local

    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(
        local, "ENV_FILES", (tmp_path / "validator.env", tmp_path / "miner.env")
    )
    assert entry.main(["--runs", str(tmp_path / "runs")]) == 1
    assert not (tmp_path / "runs").exists(), "nothing started"

    (tmp_path / "miner.env").write_text("SCRAPINGDOG_API_KEY=from-the-miner-env\n")
    assert entry.load_scrapingdog_key()
    assert entry.os.environ["SCRAPINGDOG_API_KEY"] == "from-the-miner-env"
