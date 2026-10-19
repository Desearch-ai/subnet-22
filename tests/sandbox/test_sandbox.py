import asyncio

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aiohttp import web

from sandbox import local, run, urls
from sandbox.chain import FakeChain
from sandbox.local import Ports
from tests.local_http import serving


def test_ports_derive_from_the_api_port():
    ports = Ports(19000)
    assert (ports.redis, ports.minio, ports.reference) == (19001, 19002, 19003)
    assert ports.api_url == "http://127.0.0.1:19000"
    assert ports.storage_url == "http://127.0.0.1:19002/sandbox-uploads"
    assert ports.reference_url == "http://127.0.0.1:19003/v1/embeddings"


def test_the_api_listens_on_its_port_inside_and_reaches_redis_and_storage_on_localhost():
    found = local.api_env(Ports(19000), genesis=5.0, task_size=100)
    assert found["TASK_API_LISTEN"] == "0.0.0.0:19000"
    assert found["TASK_API_REDIS"] == "redis://127.0.0.1:19001/0"
    assert found["CF_R2_ENDPOINT"] == "http://127.0.0.1:19002"
    assert (found["TASK_API_DATA"], found["TASK_API_TASK_URLS"]) == ("/data", "100")
    assert (found["TASK_API_REGISTRY"], found["TASK_API_SEEDS"]) == ("local", "local")
    assert (found["TASK_API_GENESIS"], found["TASK_API_CHECK_SHARE"]) == ("5.0", "1")
    assert found["TASK_API_EMBED_CHECK_SHARE"] == "1", (
        "every embed upload is checked too"
    )
    assert found["TASK_API_EMBED_TASKS"] == "0"
    assert (
        local.api_env(Ports(19000), 5.0, 100, embed=True)["TASK_API_EMBED_TASKS"] == "1"
    )


def test_the_image_is_named_for_its_sources_and_ignores_build_output(tmp_path):
    for name, body in {
        "crates/desearch/src/lib.rs": "pub fn a() {}",
        "crates/desearch/target/release/junk": "x",
        "task-api/Cargo.toml": "[package]",
        "task-api/Cargo.lock": "",
        "task-api/src/main.rs": "fn main() {}",
        "task-api/app/main.py": "",
    }.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    tag = local.image_tag(tmp_path)
    assert tag.startswith("desearch-sandbox-task-api:") and len(tag.split(":")[1]) == 12

    (tmp_path / "crates/desearch/target/release/junk").write_text("y")
    (tmp_path / "task-api/app/main.py").write_text("changed")
    assert local.image_tag(tmp_path) == tag, (
        "build output and other folders do not count"
    )
    (tmp_path / "task-api/src/main.rs").write_text("fn main() { }")
    assert local.image_tag(tmp_path) != tag


def test_every_container_shares_the_storage_network_and_has_a_memory_cap(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr(
        local.subprocess, "run", lambda command, **_: commands.append(command)
    )
    local.start_storage(Ports(19000), tmp_path)
    local.start_api("image:tag", {"TASK_API_LISTEN": "0.0.0.0:19000"}, tmp_path / "api")
    runs = {c[c.index("--name") + 1]: c for c in commands if c[:2] == ["docker", "run"]}
    assert set(runs) == {local.MINIO, local.REDIS, local.API}
    assert all("--memory" in command for command in runs.values())
    minio = runs[local.MINIO]
    published = [minio[i + 1] for i, flag in enumerate(minio) if flag == "-p"]
    assert published == [f"127.0.0.1:{p}:{p}" for p in (19000, 19001, 19002)]
    assert minio[-4:] == ["/data", "--address", ":19002", "--quiet"]
    for name in (local.REDIS, local.API):
        assert f"container:{local.MINIO}" in runs[name]
        assert "-p" not in runs[name]
    assert "--port" in runs[local.REDIS] and "19001" in runs[local.REDIS]
    api = runs[local.API]
    assert api[api.index("--user") + 1] == f"{local.os.getuid()}:{local.os.getgid()}"
    assert "127.0.0.1:19000/v1/ping" in api[api.index("--health-cmd") + 1]
    assert f"{tmp_path / 'api'}:/data" in api
    assert "TASK_API_LISTEN=0.0.0.0:19000" in api and api[-1] == "image:tag"


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


def test_a_verdict_on_an_upload_changed_after_its_completion_says_so():
    task = {
        "task_id": "t3",
        "miner": "5GYoQ4rynWSzEBVTwKypeiK5AjhuB7KtSZwTbPgydvNqe8EY",
        "verdict": "fail",
        "reason": "upload_changed",
        "credited": 0,
        "returned": 0,
        "matched": 0,
        "mismatched": 0,
        "unverifiable": 0,
    }
    lines = run.describe(task, []).splitlines()
    assert lines == [
        "task t3 miner 5GYoQ4ry: FAIL (upload_changed),"
        " the upload changed after it was completed"
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


def test_more_tasks_are_queued_when_the_miner_is_only_offered_tasks_it_held(tmp_path):
    import time

    import aiohttp

    refusals = []

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"events": refusals})

    async def held(base: str) -> bool:
        sandbox = run.Sandbox(
            Ports(int(base.rsplit(":", 1)[1].strip("/"))), tmp_path, None, 100
        )
        async with aiohttp.ClientSession() as http:
            return await sandbox.only_held_tasks_left(http)

    async def scenario() -> list[bool]:
        async with serving(handler) as base:
            answers = [await held(base)]
            refusals[:] = [{"at": time.time(), "refusal": {"code": "ALREADY_HELD"}}]
            answers.append(await held(base))
            refusals[0]["at"] = time.time() - run.HELD_RECENTLY_S - 1
            answers.append(await held(base))
            refusals[:] = [{"at": time.time(), "refusal": {"code": "NO_CAPACITY"}}]
            answers.append(await held(base))
        return answers

    assert asyncio.run(scenario()) == [False, True, False, False]


def embed_set(folder, pages: int = 3, texts_per_page: int = 2, dims: int = 2560):
    """A test set in the published layout: rows grouped by page, each with its reference vector."""
    import json

    import numpy as np

    from desearch.embedding import INPUT_SCHEMA

    rows = []
    for page in range(pages):
        for index in range(texts_per_page):
            vector = np.zeros(dims, dtype="<f4")
            vector[page * texts_per_page + index] = 1.0
            rows.append(
                {
                    "text_id": f"t{page}-{index}",
                    "page_key": f"p{page}",
                    "url": f"https://www.site{page}.example/a",
                    "content_sha1": f"c{page}",
                    "kind": "page" if index == 0 else "passage",
                    "index": index,
                    "text": f"text {index} of page {page} " + "x" * (10 * page),
                    "reference": vector.tobytes(),
                }
            )
    schema = pa.schema([*INPUT_SCHEMA, pa.field("reference", pa.binary())])
    (folder / "data").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), folder / "data/texts-00000.parquet"
    )
    manifest = {
        "model": "desearch-4b-v1",
        "dims": dims,
        "pages": pages,
        "files": ["data/texts-00000.parquet"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return rows


def test_the_embed_set_is_read_by_page_with_a_reference_for_each_text(tmp_path):
    from sandbox.embed import EmbedSet

    rows = embed_set(tmp_path)
    texts = EmbedSet.read(tmp_path)
    assert [[r["text_id"] for r in page] for page in texts.pages] == [
        ["t0-0", "t0-1"],
        ["t1-0", "t1-1"],
        ["t2-0", "t2-1"],
    ]
    assert "reference" not in texts.pages[0][0]
    assert texts.reference(rows[3]["text"]).argmax() == 3
    assert texts.reference("a text the set does not have") is None


def test_inputs_hold_whole_pages_up_to_the_task_size_and_the_character_cap(tmp_path):
    from sandbox.embed import EmbedSet

    embed_set(tmp_path, pages=5)
    texts = EmbedSet.read(tmp_path)
    sized = texts.inputs(task_size=2)
    assert [len(given.pages) for given in sized] == [2, 2, 1]
    assert [given.texts for given in sized] == [4, 4, 2]
    assert sized[0].chars == sum(len(r["text"]) for r in sized[0].rows())

    page_chars = [sum(len(r["text"]) for r in page) for page in texts.pages]
    capped = texts.inputs(task_size=100, max_chars=page_chars[0] + page_chars[1])
    assert [len(given.pages) for given in capped][:2] == [2, 1]
    assert all(
        given.chars <= page_chars[0] + page_chars[1] or len(given.pages) == 1
        for given in capped
    ), "a page is never split, so one over the cap goes alone"


def test_an_input_is_the_exact_input_columns_and_its_entry_is_the_planners(tmp_path):
    import hashlib
    import re

    from desearch.embedding import INPUT_SCHEMA, read_parquet
    from sandbox.embed import EmbedSet, input_body, input_entry, input_key

    embed_set(tmp_path)
    given = EmbedSet.read(tmp_path).inputs(task_size=2)[0]
    body = input_body(given)
    assert pq.read_table(pa.BufferReader(body)).column_names == INPUT_SCHEMA.names
    assert read_parquet(body, INPUT_SCHEMA) == given.rows()

    key = input_key(86_400.0 * 3)
    assert re.fullmatch(r"embed/inputs/dt=1970-01-04/[0-9a-f]{32}\.parquet", key)
    assert input_key(0.0) != input_key(0.0), "a new key each time"
    entry = input_entry(given, key, body, now=12.5)
    assert entry == {
        "input_key": key,
        "input_sha256": hashlib.sha256(body).hexdigest(),
        "written_at": 12.5,
        "texts": 4,
        "chars": given.chars,
        "pages": [
            {
                "page_key": "p0",
                "content_sha1": "c0",
                "url": "https://www.site0.example/a",
                "host": "site0.example",
            },
            {
                "page_key": "p1",
                "content_sha1": "c1",
                "url": "https://www.site1.example/a",
                "host": "site1.example",
            },
        ],
    }


def test_the_reference_server_answers_the_validators_client_with_the_stored_vectors(
    tmp_path,
):
    import aiohttp
    import numpy as np

    from desearch.embedding import MODELS, EmbeddingClient
    from sandbox.embed import EmbedSet, serve_reference

    rows = embed_set(tmp_path)
    texts = EmbedSet.read(tmp_path)

    async def scenario():
        runner = await serve_reference(texts, 0)
        port = runner.addresses[0][1]
        url = f"http://127.0.0.1:{port}/v1/embeddings"
        client = EmbeddingClient(url, MODELS["desearch-4b-v1"])
        try:
            served = await client.served_models()
            vectors = await client.embed([rows[4]["text"], rows[1]["text"]])
            async with aiohttp.ClientSession() as http:
                async with http.get(f"http://127.0.0.1:{port}/v1/models") as r:
                    listed = (await r.json())["data"][0]
                answers = []
                for body in (
                    {"model": "desearch-4b-v1", "input": rows[2]["text"]},
                    {"model": "desearch-4b-v1", "input": ["never seen"]},
                    {"model": "another-model", "input": [rows[0]["text"]]},
                    {
                        "model": "desearch-4b-v1",
                        "input": [rows[0]["text"]],
                        "encoding_format": "int8",
                    },
                ):
                    async with http.post(url, json=body) as r:
                        answers.append((r.status, await r.json()))
        finally:
            await client.aclose()
            await runner.cleanup()
        return served, vectors, listed, answers

    served, vectors, listed, answers = asyncio.run(scenario())
    assert served == ["desearch-4b-v1"] and listed["max_model_len"] == 2048
    assert vectors.shape == (2, 2560) and list(vectors.argmax(axis=1)) == [4, 1]
    (ok, floats), (unknown, _), (other, _), (encoding, _) = answers
    assert ok == 200 and np.argmax(floats["data"][0]["embedding"]) == 2, (
        "floats unless asked for base64"
    )
    assert (unknown, other, encoding) == (400, 404, 400)


EMBED_TASK = {
    "task_id": "e1",
    "miner": "5GYoQ4rynWSzEBVTwKypeiK5AjhuB7KtSZwTbPgydvNqe8EY",
    "verdict": "fail",
    "reason": "vectors_mismatch",
    "sampled": 48,
    "matched": 46,
    "mismatched": 2,
    "claimed_at": 100.0,
    "completed_at": 141.25,
}


def test_an_embed_verdict_shows_the_sample_the_size_the_time_and_each_mismatch():
    from sandbox import embed

    report = {
        "min_similarity": 0.91,
        "samples": [
            {"text_id": "a", "outcome": "matched", "similarity": 1.0},
            {
                "text_id": "b",
                "outcome": "mismatched",
                "similarity": 0.98,
                "url": "https://w/B",
                "kind": "passage",
                "position": 3,
            },
            {
                "text_id": "c",
                "outcome": "mismatched",
                "similarity": 0.91,
                "url": "https://w/C",
                "kind": "page",
                "position": 0,
            },
        ],
    }
    lines = embed.describe(
        EMBED_TASK,
        report,
        1043,
        812331,
        upload="http://u/e1.parquet",
        details="/runs/e1.json",
    ).splitlines()
    assert lines == [
        "task e1 miner 5GYoQ4ry: FAIL (vectors_mismatch), 46/48 sampled texts matched, lowest similarity 0.91",
        "  1,043 texts, 812,331 characters, uploaded 41.2s after the claim (limit 300s)",
        "  https://w/C page 0: similarity 0.910000",
        "  https://w/B passage 3: similarity 0.980000",
        "  upload:  http://u/e1.parquet",
        "  details: /runs/e1.json",
    ]

    malformed = {**EMBED_TASK, "reason": "vectors_malformed", "matched": 47}
    lines = embed.describe(
        malformed,
        {
            "malformed": 1,
            "min_similarity": 0.9999,
            "samples": [
                {"text_id": "d", "outcome": "malformed", "similarity": None, "index": 7}
            ],
        },
        1043,
        None,
    ).splitlines()
    assert lines[0].endswith(
        "FAIL (vectors_malformed), 47/48 sampled texts matched, 1 malformed,"
        " lowest similarity 0.9999"
    )
    assert lines[2] == "  d: not a finite vector of length 1"

    wrong = {**EMBED_TASK, "reason": "wrong_size", "sampled": 0, "matched": 0}
    head = embed.describe(wrong, {"sampled": 0, "samples": []}, 1043, None).splitlines()
    assert head[0].endswith(
        "FAIL (wrong_size), the upload's size is not one vector for each text"
    )
    assert head[1].startswith("  1,043 texts, uploaded 41.2s")

    changed = {**EMBED_TASK, "reason": "upload_changed", "sampled": 0, "matched": 0}
    head = embed.describe(changed, {"sampled": 0, "samples": []}, 1043, None)
    assert head.splitlines()[0].endswith(
        "FAIL (upload_changed), the upload changed after it was completed"
    )


def test_a_sampled_text_keeps_its_row_in_the_upload_and_gains_its_place_in_the_page(
    tmp_path,
):
    from sandbox.embed import EmbedSandbox, EmbedSet

    rows = embed_set(tmp_path)
    sandbox = object.__new__(EmbedSandbox)
    sandbox.texts = EmbedSet.read(tmp_path)
    sample = {"text_id": rows[1]["text_id"], "index": 9, "outcome": "malformed"}

    found = {**sample, **sandbox.text_of(sample)}

    assert found["index"] == 9 and found["position"] == rows[1]["index"]
    assert found["text"] == rows[1]["text"]


def test_the_embed_summary_rates_count_only_work_checked_after_the_first_verdict():
    from sandbox.embed import EmbedStats

    stats = EmbedStats(first_at=0.0)
    stats.add("pass", texts=1000, chars=800_000)
    assert "rate after the next task" in stats.line("5GYoQ4rynWSz", 3, now=1.0)
    stats.add("fail", texts=1000, chars=700_000)
    assert stats.line("5GYoQ4rynWSz", budget=3, now=120.0) == (
        "  5GYoQ4ry: 2 tasks checked, 1 passed, 2,000 texts (500/min),"
        " 1,500,000 characters (350,000/min), budget 3"
    )


def test_in_embed_mode_the_validator_sets_weights_without_a_scrapingdog_key():
    from types import SimpleNamespace

    from sandbox.validator import SandboxValidator

    validator = object.__new__(SandboxValidator)
    validator.scrapingdog_key = ""
    validator.config = SimpleNamespace(
        neuron=SimpleNamespace(storage_url="http://s", disable_set_weights=False)
    )
    validator.stopping = asyncio.Event()
    validator.stopping.set()

    validator.embed = False
    assert "SCRAPINGDOG_API_KEY" in validator.crawl_disabled_reason()
    assert not validator.should_set_weights()

    validator.embed = True
    assert validator.crawl_disabled_reason() is None
    assert validator.should_set_weights()
    asyncio.run(validator.check_crawl_tasks())
    assert not hasattr(validator, "crawl_checker"), "no crawl checker in embed mode"


def test_the_sandbox_validator_checks_only_its_own_mode_s_tasks():
    from sandbox.validator import SandboxValidator

    validator = object.__new__(SandboxValidator)
    validator.stopping = asyncio.Event()
    validator.stopping.set()
    validator.embed = False
    assert asyncio.run(validator.check_embed_tasks()) is None, (
        "crawl mode waits instead of checking embed tasks"
    )


def test_paid_miners_join_the_pretend_chain_so_their_share_reaches_the_weights():
    from types import SimpleNamespace

    from neurons.validators.weights import (
        CRAWL_PERC,
        EMBED_PERC,
        EMISSION_CONTROL_HOTKEY,
        EMISSION_CONTROL_PERC,
    )
    from sandbox.validator import SandboxValidator

    def paid(amount: float) -> SimpleNamespace:
        return SimpleNamespace(amount=amount, uploads=1, checked=1, failed=0, rate=1.0)

    class Ledger:
        def crawl_paid(self):
            return {"crawler": paid(100.0)}

        def embed_paid(self):
            return {"embedder": paid(5000.0)}

    chain = FakeChain(
        [EMISSION_CONTROL_HOTKEY, "validator"], genesis=0.0, block_seconds=1.0, tempo=60
    )
    validator = object.__new__(SandboxValidator)
    validator.chain, validator.ledger = chain, Ledger()
    validator.config = SimpleNamespace(netuid=22)

    weights = asyncio.run(validator.weights())

    assert chain.hotkeys == [
        EMISSION_CONTROL_HOTKEY,
        "validator",
        "crawler",
        "embedder",
    ]
    miners = 1.0 - EMISSION_CONTROL_PERC
    assert weights[0] == pytest.approx(EMISSION_CONTROL_PERC)
    assert weights[2] == pytest.approx(miners * CRAWL_PERC) and weights[
        3
    ] == pytest.approx(miners * EMBED_PERC)
