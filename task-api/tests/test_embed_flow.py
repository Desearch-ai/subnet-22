import asyncio
import json

from app import lifecycle, queues

from tests.test_api_flow import Harness, _upload, open_list, opened, revealed, verifier

PAGES = [
    {
        "url": f"https://site{i}.example/story/{i}",
        "host": f"site{i}.example",
        "page_key": f"pages/{i:040x}.zst",
        "content_sha1": f"{i:040x}",
    }
    for i in range(2)
]
INPUT = {
    "input_key": "embed-inputs/dt=2026-09-24/task=crawl-1.parquet",
    "input_sha256": "ab" * 32,
    "texts": 6,
    "chars": 5400,
    "pages": PAGES,
}


def embed_score(
    verdict: str, outcome: str, reason: str = "ok", returned: int = 6
) -> dict:
    return {
        "returned": returned,
        "sampled": 2,
        "matched": 2 * (outcome == "matched"),
        "mismatched": 2 * (outcome == "mismatched"),
        "min_similarity": 0.999 if outcome == "matched" else 0.12,
        "verdict": verdict,
        "reason": reason,
        "samples": [
            {"text_id": f"t{i}", "outcome": outcome, "similarity": 0.999}
            for i in range(2)
        ],
    }


async def embed_round(h: Harness, entry: dict = INPUT) -> str:
    await h.redis.rpush(queues.EMBED_INPUTS, json.dumps(entry))
    round_ = await lifecycle.open_embed_rounds(h.core)
    await revealed(h.core)
    return round_.round_id


async def embed_task(h: Harness, miner) -> dict:
    task = (await miner.post("/v1/tasks/claim", {"kind": "embed"}))["task"]
    await _upload(h, task["upload"], b"vectors")
    await miner.post(
        f"/v1/tasks/{task['task_id']}/complete",
        {"key": task["upload"]["key"], "bytes": 7},
    )
    return task


def test_a_published_page_is_embedded_credited_and_recorded(api_env, memory):
    api_env.setenv("TASK_API_EMBED_TASKS", "1")
    asyncio.run(_embedded(memory))


async def _embedded(backend) -> None:
    async with Harness(backend) as h:
        round_id = await embed_round(h)
        refused = await h.miner.post("/v1/tasks/claim")
        assert refused["refusal"]["code"] == "QUEUE_EMPTY", "crawl sees no embed work"

        task = await embed_task(h, h.miner)
        assert (task["kind"], task["round_id"]) == ("embed", round_id)
        assert (task["model"], task["texts"]) == (h.core.embed_model, 6)
        assert task["input"]["sha256"] == INPUT["input_sha256"]
        assert task["urls"] == [page["url"] for page in PAGES]

        assert [m["kind"] for m in (await open_list(h))["uploads"]] == ["embed"]
        job = await opened(h, h.validator, ("embed",))
        assert (job["kind"], job["model"]) == ("embed", h.core.embed_model)
        assert job["input"]["url"]

        scored = await h.validator.post(
            f"/v1/validation/{task['task_id']}/score", embed_score("pass", "matched")
        )
        assert (scored["verdict"], scored["credited"]) == ("pass", INPUT["chars"])

        miner = (await h.public.get(f"/v1/miners/{h.miner.hotkey}")).json()
        assert miner["pools"]["embed"]["budget"] == 2
        assert miner["pools"]["crawl"]["budget"] == 1
        shares = (await h.public.get("/v1/shares")).json()["pools"]
        assert shares == {"embed": {h.miner.hotkey: 1.0}}

        catalog = await h.core.db(h.core.embeddings.of_page, PAGES[0]["page_key"])
        assert [(row["model"], row["state"]) for row in catalog] == [
            (h.core.embed_model, "done")
        ]
        assert catalog[0]["vectors_key"].endswith(f"task={task['task_id']}.parquet")
        publish = await h.core.publish.claim(1)
        assert publish[0]["kind"] == "embed"
        assert publish[0]["vectors_key"] == catalog[0]["vectors_key"]

        assert await lifecycle.close_finished(h.core) == [round_id]
        published = (await h.public.get(f"/v1/rounds/{round_id}")).json()
        assert published["kind"] == "embed"
        assert published["manifest"][0]["input_sha256"] == INPUT["input_sha256"]
        assert (
            verifier.manifest_hash(published["manifest"], published["seed_block"])
            == published["manifest_hash"]
        )
        entries = (await h.public.get(f"/v1/rounds/{round_id}/log")).json()["entries"]
        ok, why = verifier._replay(
            published["manifest"], published["serve_order"], entries
        )
        assert ok, why


def test_the_same_page_version_is_not_embedded_twice(api_env, memory):
    api_env.setenv("TASK_API_EMBED_TASKS", "1")
    asyncio.run(_not_twice(memory))


async def _not_twice(backend) -> None:
    async with Harness(backend) as h:
        await embed_round(h)
        await h.redis.rpush(queues.EMBED_INPUTS, json.dumps(INPUT))
        assert await lifecycle.open_embed_rounds(h.core) is None
        assert await h.redis.llen(queues.EMBED_INPUTS) == 0

        changed = {**INPUT, "pages": [{**PAGES[0], "content_sha1": "f" * 40}]}
        assert await lifecycle.open_embed_rounds(h.core) is None
        await h.redis.rpush(queues.EMBED_INPUTS, json.dumps(changed))
        assert await lifecycle.open_embed_rounds(h.core) is not None


def test_a_failed_embed_task_strikes_only_the_embed_pool(api_env, memory):
    api_env.setenv("TASK_API_EMBED_TASKS", "1")
    asyncio.run(_embed_strikes(memory))


async def _embed_strikes(backend) -> None:
    async with Harness(backend) as h:
        await embed_round(h)
        task = await embed_task(h, h.miner)
        await opened(h, h.validator, ("embed",))
        scored = await h.validator.post(
            f"/v1/validation/{task['task_id']}/score",
            embed_score("fail", "mismatched", "vectors_mismatch"),
        )
        assert scored["verdict"] == "fail"

        strikes = h.core.budgets.db.execute(
            "SELECT pool, reason FROM strikes WHERE hotkey = ?", (h.miner.hotkey,)
        ).fetchall()
        assert strikes == [("embed", "vectors_mismatch")]
        again = await h.miner.post("/v1/tasks/claim", {"kind": "embed"})
        assert again["refusal"]["code"] == "ALREADY_HELD"
        rival = await h.rival.post("/v1/tasks/claim", {"kind": "embed"})
        assert rival["task"]["task_id"] == task["task_id"]


def test_embed_tasks_stay_closed_until_switched_on(api_env, memory):
    asyncio.run(_closed(memory))


async def _closed(backend) -> None:
    async with Harness(backend) as h:
        await h.redis.rpush(queues.EMBED_INPUTS, json.dumps(INPUT))
        assert await lifecycle.open_embed_rounds(h.core) is None
        assert await h.redis.llen(queues.EMBED_INPUTS) == 1, "nothing is thrown away"

        answer = await h.miner.post("/v1/tasks/claim", {"kind": "embed"})
        assert answer["refusal"] == {
            "code": "KIND_CLOSED",
            "inputs": {"kind": "embed", "retry_after": 3600.0},
        }
        assert answer["receipt"]
        health = (await h.public.get("/v1/health")).json()
        assert (health["embed_tasks"], health["embed_model"]) == (
            False,
            "qwen3-embedding-8b",
        )
