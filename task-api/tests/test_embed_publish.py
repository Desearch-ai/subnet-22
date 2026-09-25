import asyncio
import hashlib
import json

import numpy as np
from app import lifecycle, queues
from publisher.worker import VECTORS_SCHEMA, Publisher, embed_texts

from desearch.embedding import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    encode_vector,
    read_parquet,
    write_parquet,
)
from tests.test_api_flow import Harness, _score, _upload, opened, revealed
from tests.test_embed_flow import embed_score


def test_a_page_becomes_a_head_a_full_text_and_its_passages():
    paragraph = (
        "A paragraph of the story that is long enough to be its own passage. " * 2
    )
    text = "\n".join([paragraph] * 3)
    change = {
        "key": "pages/abc.zst",
        "url": "https://a.example/story",
        "content_sha1": "c" * 40,
        "title": "A story",
        "text": text,
    }

    rows = embed_texts(change)

    assert [(row["kind"], row["index"]) for row in rows] == [
        ("head", 0),
        ("full", 0),
        ("chunk", 0),
        ("chunk", 1),
        ("chunk", 2),
    ]
    assert rows[0]["text"].startswith("A story\n")
    assert {row["text_id"] for row in rows} == {
        "pages/abc.zst#head0",
        "pages/abc.zst#full0",
        "pages/abc.zst#chunk0",
        "pages/abc.zst#chunk1",
        "pages/abc.zst#chunk2",
    }


def test_published_pages_are_embedded_and_their_vectors_published(api_env, memory):
    api_env.setenv("TASK_API_EMBED_TASKS", "1")
    asyncio.run(_crawl_to_vectors(memory))


def test_with_embedding_off_the_publisher_prepares_nothing_to_embed(api_env, memory):
    asyncio.run(_publish_without_embedding(memory))


async def _publish_without_embedding(backend) -> None:
    async with Harness(backend) as h:
        await h.enqueue()
        task = await h.mine()
        await opened(h, h.validator)
        await h.validator.post(
            f"/v1/validation/{task['task_id']}/score",
            _score("pass", 3, url=task["urls"][0]),
        )
        publisher = Publisher(h.core.publish, h.core.storage, h.core.pages, workers=2)
        try:
            assert await publisher.run_once() == 1
        finally:
            publisher.close()
        assert await h.redis.llen(queues.EMBED_INPUTS) == 0


def _read(storage, key: str) -> bytes:
    return storage.client.get_object(Bucket=storage.bucket, Key=storage.path(key))[
        "Body"
    ].read()


async def _crawl_to_vectors(backend) -> None:
    async with Harness(backend) as h:
        await h.enqueue()
        task = await h.mine()
        await opened(h, h.validator)
        await h.validator.post(
            f"/v1/validation/{task['task_id']}/score",
            _score("pass", 3, url=task["urls"][0]),
        )
        publisher = Publisher(
            h.core.publish, h.core.storage, h.core.pages, workers=2, embed_inputs=True
        )
        try:
            assert await publisher.run_once() == 1
            (entry,) = map(json.loads, await h.redis.lrange(queues.EMBED_INPUTS, 0, -1))
            assert sorted(page["url"] for page in entry["pages"]) == sorted(
                task["urls"]
            )
            given = _read(h.core.storage, entry["input_key"])
            assert hashlib.sha256(given).hexdigest() == entry["input_sha256"]
            rows = read_parquet(given, INPUT_SCHEMA)
            assert entry["texts"] == len(rows) == 3 * 3, (
                "head, full and one passage each"
            )
            assert entry["chars"] == sum(len(row["text"]) for row in rows)

            await lifecycle.open_embed_rounds(h.core)
            await revealed(h.core)
            embed = (await h.miner.post("/v1/tasks/claim", {"kind": "embed"}))["task"]
            downloaded = await h.r2.get(embed["input"]["url"])
            assert downloaded.body == given
            rng = np.random.default_rng(7)
            vectors = {
                row["text_id"]: encode_vector(rng.normal(size=4096)) for row in rows
            }
            body = write_parquet(
                [{"text_id": t, "vector": v} for t, v in vectors.items()], OUTPUT_SCHEMA
            )
            await _upload(h, embed["upload"], body)
            await h.miner.post(
                f"/v1/tasks/{embed['task_id']}/complete",
                {"key": embed["upload"]["key"], "bytes": len(body)},
            )
            await opened(h, h.validator, ("embed",))
            await h.validator.post(
                f"/v1/validation/{embed['task_id']}/score",
                embed_score("pass", "matched", returned=embed["texts"]),
            )

            assert await publisher.run_once() == 1
        finally:
            publisher.close()

        page_key = entry["pages"][0]["page_key"]
        (catalog,) = await h.core.db(h.core.embeddings.of_page, page_key)
        published = read_parquet(
            _read(h.core.pages, catalog["vectors_key"]), VECTORS_SCHEMA
        )
        assert len(published) == len(rows)
        assert {row["model"] for row in published} == {h.core.embed_model}
        assert all(row["vector"] == vectors[row["text_id"]] for row in published)
        assert all(row["text"] for row in published)
        assert await h.size(entry["input_key"]) is None
        assert await h.size(embed["upload"]["key"]) is None
