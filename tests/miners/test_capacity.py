import asyncio
import base64
import hashlib

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from desearch import capacity
from desearch.capacity_chain import Commitment, commitment_from
from desearch.embedding import EmbeddingClient, EmbeddingModel
from neurons.miners import capacity as miner_capacity
from neurons.miners.capacity import CapacityTester
from neurons.miners.config import Settings
from tests.local_http import serving

DIMS = 8
HOTKEY = "5Tester"
MODEL = EmbeddingModel("tiny", "tiny/tiny", "rev", DIMS, 16)


def block_hash(block: int, variant: bytes = b"") -> bytes:
    return hashlib.sha256(variant + block.to_bytes(8, "big")).digest()


def vector_of(text: str) -> np.ndarray:
    raw = hashlib.shake_256(text.encode()).digest(DIMS)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5


class Chain:
    def __init__(self, drop_commits: int = 0):
        self.head = 0
        self.final = 0
        self.hashes: dict[int, bytes] = {}
        self.commits: list[Commitment] = []
        self.drop_commits = drop_commits
        self.sent = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def block_hash(self, block: int) -> bytes:
        return self.hashes.get(block) or block_hash(block)

    async def finalized(self) -> int:
        return self.final

    async def commit(self, tag: bytes, root: bytes) -> None:
        self.sent += 1
        if self.sent > self.drop_commits:
            self.commits.append(Commitment(self.head + 1, tag, root))

    async def commitment(
        self, hotkey: str, block: int | None = None
    ) -> Commitment | None:
        found = [c for c in self.commits if block is None or c.block <= block]
        return found[-1] if found else None


class Embedder:
    batch = 4
    servers = 1

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.calls = 0
        self.events: list[str] = []

    def pause(self) -> None:
        self.events.append("pause")

    def resume(self) -> None:
        self.events.append("resume")

    async def embed_now(self, texts):
        self.calls += 1
        await asyncio.sleep(0)
        if self.calls <= self.fail_first:
            raise ValueError("the server had a bad moment")
        await asyncio.sleep(0.01)
        return [vector_of(text) for text in texts]


class Api:
    hotkey = HOTKEY

    def __init__(self, url: str = ""):
        self.url = url
        self.posts: list[dict] = []

    async def post(self, path: str, body: dict) -> dict:
        assert path == miner_capacity.REVEAL_PATH
        self.posts.append(body)
        return {"url": self.url, "content_type": "application/octet-stream"}


SETTINGS = Settings(capacity_in_flight=2, capacity_stop_blocks=2)


def make_tester(chain, embedder, api, stopping=None) -> CapacityTester:
    return CapacityTester(SETTINGS, embedder, api, chain, stopping or asyncio.Event())


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(capacity, "CANDIDATE_ONE_IN", 1)
    monkeypatch.setattr(capacity, "QUIET_BLOCKS", 2)
    monkeypatch.setattr(miner_capacity, "COMMIT_WAIT_S", 0.01)
    monkeypatch.setattr(miner_capacity, "FINALITY_POLL_S", 0.01)
    monkeypatch.setattr(miner_capacity, "REVEAL_RETRY_S", 0.01)


TEST = capacity.Test(100, block_hash(100))


async def move_head(sitter: CapacityTester, chain: Chain, block: int) -> None:
    chain.head = block
    async with sitter.head_moved:
        sitter.head = block
        sitter.head_moved.notify_all()
    await asyncio.sleep(0.03)


async def sit_through(
    sitter: CapacityTester, chain: Chain, test=TEST, until: int | None = None
) -> None:
    task = asyncio.create_task(sitter.sit(test))
    await asyncio.sleep(0.05)
    for block in range(test.block + 1, (until or test.deadline) + 1):
        await move_head(sitter, chain, block)
    chain.final = test.sample_block
    await asyncio.wait_for(task, 5)


async def test_an_honest_test_commits_and_its_reveal_passes():
    uploaded: list[bytes] = []

    async def handler(request: web.Request) -> web.Response:
        assert (
            request.method == "PUT"
            and request.content_type == "application/octet-stream"
        )
        uploaded.append(await request.read())
        return web.Response()

    chain, embedder = Chain(), Embedder()
    async with serving(handler) as url:
        api = Api(url + "/reveal")
        sitter = make_tester(chain, embedder, api)
        await sit_through(sitter, chain)
        await sitter.aclose()

    assert embedder.events == ["pause", "resume"]
    commit = chain.commits[-1]
    assert TEST.block < commit.block <= TEST.deadline
    block, prefix, count = capacity.parse_tag(commit.tag)
    assert (
        (block, prefix) == (TEST.block, TEST.seed[:4])
        and count > 0
        and count % embedder.batch == 0
    )
    assert api.posts == [{"test_block": 100}, {"test_block": 100, "uploaded": True}]

    reveal = capacity.decode_reveal(uploaded[0])
    sample_hash = block_hash(TEST.sample_block)
    opened = capacity.opened_vectors(
        TEST.seed, HOTKEY, count, commit.root, sample_hash, reveal, DIMS
    )
    assert not isinstance(opened, str)
    picked, vectors = opened
    reference = np.array(
        [vector_of(capacity.text_at(TEST.seed, HOTKEY, i)) for i in picked]
    )
    assert capacity.matches(capacity.similarities(vectors, reference))


async def test_it_stops_before_the_deadline_so_the_commitment_can_land():
    chain, embedder = Chain(), Embedder()
    sitter = make_tester(chain, embedder, Api())
    task = asyncio.create_task(sitter.embedded_until(TEST, TEST.deadline - 2))
    await asyncio.sleep(0.05)
    await move_head(sitter, chain, TEST.deadline - 3)
    assert not task.done()
    await move_head(sitter, chain, TEST.deadline - 2)
    assert await asyncio.wait_for(task, 1)


async def test_a_failed_batch_is_done_again_and_the_count_has_no_holes():
    chain, embedder = Chain(), Embedder(fail_first=1)
    sitter = make_tester(chain, embedder, Api())
    task = asyncio.create_task(sitter.embedded_until(TEST, TEST.block + 1))
    await asyncio.sleep(0.7)
    await move_head(sitter, chain, TEST.block + 1)
    raws = await asyncio.wait_for(task, 1)
    assert raws
    for index in (0, len(raws) - 1):
        assert np.frombuffer(raws[index], dtype="<f2").size == DIMS


async def test_a_dropped_commitment_is_sent_again():
    chain = Chain(drop_commits=2)
    chain.head = TEST.deadline - 2
    sitter = make_tester(chain, Embedder(), Api())
    sitter.head = chain.head
    assert await sitter.committed(TEST, capacity.make_tag(TEST, 8), b"\1" * 32)
    assert chain.sent == 3


async def test_a_commitment_that_missed_the_deadline_is_not_revealed():
    chain = Chain(drop_commits=1000)
    sitter = make_tester(chain, Embedder(), Api())
    sitter.head = TEST.deadline
    assert not await sitter.committed(TEST, capacity.make_tag(TEST, 8), b"\1" * 32)


async def test_a_test_block_that_left_the_chain_reveals_nothing():
    chain = Chain()
    chain.hashes[TEST.block] = block_hash(TEST.block, b"another fork")
    chain.final = TEST.sample_block
    api = Api()
    sitter = make_tester(chain, Embedder(), api)
    await sitter.revealed(TEST, [b"\0" * 16], [[b"\0" * 32]], b"\0" * 32, 1)
    assert api.posts == []


async def test_a_head_starts_a_test_only_after_a_quiet_run(monkeypatch):
    chain = Chain()
    started: list[capacity.Test] = []
    sitter = make_tester(chain, Embedder(), Api())

    async def sit(test):
        started.append(test)

    monkeypatch.setattr(sitter, "sit", sit)
    monkeypatch.setattr(capacity, "CANDIDATE_ONE_IN", 2)
    flags = {n: capacity.is_candidate(block_hash(n)) for n in range(3000)}
    first = next(
        n for n in range(10, 3000) if flags[n] and not flags[n - 1] and not flags[n - 2]
    )
    for block in range(first - 3, first + 1):
        await sitter.on_head(block)
    await asyncio.sleep(0)
    assert [t.block for t in started] == [first]
    assert started[0].seed == block_hash(first)


async def test_a_chain_failure_never_stops_the_tester(monkeypatch):
    stopping = asyncio.Event()
    sitter = make_tester(Chain(), Embedder(), Api(), stopping)
    follows = 0

    async def follow():
        nonlocal follows
        follows += 1
        if follows == 2:
            stopping.set()
        raise ConnectionError("the node went away")

    monkeypatch.setattr(sitter, "follow", follow)
    monkeypatch.setattr(miner_capacity, "RESTART_S", 0.01)
    await asyncio.wait_for(sitter.run(), 1)
    assert follows == 2


def test_a_stored_commitment_is_read_in_every_shape_the_chain_gives_it():
    tag, root = b"sn22c1:1:00000000:2", b"\7" * 32
    flat = {
        "block": 9,
        "info": {
            "fields": [
                {f"Raw{len(tag)}": "0x" + tag.hex()},
                {"Sha256": "0x" + root.hex()},
            ]
        },
    }
    nested = {
        "block": 9,
        "info": {"fields": [[{f"Raw{len(tag)}": tag}, {"Sha256": root}]]},
    }
    assert commitment_from(flat) == commitment_from(nested) == Commitment(9, tag, root)
    assert commitment_from(None) is None
    assert (
        commitment_from({"block": 9, "info": {"fields": [{"Raw3": "0x616263"}]}})
        is None
    )


async def test_real_tasks_wait_between_batches_while_a_test_runs():
    seen: list[list[str]] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append(body["input"])
        rows = [
            {
                "index": i,
                "embedding": base64.b64encode(
                    np.ones(DIMS, dtype=np.float32).tobytes()
                ).decode(),
            }
            for i in range(len(body["input"]))
        ]
        return web.json_response({"data": rows})

    async with serving(handler) as url:
        client = EmbeddingClient(url + "/v1/embeddings", MODEL, batch=1, in_flight=1)
        client.pause()
        real = asyncio.create_task(client.embed(["real one", "real two"]))
        await asyncio.sleep(0.05)
        assert seen == []
        await client.embed_now(["test text"])
        assert seen == [["test text"]]
        client.resume()
        assert (await asyncio.wait_for(real, 2)).shape == (2, DIMS)
        assert client.paused_since(0.0) > 0
        await client.aclose()


async def test_a_node_that_is_down_at_the_start_never_stops_the_tester(monkeypatch):
    stopping = asyncio.Event()

    class Down(Chain):
        tries = 0

        async def __aenter__(self):
            Down.tries += 1
            if Down.tries == 2:
                stopping.set()
            raise ConnectionError("no node")

    monkeypatch.setattr(miner_capacity, "RESTART_S", 0.01)
    await asyncio.wait_for(make_tester(Down(), Embedder(), Api(), stopping).run(), 1)
    assert Down.tries == 2


async def test_a_claim_close_to_expiring_keeps_going_through_a_test():
    chain, embedder = Chain(), Embedder()
    sitter = CapacityTester(
        SETTINGS, embedder, Api(), chain, asyncio.Event(), may_pause=lambda: False
    )
    await sit_through(sitter, chain, until=TEST.deadline - 2)
    assert embedder.events == []


async def test_several_servers_take_batches_by_how_busy_they_are():
    calls = {"fast": 0, "slow": 0}

    def server(name: str, delay: float):
        async def handler(request: web.Request) -> web.Response:
            body = await request.json()
            calls[name] += 1
            await asyncio.sleep(delay)
            one = base64.b64encode(np.ones(DIMS, dtype=np.float32).tobytes()).decode()
            rows = [{"index": i, "embedding": one} for i in range(len(body["input"]))]
            return web.json_response({"data": rows})

        return handler

    async with serving(server("fast", 0.001)) as fast, serving(
        server("slow", 0.05)
    ) as slow:
        client = EmbeddingClient(
            f"{fast}/v1/embeddings, {slow}/v1/embeddings", MODEL, batch=1, in_flight=1
        )
        assert client.servers == 2 and client.in_flight == 2
        await client.embed([f"text {i}" for i in range(40)])
        await client.aclose()
    assert calls["fast"] > 2 * calls["slow"] > 0


async def test_a_stalled_node_cannot_keep_real_tasks_paused(monkeypatch):
    monkeypatch.setattr(capacity, "SECONDS_PER_BLOCK", 0.01)
    chain, embedder = Chain(), Embedder()
    sitter = make_tester(chain, embedder, Api())
    # The head never moves.
    await asyncio.wait_for(sitter.sit(TEST), 2)
    assert embedder.events == ["pause", "resume"]


async def test_a_stopping_miner_lets_real_tasks_go_at_once():
    stopping = asyncio.Event()
    embedder = Embedder()
    sitter = make_tester(Chain(), embedder, Api(), stopping)
    task = asyncio.create_task(sitter.sit(TEST))
    await asyncio.sleep(0.02)
    assert embedder.events == ["pause"]
    stopping.set()
    await asyncio.wait_for(task, 1)
    assert embedder.events == ["pause", "resume"]


async def test_a_test_block_the_stream_skipped_still_starts_the_test(monkeypatch):
    chain = Chain()
    started: list[capacity.Test] = []
    sitter = make_tester(chain, Embedder(), Api())

    async def sit(test):
        started.append(test)

    monkeypatch.setattr(sitter, "sit", sit)
    monkeypatch.setattr(capacity, "CANDIDATE_ONE_IN", 2)
    flags = {n: capacity.is_candidate(block_hash(n)) for n in range(3000)}
    first = next(
        n
        for n in range(10, 3000)
        if flags[n] and not flags[n - 1] and not flags[n - 2] and not flags[n + 1]
    )
    await sitter.on_head(first - 1)
    await sitter.on_head(first + 1)
    await asyncio.sleep(0)
    assert [t.block for t in started] == [first]


async def test_stopping_zero_blocks_before_the_deadline_still_commits():
    chain, embedder = Chain(), Embedder()
    sitter = CapacityTester(
        Settings(capacity_in_flight=1, capacity_stop_blocks=0),
        embedder,
        Api(),
        chain,
        asyncio.Event(),
    )
    task = asyncio.create_task(sitter.sit(TEST))
    await asyncio.sleep(0.05)
    await move_head(sitter, chain, TEST.deadline - 1)
    for _ in range(100):
        if chain.commits:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert chain.commits


async def test_a_dead_server_among_several_does_not_fail_batches():
    async def alive(request: web.Request) -> web.Response:
        body = await request.json()
        one = base64.b64encode(np.ones(DIMS, dtype=np.float32).tobytes()).decode()
        rows = [{"index": i, "embedding": one} for i in range(len(body["input"]))]
        return web.json_response({"data": rows})

    async def dead(request: web.Request) -> web.Response:
        return web.Response(status=500)

    async with serving(dead) as bad, serving(alive) as good:
        client = EmbeddingClient(
            f"{bad}/v1/embeddings,{good}/v1/embeddings", MODEL, batch=1, in_flight=1
        )
        vectors = await client.embed([f"text {i}" for i in range(10)])
        assert vectors.shape == (10, DIMS)
        assert f"{bad}/v1/embeddings" in client.benched
        await client.aclose()


async def test_one_server_failing_still_raises_as_before():
    async def dead(request: web.Request) -> web.Response:
        return web.Response(status=500)

    async with serving(dead) as bad:
        client = EmbeddingClient(f"{bad}/v1/embeddings", MODEL, batch=1, in_flight=1)
        with pytest.raises(aiohttp.ClientError):
            await client.embed_now(["text"])
        await client.aclose()


def test_only_a_pause_during_the_task_counts_against_its_pace():
    client = EmbeddingClient("http://x/v1/embeddings", MODEL)
    client.pauses.append((10.0, 40.0))
    assert client.paused_since(50.0) == 0.0
    assert client.paused_since(30.0) == 10.0
    assert client.paused_since(0.0) == 30.0
