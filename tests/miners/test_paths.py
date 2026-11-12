from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from aiohttp import web
from bittensor.wallets import Keypair

from desearch import witness
from desearch.client import TaskApiError
from neurons.miners.config import Settings
from neurons.miners.miner import Miner, task_api
from neurons.miners.paths import MultiPathClient, Path, keep_witness
from tests.local_http import serving

KEY = Keypair.create_from_uri("//paths-test-miner")
RELAY = Keypair.create_from_uri("//paths-test-relay")


class Fake:
    """A client for one path: answers, or fails with a status, and counts its calls."""

    def __init__(self, status: int = 200):
        self.status = status
        self.calls = 0

    async def post(self, path: str, body: dict | None = None) -> dict:
        self.calls += 1
        if self.status != 200:
            raise TaskApiError(self.status, "no")
        return {"ok": True}

    async def aclose(self) -> None:
        pass


def client(*statuses: int, rng=None) -> MultiPathClient:
    made = MultiPathClient(
        KEY,
        "http://direct.example",
        [f"http://relay{i}.example" for i in range(len(statuses) - 1)],
        rng=rng,
    )
    made.paths = [Path(p.name, Fake(status)) for p, status in zip(made.paths, statuses)]
    return made


class InOrder(random.Random):
    def shuffle(self, x, random=None):
        return None


async def test_a_path_that_does_not_answer_is_passed_over_and_rests():
    paths = client(0, 503, 200, rng=InOrder())
    assert await paths.post("/v1/tasks/claim") == {"ok": True}
    assert [p.client.calls for p in paths.paths] == [1, 1, 1]
    assert [bool(p.failures) for p in paths.paths] == [True, True, False]
    await paths.post("/v1/tasks/claim")
    assert [p.client.calls for p in paths.paths] == [1, 1, 2], "resting paths go last"


async def test_an_answer_from_the_api_itself_is_not_a_path_failing():
    paths = client(409, 200, rng=InOrder())
    with pytest.raises(TaskApiError) as refused:
        await paths.post("/v1/tasks/t/complete")
    assert refused.value.status == 409 and paths.paths[1].client.calls == 0


async def test_when_every_path_fails_the_last_failure_is_raised():
    paths = client(0, 502, rng=InOrder())
    with pytest.raises(TaskApiError) as failed:
        await paths.post("/v1/tasks/claim")
    assert failed.value.status == 502


async def test_requests_spread_over_every_working_path_at_random():
    paths = client(200, 200, 200, rng=random.Random(7))
    for _ in range(300):
        await paths.post("/v1/tasks/claim")
    used = Counter(p.client.calls for p in paths.paths)
    assert all(70 < p.client.calls < 130 for p in paths.paths), used


def test_a_miner_given_relays_reaches_the_api_over_all_of_them(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "keypair", lambda self: KEY)
    settings = Settings(
        relay_urls=("http://a.example", "http://b.example"),
        witnesses_file=str(tmp_path / "w.jsonl"),
    )
    made = task_api(settings)
    assert made.hotkey == KEY.ss58_address
    assert [p.name for p in made.paths] == [
        "direct",
        "relay http://a.example",
        "relay http://b.example",
    ]
    relays_only = Settings(relay_urls=("http://a.example",), direct_path=False)
    assert [p.name for p in task_api(relays_only).paths] == ["relay http://a.example"]
    env = Settings.from_env(
        {"RELAY_URLS": "http://a.example, http://b.example", "DIRECT_PATH": "0"}
    )
    assert (
        env.relay_urls == ("http://a.example", "http://b.example")
        and env.direct_path is False
    )


def test_the_witnesses_relays_sign_for_the_miner_are_kept(tmp_path):
    kept = tmp_path / "witnesses.jsonl"
    record = witness.sign(RELAY, {"kind": witness.STORAGE, "key": "k", "at": 1.0})
    keep_witness(str(kept))(record)
    assert json.loads(kept.read_text()) == record


async def test_the_completion_pins_the_upload_by_what_its_answer_named():
    async def stored(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(headers={"ETag": '"abc"', "x-amz-version-id": "3HL4kqtJ"})

    async def unversioned(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(headers={"ETag": '"abc"', "x-amz-version-id": "null"})

    miner = Miner(Settings(), api=Fake())
    try:
        async with serving(stored) as url:
            pins = await miner.upload(
                {"url": f"{url}put", "content_type": "application/octet-stream"},
                b"PAR1",
            )
        assert pins == {"etag": '"abc"', "version": "3HL4kqtJ"}
        async with serving(unversioned) as url:
            pins = await miner.upload(
                {"url": f"{url}put", "content_type": "application/octet-stream"},
                b"PAR1",
            )
        assert pins == {"etag": '"abc"'}, "a bucket without versions names none"
    finally:
        await miner.aclose()


async def test_a_completion_refused_for_a_swapped_upload_uploads_once_more():
    class Api:
        hotkey = KEY.ss58_address

        def __init__(self):
            self.reports = []

        async def post(self, path: str, body: dict | None = None) -> dict:
            self.reports.append(body)
            if len(self.reports) == 1:
                raise TaskApiError(
                    409,
                    "the issued key holds another object than the one you uploaded; upload again and complete",
                )
            return {"status": "open_for_validation"}

        async def aclose(self) -> None:
            pass

    uploads = []

    async def stored(request: web.Request) -> web.Response:
        uploads.append(await request.read())
        return web.Response(headers={"ETag": f'"e{len(uploads)}"'})

    api = Api()
    miner = Miner(Settings(), api=api)
    try:
        async with serving(stored) as url:
            upload = {
                "url": f"{url}put",
                "content_type": "application/octet-stream",
                "key": "uploads/k.parquet",
            }
            await miner.deliver(
                "t", upload, b"PAR1", {"key": "uploads/k.parquet", "rows": 1}
            )
    finally:
        await miner.aclose()
    assert len(uploads) == 2
    assert [r["etag"] for r in api.reports] == ['"e1"', '"e2"']
