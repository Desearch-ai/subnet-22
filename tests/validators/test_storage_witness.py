"""Uploads pinned by version, and what every validator signs it saw in storage.

The scenario: once a miner's completion is recorded, the bucket's owner writes something else over its key.
Read by entity tag, validators got 412 and failed the miner (`upload_changed`); read by the version the
completion pinned, they get the miner's file, and every validator's own look at storage is on record.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import AsyncExitStack

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from bittensor.wallets import Keypair

from desearch import witness
from desearch.manifest import payload, verify
from desearch.witness import STORAGE
from neurons.validators.crawl import CrawlValidator
from neurons.validators.tasks import (
    DownloadFailed,
    UploadChanged,
    by_version,
    upload_url,
)
from neurons.validators.upload_log import UploadLog
from neurons.validators.witness import ChainHead, StorageWitness, WitnessStore
from tests.validators.test_worklist import API_KEY, SIGNER, manifest

VALIDATOR = Keypair.create_from_uri("//storage-witness-validator")
KEY = "uploads/0123456789abcdef0123456789abcdef.parquet"


def etag_of(data: bytes) -> str:
    return f'"{hashlib.md5(data).hexdigest()}"'


class VersionedBucket:
    """A public bucket with versioning on: every write a new version, reads by version id or of the current one."""

    def __init__(self, versioned: bool = True, version_reads: bool = True):
        self.versions: dict[str, list[tuple[str, bytes]]] = {}
        self.versioned = versioned
        self.version_reads = version_reads
        self.reads: list[dict] = []

    def put(self, key: str, data: bytes) -> str:
        made = self.versions.setdefault(key, [])
        version = f"v{len(made) + 1}"
        if not self.versioned:
            made.clear()
        made.append((version, data))
        return version

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{key:.*}", self.handle)
        return app

    async def handle(self, request: web.Request) -> web.Response:
        key = request.match_info["key"]
        self.reads.append(
            {
                "method": request.method,
                "query": dict(request.query),
                "if_match": request.headers.get("If-Match"),
            }
        )
        held = self.versions.get(key) or []
        wanted = request.query.get("versionId")
        if wanted and not self.version_reads:
            return web.Response(status=403)
        found = (
            next((v for v in held if v[0] == wanted), None)
            if wanted
            else (held[-1] if held else None)
        )
        if found is None:
            return web.Response(status=404)
        version, data = found
        if request.headers.get("If-Match") and request.headers["If-Match"] != etag_of(
            data
        ):
            return web.Response(status=412)
        headers = {
            "ETag": etag_of(data),
            "Last-Modified": "Wed, 04 Nov 2026 10:00:00 GMT",
        }
        if self.versioned:
            headers |= {
                "x-amz-version-id": version,
                "x-amz-object-lock-mode": "COMPLIANCE",
                "x-amz-object-lock-retain-until-date": "2026-11-05T10:00:00.000Z",
            }
        if request.method == "HEAD":
            return web.Response(headers={**headers, "Content-Length": str(len(data))})
        if ranged := request.headers.get("Range", "").removeprefix("bytes="):
            start, end = (int(n) for n in ranged.split("-"))
            end = min(end, len(data) - 1)
            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return web.Response(status=206, body=data[start : end + 1], headers=headers)
        return web.Response(body=data, headers=headers)


async def serve(stack: AsyncExitStack, bucket: VersionedBucket) -> str:
    server = TestServer(bucket.app(), host="127.0.0.1")
    await server.start_server()
    stack.push_async_callback(server.close)
    return str(server.make_url("")).rstrip("/")


def test_a_note_without_a_version_signs_the_same_bytes_as_before():
    plain = manifest("t")
    assert b"version" not in payload(plain) and verify(plain, SIGNER)
    pinned = manifest("t", version="v1")
    assert b'"version":"v1"' in payload(pinned) and verify(pinned, SIGNER)
    assert not verify({**pinned, "version": "v2"}, SIGNER), "the version is signed"


def test_an_upload_pinned_by_version_is_read_by_that_version():
    url = upload_url("https://files.example", manifest("t", version="3/HL4kqtJ+"))
    assert (
        url == f"https://files.example/{manifest('t')['key']}?versionId=3%2FHL4kqtJ%2B"
    )
    assert by_version(url)
    assert not by_version(upload_url("https://files.example", manifest("t")))


async def test_an_overwrite_after_the_completion_cannot_fail_an_upload_pinned_by_version():
    mine, theirs = b"PAR1 the miner's rows PAR1", b"PAR1 junk PAR1"
    async with AsyncExitStack() as stack:
        bucket = VersionedBucket()
        storage = await serve(stack, bucket)
        http = await stack.enter_async_context(aiohttp.ClientSession())
        pinned_version = bucket.put(KEY, mine)
        bucket.put(KEY, theirs)
        checker = CrawlValidator(None, None, http, storage_url=storage, signer=SIGNER)

        note = manifest("t", key=KEY, etag=etag_of(mine), version=pinned_version)
        job = checker.job_of(note, seed="s")
        assert await checker.download(job["download_url"], "t", job["etag"]) == mine
        body, total = await checker.read_range(
            job["download_url"], 0, 4, "t", job["etag"]
        )
        assert (body, total) == (b"PAR1", len(mine))
        assert all(
            read["if_match"] is None for read in bucket.reads
        ), "a version read needs no entity tag"

        # The same overwrite, before versions: the miner's pinned entity tag is gone and its check fails.
        unpinned = checker.job_of(manifest("t", key=KEY, etag=etag_of(mine)), seed="s")
        with pytest.raises(UploadChanged):
            await checker.download(unpinned["download_url"], "t", unpinned["etag"])


async def test_a_bucket_that_refuses_reads_by_version_is_storage_trouble_not_the_miners():
    async with AsyncExitStack() as stack:
        bucket = VersionedBucket(version_reads=False)
        storage = await serve(stack, bucket)
        http = await stack.enter_async_context(aiohttp.ClientSession())
        version = bucket.put(KEY, b"PAR1 rows PAR1")
        checker = CrawlValidator(None, None, http, storage_url=storage, signer=SIGNER)
        job = checker.job_of(
            manifest("t", key=KEY, etag='"e"', version=version), seed="s"
        )
        with pytest.raises(DownloadFailed):
            await checker.download(job["download_url"], "t", job["etag"])


async def observed(
    bucket: VersionedBucket, entries: list[dict], tmp_path
) -> tuple[list[dict], int]:
    async with AsyncExitStack() as stack:
        storage = await serve(stack, bucket)
        http = await stack.enter_async_context(aiohttp.ClientSession())
        store = WitnessStore(str(tmp_path / "witness.db"))
        stack.callback(store.close)
        head = ChainHead(lambda: asyncio.sleep(0, 42))
        await head.refresh()
        looking = StorageWitness(http, storage, VALIDATOR, store, head)
        changed = await looking.observe(entries)
        return store.records(STORAGE), changed


async def test_every_validator_signs_what_storage_held_for_each_completed_upload(
    tmp_path,
):
    bucket = VersionedBucket()
    kept_version = bucket.put(KEY, b"PAR1 kept PAR1")
    bucket.put(KEY, b"PAR1 written over PAR1")
    entries = [
        {
            "task_id": "t",
            "miner": "m",
            "key": KEY,
            "etag": etag_of(b"PAR1 kept PAR1"),
            "version": kept_version,
        },
        {
            "task_id": "u",
            "miner": "m",
            "key": "uploads/gone.parquet",
            "etag": '"x"',
            "version": "v1",
        },
    ]
    records, changed = await observed(bucket, entries, tmp_path)
    by_task = {r["task_id"]: r for r in records}
    assert changed == 1
    kept = by_task["t"]
    assert (
        witness.verify(kept)
        and kept["witness"] == VALIDATOR.ss58_address
        and kept["block"] == 42
    )
    assert (kept["status"], kept["matches"], kept["object_version"]) == (
        200,
        True,
        kept_version,
    )
    assert kept["lock_mode"] == "COMPLIANCE" and kept["lock_until"] > kept["at"] - 86400
    assert (by_task["u"]["status"], by_task["u"]["matches"]) == (404, False)


async def test_on_a_bucket_without_versions_an_overwrite_is_on_record(tmp_path):
    bucket = VersionedBucket(versioned=False)
    bucket.put(KEY, b"PAR1 the miner's PAR1")
    pinned = etag_of(b"PAR1 the miner's PAR1")
    bucket.put(KEY, b"PAR1 someone else's PAR1")
    records, changed = await observed(
        bucket, [{"task_id": "t", "miner": "m", "key": KEY, "etag": pinned}], tmp_path
    )
    assert (
        changed == 1 and records[0]["matches"] is False and records[0]["etag"] != pinned
    )


class Ledger:
    def __init__(self):
        self.kept: dict[str, str] = {}

    def state(self, name):
        return self.kept.get(name)

    def set_state(self, name, value):
        self.kept[name] = value

    def add_uploads(self, entries):
        return len(entries)


class Watching:
    def __init__(self):
        self.seen: list[dict] = []

    async def observe(self, entries):
        self.seen.extend(entries)


async def test_the_upload_log_hands_every_new_upload_to_the_storage_witness():
    log_files = {
        "log/uploads/latest.json": {"seq": 2},
        "log/uploads/seq/000000000002.json": {
            "seq": 2,
            "written_at": 1.0,
            "entries": [{"key": KEY, "task_id": "t"}],
        },
    }
    for body in log_files.values():
        if "entries" in body:
            body["signer"] = SIGNER
            body["signature"] = API_KEY.sign(
                witness.canonical({k: v for k, v in body.items() if k != "signature"})
            ).hex()

    async def handle(request: web.Request) -> web.Response:
        found = log_files.get(request.match_info["key"])
        return web.json_response(found) if found else web.Response(status=404)

    app = web.Application()
    app.router.add_get("/{key:.*}", handle)
    async with AsyncExitStack() as stack:
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        stack.push_async_callback(server.close)
        http = await stack.enter_async_context(aiohttp.ClientSession())
        ledger, watching = Ledger(), Watching()
        ledger.set_state("upload_log_next", "2")
        upload_log = UploadLog(
            http, str(server.make_url("")).rstrip("/"), SIGNER, ledger, watching
        )
        assert await upload_log.poll() == 1
    assert watching.seen == [{"key": KEY, "task_id": "t"}]
