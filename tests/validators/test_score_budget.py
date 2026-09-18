import asyncio
import datetime
import io
import multiprocessing
import struct
import sys
from types import SimpleNamespace

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aiohttp import web

from desearch.extraction.schema import PAGE_SCHEMA
from neurons.validators import crawl
from neurons.validators.crawl import CrawlValidator
from neurons.validators.scoring import (
    FetchedPage,
    load_upload,
    sample_seed,
    score,
)
from neurons.validators.scoring_process import cap_memory
from tests.local_http import serving
from tests.synthetic import synthetic, to_parquet


def score_task(rows, assigned, fetched, **options):
    parquet = to_parquet(rows, task_id="t1", hotkey="m")

    async def fetch(url: str, rendered: bool = False) -> FetchedPage:
        return fetched[url]

    async def upload(_) -> web.Response:
        return web.Response(body=parquet)

    async def run():
        async with serving(upload) as base, aiohttp.ClientSession() as http:
            job = {
                "task_id": "t1",
                "miner": "m",
                "urls": assigned,
                "download_url": base + "x",
            }
            validator = CrawlValidator(
                SimpleNamespace(hotkey="v"), fetch, http, min_samples=5, **options
            )
            return await validator.score_task(job)

    return asyncio.run(run())


def test_scoring_in_a_child_gives_the_same_result():
    rows, assigned, fetched = synthetic(8)
    seed = sample_seed("t1", "v", crawl.SAMPLE_SALT)

    result = score_task(rows, assigned, fetched)
    urls = result.pop("urls")
    result.pop("took_ms")

    assert result == score(rows, assigned, fetched, seed, 5, 0.8)
    assert [detail["url"] for detail in urls] == [row["url"] for row in rows]
    assert {d["url"] for d in urls if d["sampled"]} == {
        s["url"] for s in result["samples"]
    }


def test_a_task_that_cannot_be_scored_in_time_fails_as_unscorable():
    rows, assigned, fetched = synthetic(4)

    result = score_task(rows, assigned, fetched, score_timeout=0.001)

    assert (result["verdict"], result["reason"]) == ("fail", "unscorable")


def varint(n: int) -> bytes:
    zigzag, out = n << 1, bytearray()
    while True:
        byte, zigzag = zigzag & 0x7F, zigzag >> 7
        if not zigzag:
            return bytes(out + bytes([byte]))
        out.append(byte | 0x80)


def bomb(decoded: int, claimed: int) -> bytes:
    row = {
        name: ""
        for name in PAGE_SCHEMA.names
        if PAGE_SCHEMA.field(name).type == pa.string()
    } | {
        "status": 200,
        "error": None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc),
        "elapsed_ms": 1,
        "html_bytes": decoded,
        "html": b"<p>a</p>" * (decoded // 8),
        "json_ld_types": [],
        "headings": [],
        "text": "",
        "url": "https://a.example/",
    }
    sink = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist([row], schema=PAGE_SCHEMA), sink, compression="zstd"
    )
    data = bytearray(sink.getvalue())
    meta = pq.ParquetFile(io.BytesIO(bytes(data))).metadata
    size = max(
        meta.row_group(0).column(c).total_uncompressed_size
        for c in range(meta.num_columns)
    )
    footer_len = struct.unpack("<I", data[-8:-4])[0]
    start = len(data) - 8 - footer_len
    old, new = varint(size), varint(claimed)
    assert len(old) == len(new)
    at = bytes(data[start:]).index(old)
    data[start + at : start + at + len(old)] = new
    return bytes(data)


def test_a_footer_that_understates_the_upload_does_not_get_it_decoded():
    upload = bomb(decoded=20_000_000, claimed=2_000_000)
    claimed = pq.ParquetFile(io.BytesIO(upload)).metadata
    assert (
        sum(
            claimed.row_group(0).column(c).total_uncompressed_size
            for c in range(claimed.num_columns)
        )
        < 6_000_000
    )

    assert load_upload(upload, ["https://a.example/"], "seed") == (None, {})


def virtual_mb() -> int:
    with open("/proc/self/status") as status:
        line = next(line for line in status if line.startswith("VmSize:"))
    return int(line.split()[1]) // 1024


def allocate_past_the_cap(conn) -> None:
    cap_memory(virtual_mb() + 128)
    try:
        bytearray(512 * 1024 * 1024)
        conn.send("allocated")
    except MemoryError:
        conn.send("refused")


linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="only Linux enforces RLIMIT_AS"
)


@linux_only
def test_the_memory_cap_turns_an_oversized_allocation_into_a_memory_error():
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=allocate_past_the_cap, args=(sender,))
    child.start()
    assert receiver.poll(60)
    assert receiver.recv() == "refused"
    child.join(10)
