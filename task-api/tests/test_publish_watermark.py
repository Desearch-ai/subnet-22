from datetime import datetime, timedelta, timezone

from app.canonical import canonicalize
from publisher.index import VersionIndex
from publisher.records import build_record, page_key, record_key
from publisher.worker import Publisher, _indexed

from tests.synthetic import page_row, synthetic_html, to_parquet

T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=2)
WIDE = (T0 - timedelta(days=30), T0 + timedelta(days=30))
URL = "https://www.site.example/story"


def observed(n: int, at: datetime, task_id: str, url: str = URL) -> dict:
    row = {**page_row(url, synthetic_html(n)), "fetched_at": at}
    return build_record(row, task_id, "miner-a", WIDE)


def buckets(memory):
    temp = memory.storage()
    return temp, memory.storage(bucket=temp.bucket, prefix="pages-bucket/")


def publish(publisher: Publisher, record: dict) -> tuple[list[dict], list[dict]]:
    changes, unchanged = publisher.decide({record_key(record): record})
    publisher.index.store([_indexed(change) for change in changes])
    publisher.index.touch([(record_key(r), r["fetched_at"]) for r in unchanged])
    return changes, unchanged


def test_an_unchanged_page_seen_again_moves_its_fetch_time_forward(memory):
    temp, pages = buckets(memory)
    publisher = Publisher(None, temp, pages, workers=2, index=VersionIndex(":memory:"))
    try:
        first, _ = publish(publisher, observed(1, T0, "t1"))
        same, seen = publish(publisher, observed(1, T0 + timedelta(seconds=300), "t2"))
        older, kept_out = publish(
            publisher, observed(2, T0 + timedelta(seconds=200), "t3")
        )
    finally:
        publisher.close()

    assert [c["kind"] for c in first] == ["new"]
    assert same == [] and len(seen) == 1
    assert older == [] and len(kept_out) == 1, "an older fetch cannot replace it"
    current = publisher.index.current(page_key(canonicalize(URL)))
    assert current.task_id == "t1" and current.fetched_at > first[0]["fetched_at"]


def test_rows_the_verdict_rejected_are_not_published(memory):
    temp, pages = buckets(memory)
    good, bad = "https://www.site.example/good", "https://www.site.example/bad"
    rows = [
        {**page_row(good, synthetic_html(1)), "fetched_at": T0},
        {**page_row(bad, synthetic_html(2)), "fetched_at": T0},
    ]
    key = "submitted/t1.parquet"
    temp.client.put_object(
        Bucket=temp.bucket,
        Key=temp.path(key),
        Body=to_parquet(rows, task_id="t1", hotkey="miner-a"),
    )
    job = {
        "task_id": "t1",
        "miner": "miner-a",
        "key": key,
        "etag": None,
        "round_id": "r",
        "urls": [good, bad],
        "skip": [bad],
        "validator": "5Validator",
        "completed_at": (T0 + timedelta(seconds=30)).timestamp(),
        "claim_ttl": 900,
    }
    publisher = Publisher(None, temp, pages, workers=2)
    try:
        records, failed = publisher.collect(job)
    finally:
        publisher.close()

    assert [record["assigned_url"] for record in records] == [good]
    assert records[0]["validator"] == "5Validator"
    assert failed == [{"url": bad, "task_id": "t1"}]


def test_a_withdrawal_removes_only_a_version_that_is_still_current():
    index = VersionIndex(":memory:")
    first = {**observed(1, T0, "t1"), "key": "k", "version": "v1"}
    index.store([_indexed(first)])
    index.store([_indexed({**first, "version": "v2", "task_id": "t2"})])

    withdrawn = index.of_tasks(["t1"])
    index.remove([(key, version) for key, _, version, _ in withdrawn])

    assert withdrawn == [], "t2 replaced t1's version, so t1 holds nothing now"
    assert index.current("k").task_id == "t2"


def test_uploads_read_by_reader_processes_come_back_in_order_with_what_stopped_them(
    memory,
):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from publisher import worker

    temp, pages = buckets(memory)
    jobs = []
    for n in range(5):
        url = f"https://www.site.example/story-{n}"
        key = f"submitted/t{n}.parquet"
        temp.client.put_object(
            Bucket=temp.bucket,
            Key=temp.path(key),
            Body=to_parquet(
                [{**page_row(url, synthetic_html(n)), "fetched_at": T0}],
                task_id=f"t{n}",
                hotkey="miner-a",
            ),
        )
        jobs.append(
            {
                "task_id": f"t{n}",
                "miner": "miner-a",
                "key": key if n != 3 else "submitted/gone.parquet",
                "etag": None,
                "urls": [url],
                "completed_at": (T0 + timedelta(seconds=30)).timestamp(),
            }
        )
    publisher = Publisher(None, temp, pages, workers=2)
    publisher.readers, publisher.reader_count = ThreadPoolExecutor(2), 2
    worker._reader = publisher
    try:
        found = asyncio.run(publisher.read_all(jobs))
    finally:
        worker._reader = None
        publisher.close()

    assert isinstance(found[3], worker.UploadGone)
    for n in (0, 1, 2, 4):
        records, missed = found[n]
        assert [r["task_id"] for r in records] == [f"t{n}"] and missed == []
