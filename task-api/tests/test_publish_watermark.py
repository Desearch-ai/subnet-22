import asyncio
from datetime import UTC, datetime, timedelta

from app.canonical import canonicalize
from publisher.records import build_record, from_zstd, page_key
from publisher.worker import Publisher

from tests.synthetic import page_row, synthetic_html, to_parquet

T0 = datetime.now(UTC).replace(microsecond=0) - timedelta(days=2)
WIDE = (T0 - timedelta(days=30), T0 + timedelta(days=30))
URL = "https://www.site.example/story"


def observed(n: int, at: datetime, task_id: str) -> dict:
    row = {**page_row(URL, synthetic_html(n)), "fetched_at": at}
    return build_record(row, task_id, "miner-a", WIDE)


def stored(pages, url: str = URL) -> dict:
    found = pages.client.get_object(
        Bucket=pages.bucket, Key=pages.path(page_key(canonicalize(url)))
    )
    return from_zstd(found["Body"].read())


def buckets(memory):
    temp = memory.storage()
    return temp, memory.storage(bucket=temp.bucket, prefix="pages-bucket/")


def test_an_unchanged_page_seen_again_moves_its_fetch_time_forward(memory):
    temp, pages = buckets(memory)
    publisher = Publisher(None, temp, pages, workers=2)
    try:
        first = publisher.put_latest(observed(1, T0, "t1"))
        unchanged = publisher.put_latest(observed(1, T0 + timedelta(seconds=300), "t2"))
        older = publisher.put_latest(observed(2, T0 + timedelta(seconds=200), "t3"))
    finally:
        publisher.close()

    assert first["kind"] == "new"
    assert unchanged is None and older is None
    assert stored(pages)["task_id"] == "t2", (
        "an older fetch cannot overwrite the latest"
    )


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
        changes, failure = publisher.publish(job)
    finally:
        publisher.close()

    assert failure is None
    assert [change["assigned_url"] for change in changes] == [good]
    assert asyncio.run(pages.stat(page_key(canonicalize(bad)))) is None
    assert stored(pages, good)["validator"] == "5Validator"
