import asyncio
import time
from argparse import Namespace

import pytest
from feeder import loop

from desearch.client import TaskApiError


def rows(*urls: str) -> list[dict]:
    return [{"host": url.split("/")[2], "url": url} for url in urls]


class Api:
    def __init__(self, fails_after: int | None = None):
        self.sent: list[list[dict]] = []
        self.fails_after = fails_after

    async def post(self, path: str, body: dict) -> dict:
        assert path == "/v1/admin/enqueue"
        if self.fails_after is not None and len(self.sent) >= self.fails_after:
            raise TaskApiError(503, "queue is full")
        assert all(set(url) == {"host", "url"} for url in body["urls"])
        self.sent.append(body["urls"])
        return {"round_id": "r", "batches": 1}


def settings(**changes) -> Namespace:
    base = dict(
        api="http://api.test",
        per_domain=5,
        days=1,
        concurrency=2,
        batch_target=25,
        queue_cap=100,
        low_water=10,
        refresh=3600.0,
        interval=3600.0,
    )
    return Namespace(**{**base, **changes})


def sent(sent_urls: loop.SentUrls, rows: list[dict]) -> list[str]:
    due = sent_urls.due(rows)
    sent_urls.mark(due)
    return [row["url"] for row in due]


def test_a_url_is_not_sent_again_until_it_is_due(tmp_path):
    sent_urls = loop.SentUrls(str(tmp_path / "state.db"))

    assert sent(sent_urls, rows("https://a.example/1", "https://a.example/2")) == [
        "https://a.example/1",
        "https://a.example/2",
    ]
    assert sent(sent_urls, rows("https://a.example/1", "https://a.example/3")) == [
        "https://a.example/3"
    ]
    assert loop.SentUrls(str(tmp_path / "state.db")).count() == 3


def test_a_url_goes_out_again_once_its_refresh_has_passed(tmp_path):
    sent_urls = loop.SentUrls(str(tmp_path / "state.db"))
    sent(sent_urls, rows("https://a.example/1"))

    assert sent_urls.due(rows("https://a.example/1")) == []
    assert [
        r["url"] for r in sent_urls.due(rows("https://a.example/1"), refresh=0.0)
    ] == ["https://a.example/1"]


def test_a_url_goes_out_again_when_its_sitemap_says_it_changed(tmp_path):
    sent_urls = loop.SentUrls(str(tmp_path / "state.db"))
    listed = [{**row, "lastmod": 100.0} for row in rows("https://a.example/1")]
    sent(sent_urls, listed)

    assert sent_urls.due(listed) == []
    assert sent_urls.due([{**listed[0], "lastmod": 200.0}]) == [
        {**listed[0], "lastmod": 200.0}
    ]


def test_spellings_of_one_url_count_as_the_same_url(tmp_path):
    sent_urls = loop.SentUrls(str(tmp_path / "state.db"))
    sent(sent_urls, rows("https://a.example/story?utm_source=rss"))

    assert sent_urls.due(rows("https://a.example/story")) == []


def test_urls_go_out_in_batches(tmp_path):
    api, sent_urls = Api(), loop.SentUrls(str(tmp_path / "state.db"))
    many = rows(*[f"https://a.example/{n}" for n in range(1100)])

    assert asyncio.run(loop.enqueue(api, many, 25, sent_urls)) == 1100
    assert [len(batch) for batch in api.sent] == [500, 500, 100]


def test_a_refused_batch_stops_the_send_and_its_urls_stay_due(tmp_path):
    api, sent_urls = Api(fails_after=1), loop.SentUrls(str(tmp_path / "state.db"))
    many = rows(*[f"https://a.example/{n}" for n in range(1100)])

    assert asyncio.run(loop.enqueue(api, many, 25, sent_urls)) == 500
    assert len(sent_urls.due(many)) == 600


def test_a_timed_out_enqueue_leaves_its_urls_due(tmp_path):
    class Slow:
        async def post(self, path, body):
            raise TaskApiError(0, "TimeoutError: no answer")

    sent_urls = loop.SentUrls(str(tmp_path / "state.db"))
    listed = rows("https://a.example/1")

    assert asyncio.run(loop.enqueue(Slow(), listed, 25, sent_urls)) == 0
    assert sent_urls.due(listed) == listed


def test_a_cycle_enqueues_only_what_is_new(tmp_path, monkeypatch):
    api, sent_urls = Api(), loop.SentUrls(str(tmp_path / "state.db"))
    monkeypatch.setattr(loop, "queue_depth", lambda _: _ready(3))

    def source():
        return _ready(rows("https://a.example/1", "https://a.example/2"))

    first = asyncio.run(loop.feed_once(settings(), source, sent_urls, api))
    second = asyncio.run(loop.feed_once(settings(), source, sent_urls, api))

    assert (first, second) == (2, 0)
    assert len(api.sent) == 1


def test_a_full_queue_is_left_alone(tmp_path, monkeypatch):
    api, sent_urls = Api(), loop.SentUrls(str(tmp_path / "state.db"))
    monkeypatch.setattr(loop, "queue_depth", lambda _: _ready(500))

    def source():
        pytest.fail("should not read urls")

    assert (
        asyncio.run(loop.feed_once(settings(queue_cap=100), source, sent_urls, api))
        == 0
    )
    assert api.sent == []


async def _ready(value):
    return value


def test_a_drained_queue_is_topped_up_without_waiting_out_the_interval(monkeypatch):
    depths = iter([80, 40, 5])
    monkeypatch.setattr(loop, "queue_depth", lambda _: _ready(next(depths)))
    monkeypatch.setattr(loop.asyncio, "sleep", lambda _: _ready(None))

    asyncio.run(loop.wait_for_next_cycle(settings(), Api(), time.monotonic(), sent=50))

    assert next(depths, "drained") == "drained"


def test_a_cycle_that_sent_nothing_waits_out_the_interval(monkeypatch):
    monkeypatch.setattr(loop, "queue_depth", lambda _: pytest.fail("no reason to look"))
    monkeypatch.setattr(loop.asyncio, "sleep", lambda _: _ready(None))

    asyncio.run(
        loop.wait_for_next_cycle(
            settings(interval=0.0), Api(), time.monotonic(), sent=0
        )
    )


def test_a_request_carries_whole_batches_even_when_they_are_large(tmp_path):
    api, sent_urls = Api(), loop.SentUrls(str(tmp_path / "state.db"))
    many = rows(*[f"https://h{n % 50}.example/{n}" for n in range(25_000)])

    assert asyncio.run(loop.enqueue(api, many, 1000, sent_urls)) == 25_000
    assert [len(batch) for batch in api.sent] == [10_000, 10_000, 5_000]


def test_the_feeder_watches_only_the_crawl_queue():
    class Health:
        async def get(self, path):
            return {"queue_depth": {"crawl": 7, "embed": 900}}

    assert asyncio.run(loop.queue_depth(Health())) == 7
