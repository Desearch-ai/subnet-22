import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from desearch_bot import loop
from desearch_bot.visit import Visit

from .harness import MemoryRegistry, open_buckets, seed

HOSTS = [f"d{i}.com" for i in range(5)]


@pytest.fixture
def buckets(tmp_path):
    opened = open_buckets(tmp_path, HOSTS)
    seed(opened, HOSTS, datetime.now(timezone.utc) - timedelta(seconds=1))
    yield opened
    opened.close()


class Stuck:
    """A visitor whose visits never finish."""

    def __init__(self):
        self.started = 0

    async def visit(self, known, now):
        self.started += 1
        await asyncio.Event().wait()


class Quick:
    """A visitor that answers at once, counting how often each domain is visited."""

    def __init__(self):
        self.visits = Counter()

    async def visit(self, known, now):
        self.visits[known.host] += 1
        return Visit(known.host, requests=1)


class Slow(Quick):
    """A visitor whose visits each outlast several flushes."""

    async def visit(self, known, now):
        self.visits[known.host] += 1
        await asyncio.sleep(0.05 * (1 + int(known.host[1])))
        return Visit(known.host, requests=1)


async def _run_for(crawl, seconds):
    running = asyncio.create_task(crawl.run())
    await asyncio.sleep(seconds)
    crawl.stop()
    await asyncio.wait_for(running, timeout=5)


def _checked(buckets):
    return [buckets.store(host).domain(host).get("checked") for host in HOSTS]


async def test_stopping_drops_visits_that_outlast_the_grace_period(
    buckets, monkeypatch
):
    monkeypatch.setattr(loop, "STOP_GRACE", 0.2)
    monkeypatch.setattr(loop, "TICK", 0.02)
    visitor = Stuck()
    crawl = loop.Loop(buckets, visitor, 10, MemoryRegistry(), frozenset())
    crawl.load()
    await _run_for(crawl, 0.3)
    assert visitor.started == len(HOSTS) and set(_checked(buckets)) == {None}


async def test_every_domain_is_visited_once_and_reported(buckets, monkeypatch):
    monkeypatch.setattr(loop, "TICK", 0.02)
    monkeypatch.setattr(loop, "FLUSH_SECONDS", 10.0)
    visitor, registry = Quick(), MemoryRegistry()
    crawl = loop.Loop(buckets, visitor, 10, registry, frozenset())
    crawl.load()
    await _run_for(crawl, 0.5)
    assert visitor.visits == Counter(HOSTS) and None not in _checked(buckets)
    assert sorted(write.host for write, _ in registry.reported) == HOSTS


async def test_a_visit_that_outlasts_a_flush_is_still_written(buckets, monkeypatch):
    monkeypatch.setattr(loop, "TICK", 0.01)
    monkeypatch.setattr(loop, "FLUSH_SECONDS", 0.02)
    visitor = Slow()
    crawl = loop.Loop(buckets, visitor, 10, MemoryRegistry(), frozenset())
    crawl.load()
    await _run_for(crawl, 1.0)
    assert visitor.visits == Counter(HOSTS) and None not in _checked(buckets)
