import asyncio
from collections import Counter

from desearch_bot import db, loop
from desearch_bot.visit import Visit


class Stuck:
    """A visitor whose visits never finish."""

    def __init__(self):
        self.started = 0

    async def visit(self, known, now):
        self.started += 1
        await asyncio.Event().wait()


async def test_stopping_drops_visits_that_outlast_the_grace_period(pool, monkeypatch):
    monkeypatch.setattr(loop, "STOP_GRACE", 0.2)
    monkeypatch.setattr(loop, "TICK", 0.05)
    await db.load_candidates(pool, [(f"d{i}.com", i, "big_generic") for i in range(3)])
    visitor = Stuck()
    crawl = loop.Loop(pool, visitor, 5, frozenset())
    running = asyncio.create_task(crawl.run())
    for _ in range(100):
        if visitor.started == 3:
            break
        await asyncio.sleep(0.02)
    crawl.stop()
    await asyncio.wait_for(running, timeout=5)
    due = await db.due(pool, 10, [], False, loop._now())
    assert visitor.started == 3 and len(due) == 3


class Quick:
    """A visitor that answers at once, counting how often each domain is visited."""

    def __init__(self):
        self.visits = Counter()

    async def visit(self, known, now):
        self.visits[known.host] += 1
        return Visit(known.host, requests=1)


async def test_a_visit_waiting_to_be_written_is_not_handed_out_again(pool, monkeypatch):
    monkeypatch.setattr(loop, "TICK", 0.02)
    monkeypatch.setattr(loop, "FLUSH_SECONDS", 10.0)
    await db.load_candidates(pool, [(f"d{i}.com", i, "big_generic") for i in range(3)])
    visitor = Quick()
    crawl = loop.Loop(pool, visitor, 5, frozenset())
    running = asyncio.create_task(crawl.run())
    await asyncio.sleep(0.5)
    crawl.stop()
    await asyncio.wait_for(running, timeout=5)
    assert visitor.visits == Counter({"d0.com": 1, "d1.com": 1, "d2.com": 1})
    assert await db.due(pool, 10, [], False, loop._now()) == []


class Slow:
    """A visitor whose visits each outlast several flushes."""

    def __init__(self):
        self.visits = Counter()

    async def visit(self, known, now):
        self.visits[known.host] += 1
        await asyncio.sleep(0.05 * (1 + int(known.host[1])))
        return Visit(known.host, requests=1)


async def test_a_visit_that_outlasts_a_flush_is_still_written(pool, monkeypatch):
    monkeypatch.setattr(loop, "TICK", 0.01)
    monkeypatch.setattr(loop, "FLUSH_SECONDS", 0.02)
    await db.load_candidates(pool, [(f"d{i}.com", i, "big_generic") for i in range(5)])
    visitor = Slow()
    crawl = loop.Loop(pool, visitor, 5, frozenset())
    running = asyncio.create_task(crawl.run())
    await asyncio.sleep(1.0)
    crawl.stop()
    await asyncio.wait_for(running, timeout=5)
    assert visitor.visits == Counter({f"d{i}.com": 1 for i in range(5)})
    now = loop._now()
    assert (
        await db.due(pool, 10, [], True, now)
        == await db.due(pool, 10, [], False, now)
        == []
    )
