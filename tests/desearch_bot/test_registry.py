from datetime import datetime, timezone

from desearch_bot import db
from desearch_bot.buckets import Buckets, Resources, bucket_of
from desearch_bot.loop import DomainWrite
from desearch_bot.registry import Registry
from desearch_bot.states import Outcome, State
from desearch_bot.visit import Visit

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
RESOURCES = Resources(cache_bytes=8 << 20, memtable_bytes=64 << 20)


async def test_a_registry_picks_up_where_it_left_off_after_a_restart(pool, tmp_path):
    hosts = [f"site{i}.com" for i in range(30)]
    await db.load_candidates(
        pool, [(host, i, "big_generic") for i, host in enumerate(hosts)]
    )
    with Buckets(tmp_path, sorted({bucket_of(h) for h in hosts}), RESOURCES) as buckets:
        first = await Registry(pool, buckets).changes()
        assert sorted(change.host for change in first) == sorted(hosts)
        assert await Registry(pool, buckets).changes() == []
        await db.exclude_hosts(pool, [("site3.com", "adult")])
        [change] = await Registry(pool, buckets).changes()
        assert (change.host, change.state) == ("site3.com", "excluded")


async def test_reporting_a_redirect_adds_its_destination_to_the_list(pool, tmp_path):
    await db.load_candidates(pool, [("old.com", 5, "big_generic")])
    visit = Visit("old.com", Outcome.REDIRECT, canonical_host="new.com", requests=1)
    write = DomainWrite(
        "old.com", State.REDIRECTS, "redirect", 0, None, NOW, "new.com", NOW, visit
    )
    with Buckets(tmp_path, [bucket_of("old.com")], RESOURCES) as buckets:
        await Registry(pool, buckets).report([(write, 0)], NOW)
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT host, state, rank, canonical_host FROM bot.domains"
        )
    found = {row["host"]: row for row in rows}
    assert (found["old.com"]["state"], found["old.com"]["canonical_host"]) == (
        "redirects",
        "new.com",
    )
    assert (found["new.com"]["state"], found["new.com"]["rank"]) == ("new", 5)
