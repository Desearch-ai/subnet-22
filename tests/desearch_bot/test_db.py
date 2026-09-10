from datetime import datetime, timedelta, timezone

from desearch_bot import db
from desearch_bot.loop import DomainWrite
from desearch_bot.schedule import Trust
from desearch_bot.states import PUBLIC, State
from desearch_bot.visit import SitemapUpdate, Visit

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
URL = "https://a.com/sitemap.xml"


async def _domains(pool, *hosts, state="new", due=NOW):
    await db.load_candidates(
        pool, [(host, rank, "big_generic") for rank, host in enumerate(hosts, 1)]
    )
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE bot.domains SET state = $1, next_due_at = $2 WHERE host = ANY($3)",
            state,
            due,
            list(hosts),
        )


async def _row(pool, table, key, value):
    async with pool.acquire() as connection:
        return await connection.fetchrow(
            f"SELECT * FROM bot.{table} WHERE {key} = $1", value
        )


def _write(host, sitemaps=(), due=NOW + timedelta(hours=1), **visit):
    return DomainWrite(
        host,
        State.ACTIVE,
        None,
        0,
        due,
        NOW,
        None,
        NOW,
        Visit(host, sitemaps=list(sitemaps), **visit),
    )


def _update(sitemap_id, changed=True, url_count=12):
    return SitemapUpdate(
        sitemap_id,
        URL,
        "urlset",
        0,
        None,
        etag='"v1"',
        content_hash="h",
        changed=changed,
        url_count=url_count,
        interval=timedelta(hours=6),
        next_check_at=NOW + timedelta(hours=6),
        trust=Trust.TRUSTED,
    )


async def test_refreshes_come_from_their_own_tier_and_busy_hosts_are_skipped(pool):
    await _domains(pool, "a.com", "c.com", state="active")
    await _domains(pool, "b.com")
    assert {k.host for k in await db.due(pool, 10, [], True, NOW)} == {"a.com", "c.com"}
    assert [k.host for k in await db.due(pool, 10, ["a.com"], True, NOW)] == ["c.com"]
    assert [k.host for k in await db.due(pool, 10, [], False, NOW)] == ["b.com"]


async def test_nothing_is_handed_out_before_it_is_due(pool):
    await _domains(pool, "a.com", due=NOW + timedelta(hours=1))
    assert await db.due(pool, 10, [], False, NOW) == []
    assert len(await db.due(pool, 10, [], False, NOW + timedelta(hours=1))) == 1


async def test_new_domains_are_discovered_in_rank_order(pool):
    await _domains(pool, "first.com", "second.com", "third.com")
    found = await db.due(pool, 10, [], False, NOW)
    assert [k.host for k in found] == ["first.com", "second.com", "third.com"]


async def test_a_sitemap_keeps_its_id(pool):
    await _domains(pool, "a.com")
    first = await db.allocate_sitemap(pool, "a.com", URL)
    assert await db.allocate_sitemap(pool, "a.com", URL) == first


async def test_what_a_visit_saved_is_what_the_next_visit_knows(pool):
    await _domains(pool, "a.com")
    sitemap_id = await db.allocate_sitemap(pool, "a.com", URL)
    await db.save_visits(
        pool,
        [
            _write(
                "a.com",
                [_update(sitemap_id)],
                due=NOW,
                robots_read=True,
                robots_status=200,
                robots_allows=True,
                crawl_delay=2.0,
                language="en",
            )
        ],
    )
    [known] = await db.due(pool, 10, [], True, NOW)
    assert (known.state, known.crawl_delay, known.language) == (State.ACTIVE, 2.0, "en")
    assert known.robots_checked_at == NOW
    stored = known.sitemaps[URL]
    assert (stored.id, stored.etag, stored.trust) == (sitemap_id, '"v1"', Trust.TRUSTED)
    assert (stored.interval, stored.url_count) == (timedelta(hours=6), 12)
    assert (await _row(pool, "domains", "host", "a.com"))["url_count"] == 12


async def test_an_unchanged_read_keeps_the_count_and_the_robots_it_did_not_reread(pool):
    await _domains(pool, "a.com")
    sitemap_id = await db.allocate_sitemap(pool, "a.com", URL)
    await db.save_visits(
        pool,
        [_write("a.com", [_update(sitemap_id)], robots_read=True, crawl_delay=2.0)],
    )
    await db.save_visits(pool, [_write("a.com", [_update(sitemap_id, False, 0)])])
    sitemap = await _row(pool, "sitemaps", "id", sitemap_id)
    domain = await _row(pool, "domains", "host", "a.com")
    assert (sitemap["url_count"], domain["url_count"], domain["crawl_delay"]) == (
        12,
        12,
        2.0,
    )


async def test_redirect_destinations_join_ranked_like_their_source(pool):
    await _domains(pool, "old.com")
    rows = [
        ("new.com", "old.com", "big_generic", State.NEW, None),
        (
            "cdn.new.com",
            "old.com",
            "big_generic",
            State.EXCLUDED,
            "infrastructure_name",
        ),
    ]
    await db.adopt(pool, rows, NOW)
    await db.adopt(pool, rows, NOW)
    new = await _row(pool, "domains", "host", "new.com")
    assert (new["state"], new["rank"], new["next_due_at"]) == ("new", 1, NOW)
    excluded = await _row(pool, "domains", "host", "cdn.new.com")
    assert (excluded["state"], excluded["next_due_at"]) == ("excluded", None)


async def test_only_public_states_without_excluded_labels_are_published(pool):
    hosts = [f"{state.value}.com" for state in State]
    for host, state in zip(hosts, State):
        await _domains(pool, host, state=state.value)
    await _domains(pool, "labelled.com")
    assert await db.exclude_hosts(pool, [("labelled.com", "adult")]) == 1
    assert await db.exclude_hosts(pool, [("labelled.com", "adult")]) == 0
    published = [
        r["host"] async for rows in db.iter_published_hosts(pool) for r in rows
    ]
    assert sorted(published) == sorted(f"{state.value}.com" for state in PUBLIC)
