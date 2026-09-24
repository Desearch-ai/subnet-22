from datetime import datetime, timedelta, timezone

from desearch_bot import db
from desearch_bot.buckets import bucket_of
from desearch_bot.states import PUBLIC, State

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


async def _domains(pool, *hosts):
    await db.load_candidates(
        pool, [(host, rank, "big_generic") for rank, host in enumerate(hosts, 1)]
    )


async def _set_state(pool, host, state):
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE bot.domains SET state = $1 WHERE host = $2", state, host
        )


async def _row(pool, host):
    async with pool.acquire() as connection:
        return await connection.fetchrow(
            "SELECT * FROM bot.domains WHERE host = $1", host
        )


async def test_new_domains_carry_their_bucket_and_a_change_time(pool):
    await _domains(pool, "a.com")
    row = await _row(pool, "a.com")
    assert row["bucket"] == bucket_of("a.com") and row["changed_at"] is not None


async def test_changes_come_back_for_the_asked_buckets_page_by_page(pool):
    hosts = [f"site{i}.com" for i in range(40)]
    await _domains(pool, *hosts)
    mine = sorted({bucket_of(host) for host in hosts[:10]})
    seen, since = [], None
    while page := await db.registry_changes(pool, mine, since, 3):
        seen += [row["host"] for row in page]
        since = (page[-1]["changed_at"], page[-1]["host"])
    assert sorted(seen) == sorted(h for h in hosts if bucket_of(h) in mine)
    assert len(seen) == len(set(seen))


async def test_reports_keep_the_latest_visit_and_never_undo_an_exclusion(pool):
    await _domains(pool, "a.com", "b.com")
    await db.exclude_hosts(pool, [("b.com", "adult")])
    await db.report_visits(
        pool,
        [
            ("a.com", "failing", "timeout", NOW, 0, None),
            ("a.com", "active", None, NOW + timedelta(minutes=1), 12, None),
            ("b.com", "active", None, NOW, 5, None),
        ],
    )
    a, b = await _row(pool, "a.com"), await _row(pool, "b.com")
    assert (a["state"], a["url_count"], a["checked_at"]) == (
        "active",
        12,
        NOW + timedelta(minutes=1),
    )
    assert b["state"] == "excluded"


async def test_a_report_is_not_a_change_the_crawlers_need_to_pull(pool):
    await _domains(pool, "a.com")
    before = (await _row(pool, "a.com"))["changed_at"]
    await db.report_visits(pool, [("a.com", "active", None, NOW, 3, None)])
    assert (await _row(pool, "a.com"))["changed_at"] == before


async def test_excluding_a_domain_marks_it_changed(pool):
    await _domains(pool, "a.com")
    before = (await _row(pool, "a.com"))["changed_at"]
    assert await db.exclude_hosts(pool, [("a.com", "adult")]) == 1
    assert await db.exclude_hosts(pool, [("a.com", "adult")]) == 0
    row = await _row(pool, "a.com")
    assert row["state"] == "excluded" and row["changed_at"] > before


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
    new = await _row(pool, "new.com")
    assert (new["state"], new["rank"], new["bucket"], new["changed_at"]) == (
        "new",
        1,
        bucket_of("new.com"),
        NOW,
    )
    assert (await _row(pool, "cdn.new.com"))["state"] == "excluded"


async def test_only_public_states_without_excluded_labels_are_published(pool):
    hosts = [f"{state.value}.com" for state in State]
    await _domains(pool, *hosts, "labelled.com")
    for host, state in zip(hosts, State):
        await _set_state(pool, host, state.value)
    await db.exclude_hosts(pool, [("labelled.com", "adult")])
    published = [
        r["host"] async for rows in db.iter_published_hosts(pool) for r in rows
    ]
    assert sorted(published) == sorted(f"{state.value}.com" for state in PUBLIC)
