"""Postgres access for the control plane."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import asyncpg

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def dsn() -> str:
    url = os.environ.get("DESEARCH_DB")
    if not url:
        raise RuntimeError("DESEARCH_DB is not set")
    return url


async def connect(pool_size: int = 16) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn(),
        min_size=2,
        max_size=pool_size,
        command_timeout=180,
        server_settings={"search_path": "bot,public"},
    )


async def create_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(SCHEMA_FILE.read_text())


async def load_candidates(pool: asyncpg.Pool, rows) -> None:
    """COPY candidates into a staging table, then merge, so a 6M-row load is one pass."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE candidate_stage (
                    host text, rank integer, tld_group text
                ) ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "candidate_stage",
                records=rows,
                columns=["host", "rank", "tld_group"],
            )
            await connection.execute(
                """
                INSERT INTO bot.domains (host, rank, tld_group)
                SELECT DISTINCT ON (host) host, rank, tld_group
                FROM candidate_stage
                ORDER BY host, rank NULLS LAST
                ON CONFLICT (host) DO NOTHING
                """
            )


async def iter_hosts(pool: asyncpg.Pool, chunk: int = 200_000):
    """Walk every host by primary key, so a six-million-row pass never holds them all."""
    after = ""
    while True:
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT host, tld_group FROM bot.domains
                WHERE host > $1 ORDER BY host LIMIT $2
                """,
                after,
                chunk,
            )
        if not rows:
            return
        yield rows
        after = rows[-1]["host"]


async def iter_published_hosts(pool: asyncpg.Pool, chunk: int = 200_000):
    """Only what the public list should contain: reachable, canonical, not disqualified."""
    after = ""
    while True:
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT host FROM bot.published_domains
                WHERE host > $1 ORDER BY host LIMIT $2
                """,
                after,
                chunk,
            )
        if not rows:
            return
        yield rows
        after = rows[-1]["host"]


async def existing_hosts(pool: asyncpg.Pool, hosts) -> set[str]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT host FROM bot.domains WHERE host = ANY($1::text[])", list(hosts)
        )
    return {r["host"] for r in rows}


async def checked_hosts(pool: asyncpg.Pool, source: str) -> set[str]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT host FROM bot.category_checks WHERE source = $1", source
        )
    return {r["host"] for r in rows}


async def save_category_checks(pool: asyncpg.Pool, hosts, source: str) -> None:
    hosts = list(hosts)
    if not hosts:
        return
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO bot.category_checks (host, source)
            SELECT host, $2 FROM unnest($1::text[]) AS host
            ON CONFLICT (host, source) DO UPDATE SET checked_at = now()
            """,
            hosts,
            source,
        )


async def refresh_category_rollup(pool: asyncpg.Pool, hosts, priority) -> None:
    """Rebuild each domain's category columns from every source's labels."""
    hosts = list(hosts)
    if not hosts:
        return
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE bot.domains d SET categories = r.labels, category = r.labels[1]
            FROM (
                SELECT host, array_agg(category ORDER BY ordinal, category) AS labels
                FROM (
                    SELECT DISTINCT host, category,
                           coalesce(array_position($2::text[], category), 1000) AS ordinal
                    FROM bot.domain_categories WHERE host = ANY($1::text[])
                ) labelled
                GROUP BY host
            ) r
            WHERE d.host = r.host
            """,
            hosts,
            list(priority),
        )


async def save_domain_categories(pool: asyncpg.Pool, rows) -> int:
    """Append what a source said. One row per source per label, so sources never overwrite."""
    if not rows:
        return 0
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE dc_stage (
                    host text, source text, category text, raw_category text,
                    category_id integer, super_category text
                ) ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "dc_stage",
                records=rows,
                columns=[
                    "host",
                    "source",
                    "category",
                    "raw_category",
                    "category_id",
                    "super_category",
                ],
            )
            result = await connection.execute(
                """
                INSERT INTO bot.domain_categories
                    (host, source, category, raw_category, category_id, super_category)
                SELECT s.host, s.source, s.category, s.raw_category,
                       s.category_id, s.super_category
                FROM dc_stage s JOIN bot.domains d ON d.host = s.host
                ON CONFLICT DO NOTHING
                """
            )
    return int(result.split()[-1])


async def save_category_rollup(pool: asyncpg.Pool, rows) -> int:
    """The denormalised copy the publish view and the crawl filter read."""
    if not rows:
        return 0
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE rollup_stage (host text, categories text[], category text)
                ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "rollup_stage", records=rows, columns=["host", "categories", "category"]
            )
            result = await connection.execute(
                """
                UPDATE bot.domains d SET categories = s.categories, category = s.category
                FROM rollup_stage s WHERE d.host = s.host
                """
            )
    return int(result.split()[-1])


async def counts(pool: asyncpg.Pool) -> dict:
    async with pool.acquire() as connection:
        states = await connection.fetch(
            "SELECT state, count(*) AS n FROM bot.domains GROUP BY state ORDER BY state"
        )
        totals = await connection.fetchrow(
            """
            SELECT (SELECT count(*) FROM bot.domains) AS domains,
                   (SELECT count(*) FROM bot.sitemaps) AS sitemaps,
                   (SELECT coalesce(sum(url_count), 0) FROM bot.domains) AS urls
            """
        )
    return {**dict(totals), **{row["state"]: row["n"] for row in states}}


REFRESH = ("active", "failing", "down")
DISCOVERY = ("new", "unreachable", "no_sitemap", "redirects", "blocked", "ineligible")


async def due(
    pool: asyncpg.Pool, limit: int, busy: list[str], refresh: bool, now: datetime
) -> list:
    """Domains whose next visit has come, oldest first, skipping any already being visited."""
    from .schedule import Trust
    from .states import State
    from .visit import Known, KnownSitemap

    if limit <= 0:
        return []
    tier = REFRESH if refresh else DISCOVERY
    # A literal list, so the planner can match the partial index for this tier.
    predicate = "state IN (" + ", ".join(f"'{state}'" for state in tier) + ")"
    order = "next_due_at" if refresh else "next_due_at, rank"
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            f"""
            SELECT host, state, failures, last_ok_at, robots_checked_at, robots_allows,
                   crawl_delay, language, categories, canonical_host
            FROM bot.domains
            WHERE {predicate} AND next_due_at <= $3 AND host <> ALL($1::text[])
            ORDER BY {order}
            LIMIT $2
            """,
            busy,
            limit,
            now,
        )
        if not rows:
            return []
        sitemap_rows = await connection.fetch(
            """
            SELECT id, host, url, kind, depth, parent_id, etag, last_modified, content_hash,
                   check_interval_s, next_check_at, trust, index_lastmod, url_count
            FROM bot.sitemaps WHERE host = ANY($1::text[])
            """,
            [row["host"] for row in rows],
        )

    by_host: dict[str, dict] = defaultdict(dict)
    for s in sitemap_rows:
        by_host[s["host"]][s["url"]] = KnownSitemap(
            s["id"],
            s["url"],
            s["kind"],
            s["depth"],
            s["parent_id"],
            s["etag"],
            s["last_modified"],
            s["content_hash"],
            timedelta(seconds=s["check_interval_s"]),
            s["next_check_at"],
            Trust(s["trust"]),
            s["index_lastmod"],
            s["url_count"],
        )
    return [
        Known(
            host=row["host"],
            state=State(row["state"]),
            failures=row["failures"],
            last_ok_at=row["last_ok_at"],
            robots_checked_at=row["robots_checked_at"],
            robots_allows=row["robots_allows"],
            crawl_delay=row["crawl_delay"],
            language=row["language"],
            categories=frozenset(row["categories"] or ()),
            sitemaps=by_host.get(row["host"], {}),
            canonical_host=row["canonical_host"],
        )
        for row in rows
    ]


async def allocate_sitemap(pool: asyncpg.Pool, host: str, url: str) -> int:
    """The id for a sitemap file, creating its row the first time we meet it."""
    async with pool.acquire() as connection:
        return await connection.fetchval(
            """
            INSERT INTO bot.sitemaps (host, url) VALUES ($1, $2)
            ON CONFLICT (url) DO UPDATE SET host = bot.sitemaps.host
            RETURNING id
            """,
            host,
            url,
        )


async def excluded_categories(pool: asyncpg.Pool) -> frozenset[str]:
    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT category FROM bot.excluded_categories")
    return frozenset(row["category"] for row in rows)


async def save_visits(pool: asyncpg.Pool, writes) -> None:
    """Write a batch of finished visits: each domain's new state, and every sitemap it read."""
    domains = [
        (
            w.host,
            w.state.value,
            w.reason,
            w.failures,
            w.next_due_at,
            w.last_ok_at,
            w.canonical_host,
            w.checked_at,
            w.visit.robots_read,
            w.visit.robots_status,
            w.visit.robots_allows,
            w.visit.crawl_delay,
            w.visit.language,
            w.visit.declared_lang,
            w.visit.home_chars,
        )
        for w in writes
    ]
    sitemaps = [
        (
            u.id,
            u.kind,
            u.parent_id,
            u.depth,
            u.status,
            u.error,
            u.etag,
            u.last_modified,
            u.content_hash,
            u.changed,
            u.url_count,
            u.child_count,
            u.trust.value,
            u.index_lastmod,
            int(u.interval.total_seconds()),
            u.next_check_at,
            w.checked_at,
        )
        for w in writes
        for u in w.visit.sitemaps
    ]
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE visit_stage (
                    host text, state text, state_reason text, failures smallint,
                    next_due_at timestamptz, last_ok_at timestamptz, canonical_host text,
                    checked_at timestamptz, robots_read boolean, robots_status integer,
                    robots_allows boolean, crawl_delay real, language text,
                    declared_lang text, home_chars integer
                ) ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "visit_stage",
                records=domains,
                columns=[
                    "host",
                    "state",
                    "state_reason",
                    "failures",
                    "next_due_at",
                    "last_ok_at",
                    "canonical_host",
                    "checked_at",
                    "robots_read",
                    "robots_status",
                    "robots_allows",
                    "crawl_delay",
                    "language",
                    "declared_lang",
                    "home_chars",
                ],
            )
            await connection.execute(
                """
                UPDATE bot.domains d SET
                    state = s.state, state_reason = s.state_reason, failures = s.failures,
                    next_due_at = s.next_due_at, last_ok_at = s.last_ok_at,
                    canonical_host = s.canonical_host, checked_at = s.checked_at,
                    robots_checked_at = CASE WHEN s.robots_read THEN s.checked_at
                                             ELSE d.robots_checked_at END,
                    robots_status = CASE WHEN s.robots_read THEN s.robots_status
                                         ELSE d.robots_status END,
                    robots_allows = CASE WHEN s.robots_read THEN s.robots_allows
                                         ELSE d.robots_allows END,
                    crawl_delay = CASE WHEN s.robots_read THEN s.crawl_delay ELSE d.crawl_delay END,
                    language = coalesce(s.language, d.language),
                    declared_lang = coalesce(s.declared_lang, d.declared_lang),
                    home_chars = coalesce(s.home_chars, d.home_chars)
                FROM visit_stage s WHERE d.host = s.host AND d.state <> 'excluded'
                """
            )
            if sitemaps:
                await connection.execute(
                    """
                    CREATE TEMP TABLE sitemap_stage (
                        id bigint, kind text, parent_id bigint, depth smallint, status text,
                        error text, etag text, last_modified text, content_hash text,
                        changed boolean, url_count integer, child_count integer, trust text,
                        index_lastmod text, check_interval_s integer,
                        next_check_at timestamptz, fetched_at timestamptz
                    ) ON COMMIT DROP
                    """
                )
                await connection.copy_records_to_table(
                    "sitemap_stage",
                    records=sitemaps,
                    columns=[
                        "id",
                        "kind",
                        "parent_id",
                        "depth",
                        "status",
                        "error",
                        "etag",
                        "last_modified",
                        "content_hash",
                        "changed",
                        "url_count",
                        "child_count",
                        "trust",
                        "index_lastmod",
                        "check_interval_s",
                        "next_check_at",
                        "fetched_at",
                    ],
                )
                await connection.execute(
                    """
                    UPDATE bot.sitemaps m SET
                        kind = coalesce(s.kind, m.kind), parent_id = s.parent_id,
                        depth = s.depth, status = s.status, error = s.error, etag = s.etag,
                        last_modified = s.last_modified, content_hash = s.content_hash,
                        url_count = CASE WHEN s.changed THEN s.url_count ELSE m.url_count END,
                        child_count = CASE WHEN s.changed THEN s.child_count ELSE m.child_count END,
                        trust = s.trust, index_lastmod = s.index_lastmod,
                        check_interval_s = s.check_interval_s, next_check_at = s.next_check_at,
                        fetched_at = s.fetched_at,
                        changed_at = CASE WHEN s.changed THEN s.fetched_at ELSE m.changed_at END
                    FROM sitemap_stage s WHERE m.id = s.id
                    """
                )
            await connection.execute(
                """
                UPDATE bot.domains d SET url_count = coalesce((
                    SELECT sum(m.url_count) FROM bot.sitemaps m
                    WHERE m.host = d.host AND m.status = 'ok'
                ), 0)
                WHERE d.host = ANY($1::text[])
                """,
                [w.host for w in writes],
            )


async def adopt(pool: asyncpg.Pool, rows, now: datetime) -> None:
    """Add domains found through redirects, ranked like the domain that pointed at them."""
    rows = list(rows)
    if not rows:
        return
    hosts, sources, groups, states, reasons = (list(column) for column in zip(*rows))
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO bot.domains (host, rank, tld_group, state, state_reason, next_due_at)
            SELECT DISTINCT ON (t.host) t.host, d.rank, t.tld_group, t.state, t.reason,
                   CASE WHEN t.state = 'excluded' THEN NULL ELSE $6::timestamptz END
            FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[])
                AS t(host, source, tld_group, state, reason)
            LEFT JOIN bot.domains d ON d.host = t.source
            ORDER BY t.host, d.rank NULLS LAST
            ON CONFLICT (host) DO NOTHING
            """,
            hosts,
            sources,
            groups,
            [str(state) for state in states],
            reasons,
            now,
        )


async def exclude_hosts(pool: asyncpg.Pool, rows) -> int:
    """Flag domains that must never be crawled or published, keeping the reason."""
    rows = list(rows)
    if not rows:
        return 0
    hosts, reasons = (list(column) for column in zip(*rows))
    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE bot.domains d
            SET state = 'excluded', state_reason = s.reason, next_due_at = NULL
            FROM unnest($1::text[], $2::text[]) AS s(host, reason)
            WHERE d.host = s.host AND d.state <> 'excluded'
            """,
            hosts,
            reasons,
        )
    return int(result.split()[-1])
