"""Postgres access for the control plane."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

from .buckets import bucket_of

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
    rows = [(host, rank, group, bucket_of(host)) for host, rank, group in rows]
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE candidate_stage (
                    host text, rank integer, tld_group text, bucket smallint
                ) ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "candidate_stage",
                records=rows,
                columns=["host", "rank", "tld_group", "bucket"],
            )
            await connection.execute(
                """
                INSERT INTO bot.domains (host, rank, tld_group, bucket)
                SELECT DISTINCT ON (host) host, rank, tld_group, bucket
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
            UPDATE bot.domains d
            SET categories = r.labels, category = r.labels[1], changed_at = now()
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
                UPDATE bot.domains d
                SET categories = s.categories, category = s.category, changed_at = now()
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
            SELECT count(*) AS domains, coalesce(sum(url_count), 0) AS urls FROM bot.domains
            """
        )
    return {**dict(totals), **{row["state"]: row["n"] for row in states}}


async def excluded_categories(pool: asyncpg.Pool) -> frozenset[str]:
    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT category FROM bot.excluded_categories")
    return frozenset(row["category"] for row in rows)


async def adopt(pool: asyncpg.Pool, rows, now: datetime) -> None:
    """Add domains found through redirects, ranked like the domain that pointed at them."""
    rows = list(rows)
    if not rows:
        return
    hosts, sources, groups, states, reasons = (list(column) for column in zip(*rows))
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO bot.domains (host, rank, tld_group, state, state_reason, bucket, changed_at)
            SELECT DISTINCT ON (t.host) t.host, d.rank, t.tld_group, t.state, t.reason, t.bucket, $7
            FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::smallint[])
                AS t(host, source, tld_group, state, reason, bucket)
            LEFT JOIN bot.domains d ON d.host = t.source
            ORDER BY t.host, d.rank NULLS LAST
            ON CONFLICT (host) DO NOTHING
            """,
            hosts,
            sources,
            groups,
            [str(state) for state in states],
            reasons,
            [bucket_of(host) for host in hosts],
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
            SET state = 'excluded', state_reason = s.reason, changed_at = now()
            FROM unnest($1::text[], $2::text[]) AS s(host, reason)
            WHERE d.host = s.host AND d.state <> 'excluded'
            """,
            hosts,
            reasons,
        )
    return int(result.split()[-1])


async def report_visits(pool: asyncpg.Pool, rows) -> None:
    """Record what visits found, the latest per domain; an exclusion made meanwhile stands."""
    latest = {row[0]: row for row in rows}
    if not latest:
        return
    hosts, states, reasons, checked, urls, canonical = (
        list(c) for c in zip(*latest.values())
    )
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE bot.domains d SET
                state = s.state, state_reason = s.reason, checked_at = s.checked_at,
                url_count = s.urls, canonical_host = s.canonical
            FROM unnest($1::text[], $2::text[], $3::text[], $4::timestamptz[], $5::bigint[],
                        $6::text[]) AS s(host, state, reason, checked_at, urls, canonical)
            WHERE d.host = s.host AND d.state <> 'excluded'
            """,
            hosts,
            states,
            reasons,
            checked,
            urls,
            canonical,
        )


async def registry_changes(pool: asyncpg.Pool, buckets, since, limit: int):
    """Domains in these buckets changed centrally after a point, oldest change first."""
    after, host = since or (datetime(1970, 1, 1, tzinfo=timezone.utc), "")
    async with pool.acquire() as connection:
        return await connection.fetch(
            """
            SELECT host, rank, tld_group, state, state_reason, categories, changed_at
            FROM bot.domains
            WHERE bucket = ANY($1::smallint[]) AND changed_at >= $2
              AND (changed_at > $2 OR host COLLATE "C" > $3)
            ORDER BY changed_at, host COLLATE "C"
            LIMIT $4
            """,
            list(buckets),
            after,
            host,
            limit,
        )
