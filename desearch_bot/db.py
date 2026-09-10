"""Postgres access for the control plane."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
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


async def take_candidates(pool: asyncpg.Pool, limit: int):
    async with pool.acquire() as connection:
        return await connection.fetch(
            """
            SELECT host, rank, tld_group
            FROM bot.domains
            WHERE status = 'candidate'
            ORDER BY rank NULLS LAST
            LIMIT $1
            """,
            limit,
        )


async def save_domains(pool: asyncpg.Pool, results) -> None:
    """One statement for a batch of visit outcomes."""
    rows = [
        (
            r.host,
            "qualified" if r.qualified else "redirect" if r.canonical_host else "rejected",
            r.reject_reason,
            r.canonical_host,
            r.robots_status,
            r.robots_allows,
            r.crawl_delay,
            r.language,
            r.declared_lang,
            r.home_chars,
            r.sitemap_url,
            r.sitemap_kind,
            url_count,
        )
        for r, url_count in results
    ]
    if not rows:
        return
    async with pool.acquire() as connection:
        await connection.executemany(
            """
            UPDATE bot.domains SET
                status = $2, reject_reason = $3, canonical_host = $4, robots_status = $5,
                robots_allows = $6, crawl_delay = $7, language = $8, declared_lang = $9,
                home_chars = $10, sitemap_url = $11, sitemap_kind = $12, url_count = $13,
                checked_at = now()
            WHERE host = $1
            """,
            rows,
        )


async def save_sitemap(
    pool: asyncpg.Pool,
    host: str,
    url: str,
    kind: str | None,
    depth: int,
    url_count: int = 0,
    child_count: int = 0,
    parent_id: int | None = None,
    status: str = "ok",
    error: str | None = None,
    etag: str | None = None,
    last_modified: str | None = None,
    content_hash: str | None = None,
    check_interval_s: int = 86400,
) -> int | None:
    async with pool.acquire() as connection:
        return await connection.fetchval(
            """
            INSERT INTO bot.sitemaps (
                host, url, kind, parent_id, depth, url_count, child_count, status, error,
                etag, last_modified, content_hash, fetched_at, changed_at,
                check_interval_s, next_check_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now(), now(), $13, $14)
            ON CONFLICT (url) DO UPDATE SET
                kind = EXCLUDED.kind,
                url_count = EXCLUDED.url_count,
                child_count = EXCLUDED.child_count,
                status = EXCLUDED.status,
                error = EXCLUDED.error,
                etag = EXCLUDED.etag,
                last_modified = EXCLUDED.last_modified,
                content_hash = EXCLUDED.content_hash,
                fetched_at = now(),
                changed_at = CASE
                    WHEN bot.sitemaps.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                    THEN now() ELSE bot.sitemaps.changed_at END,
                unchanged_checks = CASE
                    WHEN bot.sitemaps.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                    THEN 0 ELSE bot.sitemaps.unchanged_checks + 1 END
            RETURNING id
            """,
            host,
            url,
            kind,
            parent_id,
            depth,
            url_count,
            child_count,
            status,
            error,
            etag,
            last_modified,
            content_hash,
            check_interval_s,
            datetime.now(timezone.utc) + timedelta(seconds=check_interval_s),
        )


async def record_frontier_files(pool: asyncpg.Pool, rows) -> None:
    if not rows:
        return
    async with pool.acquire() as connection:
        await connection.executemany(
            """
            INSERT INTO bot.frontier_files (path, host_bucket, url_count, bytes)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (path) DO NOTHING
            """,
            rows,
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


async def iter_unresolved_hosts(pool: asyncpg.Pool, chunk: int = 5_000):
    """Only domains not yet checked, so a restart continues rather than starting over."""
    while True:
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT host FROM bot.domains WHERE resolved_at IS NULL
                ORDER BY host LIMIT $1
                """,
                chunk,
            )
        if not rows:
            return
        yield rows


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


async def iter_resolving_hosts(pool: asyncpg.Pool, chunk: int = 5_000):
    """Domains still to canonicalise. Marking every one we check, redirect or not, is what makes
    a restart continue instead of walking the whole list again."""
    while True:
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT host FROM bot.domains
                WHERE resolves IS TRUE AND canonicalised_at IS NULL
                ORDER BY host LIMIT $1
                """,
                chunk,
            )
        if not rows:
            return
        yield rows


async def save_resolution(pool: asyncpg.Pool, rows) -> int:
    if not rows:
        return 0
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "CREATE TEMP TABLE resolution_stage (host text, resolves boolean) ON COMMIT DROP"
            )
            await connection.copy_records_to_table(
                "resolution_stage", records=rows, columns=["host", "resolves"]
            )
            result = await connection.execute(
                """
                UPDATE bot.domains d SET resolves = s.resolves, resolved_at = now()
                FROM resolution_stage s WHERE d.host = s.host
                """
            )
    return int(result.split()[-1])


async def save_canonical(pool: asyncpg.Pool, rows) -> int:
    """Record the name a domain actually serves under. A domain that redirects elsewhere stops
    being a crawl target of its own."""
    rows = list(rows)
    if not rows:
        return 0
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TEMP TABLE canonical_stage (
                    host text, canonical_host text, http_ok boolean
                ) ON COMMIT DROP
                """
            )
            await connection.copy_records_to_table(
                "canonical_stage",
                records=rows,
                columns=["host", "canonical_host", "http_ok"],
            )
            result = await connection.execute(
                """
                UPDATE bot.domains d SET
                    canonical_host = s.canonical_host,
                    http_ok = s.http_ok,
                    canonicalised_at = now(),
                    status = CASE
                        WHEN s.canonical_host IS NOT NULL
                             AND d.status IN ('candidate', 'qualified') THEN 'redirect'
                        ELSE d.status END
                FROM canonical_stage s WHERE d.host = s.host
                """
            )
    return int(result.split()[-1])


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
                columns=["host", "source", "category", "raw_category",
                         "category_id", "super_category"],
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


async def delete_by_category(pool: asyncpg.Pool, labels) -> dict:
    """Drop every domain carrying an excluded label, with the sitemaps that belong to it."""
    labels = list(labels)
    async with pool.acquire() as connection:
        async with connection.transaction():
            sitemaps = await connection.execute(
                """
                DELETE FROM bot.sitemaps WHERE host IN (
                    SELECT host FROM bot.domains WHERE categories && $1::text[]
                )
                """,
                labels,
            )
            domains = await connection.execute(
                "DELETE FROM bot.domains WHERE categories && $1::text[]", labels
            )
    return {
        "domains": int(domains.split()[-1]),
        "sitemaps": int(sitemaps.split()[-1]),
    }


async def delete_hosts(pool: asyncpg.Pool, hosts) -> dict:
    hosts = list(hosts)
    if not hosts:
        return {"domains": 0, "sitemaps": 0}
    async with pool.acquire() as connection:
        async with connection.transaction():
            sitemaps = await connection.execute(
                "DELETE FROM bot.sitemaps WHERE host = ANY($1::text[])", hosts
            )
            domains = await connection.execute(
                "DELETE FROM bot.domains WHERE host = ANY($1::text[])", hosts
            )
    return {
        "domains": int(domains.split()[-1]),
        "sitemaps": int(sitemaps.split()[-1]),
    }


async def counts(pool: asyncpg.Pool) -> dict:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT
                (SELECT count(*) FROM bot.domains) AS domains,
                (SELECT count(*) FROM bot.domains WHERE status = 'candidate') AS candidates,
                (SELECT count(*) FROM bot.domains WHERE status = 'qualified') AS qualified,
                (SELECT count(*) FROM bot.domains WHERE status = 'rejected') AS rejected,
                (SELECT count(*) FROM bot.sitemaps) AS sitemaps,
                (SELECT coalesce(sum(url_count), 0) FROM bot.domains) AS urls
            """
        )
        return dict(row)
