from __future__ import annotations

import asyncio
import json
import multiprocessing
import time

import pyarrow

from desearch.extraction import extract
from neurons.validators.confinement import confine
from neurons.validators.scoring import (
    FetchedPage,
    MISMATCHED,
    cleared_by_render,
    extraction_url,
    judge_sample,
    load_upload,
    needs_rendered_check,
    pick_samples,
    sample_count,
    score,
    url_log,
)

MEMORY_MB = 4096
MAX_ANSWER_BYTES = 64_000_000
FIRST_SHARE = 0.4


class Unscorable(Exception):
    pass


def cap_memory(megabytes: int) -> None:
    if not megabytes:
        return
    try:
        import resource

        limit = megabytes * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass


def read_on_one_thread() -> None:
    """pyarrow starts a thread per core, and each one reserves address space the cap counts."""
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


def live_texts(
    kept: dict[str, dict], fetched: dict[str, FetchedPage]
) -> dict[str, str]:
    return {
        url: extract(page.html, extraction_url(kept[url])).text
        for url, page in fetched.items()
        if url in kept and page.html
    }


def certainly_fails(
    kept: dict[str, dict],
    fetched: dict[str, FetchedPage],
    planned: int,
    match_ratio: float,
) -> bool:
    """Too many mismatches for the task to pass even if every page not checked yet matched."""
    mismatched = sum(
        1
        for url, page in fetched.items()
        if judge_sample(kept[url], page)["outcome"] == MISMATCHED
    )
    return mismatched > (1 - match_ratio) * planned


def answer(conn, kind: str, payload) -> None:
    """Plain data only: a child that reads a miner's file never hands the parent an object."""
    conn.send_bytes(json.dumps([kind, payload]).encode())


def _score_in_child(
    conn, data, assigned, seed, min_samples, match_ratio, memory_mb
) -> None:
    try:
        unconfined = confine()
        read_on_one_thread()
        cap_memory(memory_mb)
        rows, kept = load_upload(data, assigned, seed)
        del data
        planned = pick_samples(kept, seed, sample_count(len(kept), min_samples))
        answer(conn, "samples", {"urls": planned, "unconfined": unconfined})
        fetched: dict[str, FetchedPage] = {}
        while batch := conn.recv():
            answer(conn, "doubtful", needs_rendered_check(kept, batch))
            fetched |= batch | cleared_by_render(kept, conn.recv())
            settled = certainly_fails(kept, fetched, len(planned), match_ratio)
            answer(conn, "settled", settled)
        result = score(
            rows,
            assigned,
            fetched,
            seed,
            min_samples,
            match_ratio,
            only=set(fetched) if len(fetched) < len(planned) else None,
        )
        texts = live_texts(kept, fetched)
        result["urls"] = url_log(rows, result["samples"], texts, result["rejected"])
        answer(conn, "scored", result)
    except MemoryError:
        answer(conn, "too_large", None)
    finally:
        conn.close()


def _process_context():
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context(
        "forkserver" if "forkserver" in methods else "spawn"
    )
    if context.get_start_method() == "forkserver":
        context.set_forkserver_preload(["neurons.validators.scoring_process"])
    return context


class ScoringProcess:
    """Only the child's working time counts against the budget."""

    def __init__(
        self,
        data: bytes,
        assigned: list[str],
        seed: str,
        min_samples: int,
        match_ratio: float,
        budget: float,
        memory_mb: int = MEMORY_MB,
    ):
        context = _process_context()
        self.conn, self.child_conn = context.Pipe()
        self.process = context.Process(
            target=_score_in_child,
            args=(
                self.child_conn,
                data,
                assigned,
                seed,
                min_samples,
                match_ratio,
                memory_mb,
            ),
            daemon=True,
        )
        self.budget = budget

    async def start(self) -> ScoringProcess:
        # Starting pickles the whole upload into the child, so it runs off the event loop.
        await asyncio.to_thread(self.process.start)
        self.child_conn.close()
        return self

    async def ask(self, expected: str, message=None):
        await asyncio.to_thread(self.conn.send, message)
        return await self.wait_for(expected)

    async def wait_for(self, expected: str):
        started = time.monotonic()
        found = await asyncio.to_thread(self._receive, max(0.0, self.budget))
        self.budget -= time.monotonic() - started
        if found is None:
            raise Unscorable("timeout")
        kind, payload = found
        if kind != expected:
            raise Unscorable(kind)
        return payload

    def _receive(self, timeout: float):
        if not self.conn.poll(timeout):
            return None
        try:
            raw = self.conn.recv_bytes(MAX_ANSWER_BYTES)
        except (EOFError, OSError):
            raise Unscorable("died") from None
        try:
            kind, payload = json.loads(raw)
        except ValueError:
            raise Unscorable("garbled") from None
        return kind, payload

    async def aclose(self) -> None:
        if self.process.is_alive():
            self.process.kill()
        if self.process.pid is not None:
            await asyncio.to_thread(self.process.join, 5)
        self.conn.close()
