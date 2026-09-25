from __future__ import annotations

import asyncio
import multiprocessing
import time

from desearch.extraction import extract
from neurons.validators.scoring import (
    FetchedPage,
    cleared_by_render,
    extraction_url,
    load_upload,
    needs_rendered_check,
    pick_samples,
    sample_count,
    score,
    url_log,
)

MEMORY_MB = 4096


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


def live_texts(
    kept: dict[str, dict], fetched: dict[str, FetchedPage]
) -> dict[str, str]:
    return {
        url: extract(page.html, extraction_url(kept[url])).text
        for url, page in fetched.items()
        if url in kept and page.html
    }


def _score_in_child(
    conn, data, assigned, seed, min_samples, match_ratio, memory_mb
) -> None:
    try:
        cap_memory(memory_mb)
        rows, kept = load_upload(data, assigned, seed)
        del data
        size = sample_count(len(kept), min_samples)
        conn.send(("samples", pick_samples(kept, seed, size)))
        fetched = conn.recv()
        conn.send(("doubtful", needs_rendered_check(kept, fetched)))
        fetched |= cleared_by_render(kept, conn.recv())
        result = score(rows, assigned, fetched, seed, min_samples, match_ratio)
        texts = live_texts(kept, fetched)
        result["urls"] = url_log(rows, result["samples"], texts, result["rejected"])
        conn.send(("scored", result))
    except MemoryError:
        conn.send(("too_large", None))
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
        if message is not None:
            await asyncio.to_thread(self.conn.send, message)
        started = time.monotonic()
        answer = await asyncio.to_thread(self._receive, max(0.0, self.budget))
        self.budget -= time.monotonic() - started
        if answer is None:
            raise Unscorable("timeout")
        kind, payload = answer
        if kind != expected:
            raise Unscorable(kind)
        return payload

    def _receive(self, timeout: float):
        if not self.conn.poll(timeout):
            return None
        try:
            return self.conn.recv()
        except (EOFError, OSError):
            raise Unscorable("died") from None

    async def aclose(self) -> None:
        if self.process.is_alive():
            self.process.kill()
        if self.process.pid is not None:
            await asyncio.to_thread(self.process.join, 5)
        self.conn.close()
