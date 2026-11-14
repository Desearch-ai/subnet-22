"""Sits the capacity test beside the embed miner; opt in with CAPACITY_TEST=1. See docs/capacity-test.md."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import aiohttp
from yarl import URL

from desearch import capacity
from desearch.capacity_chain import CapacityChain
from desearch.client import TaskApiError
from desearch.embedding import encode_vector

log = logging.getLogger("capacity")

REVEAL_PATH = "/v1/capacity/reveal"
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60.0, sock_connect=15.0)
RESTART_S = 5.0
FINALITY_POLL_S = 3.0
REVEAL_ATTEMPTS = 3
REVEAL_RETRY_S = 5.0
# A new block normally shows the commitment first; the wait covers a stalled stream.
COMMIT_WAIT_S = 14.0


class CapacityTester:
    def __init__(
        self,
        settings,
        embedder,
        api,
        chain: CapacityChain,
        stopping: asyncio.Event,
        may_pause=lambda: True,
    ):
        self.settings = settings
        self.embedder = embedder
        self.api = api
        self.chain = chain
        self.stopping = stopping
        self.may_pause = may_pause
        self.hotkey: str = api.hotkey
        self.hashes: deque[tuple[int, bytes]] = deque(maxlen=capacity.QUIET_BLOCKS)
        self.head = 0
        self.head_moved = asyncio.Condition()
        self.sitting: asyncio.Task | None = None
        self.http: aiohttp.ClientSession | None = None

    async def run(self) -> None:
        """Follows the chain until the miner stops; a node that is down or breaks off never reaches the miner's real work."""
        while not self.stopping.is_set():
            try:
                async with self.chain:
                    await self.follow()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("capacity test: following the chain failed")
            self.hashes.clear()
            if not self.stopping.is_set():
                await asyncio.sleep(RESTART_S)

    async def follow(self) -> None:
        async for block in self.chain.heads():
            if self.stopping.is_set():
                return
            await self.on_head(block)

    async def on_head(self, block: int) -> None:
        if block <= self.head and self.hashes:
            return
        # A stream under load can skip blocks, a test's among them.
        first = block
        if self.hashes:
            first = max(self.head + 1, block - capacity.WINDOW_BLOCKS + 1)
        test = None
        for number in range(first, block + 1):
            block_hash = await self.chain.block_hash(number)
            earlier = await self.earlier_hashes(number)
            self.hashes.append((number, block_hash))
            test = capacity.test_at(number, block_hash, earlier) or test
        async with self.head_moved:
            self.head = block
            self.head_moved.notify_all()
        if test is not None and (self.sitting is None or self.sitting.done()):
            self.sitting = asyncio.create_task(self.sit(test))

    async def earlier_hashes(self, block: int) -> list[bytes]:
        """The hashes of the quiet run before a block, filling any the stream skipped or this process never saw."""
        known = dict(self.hashes)
        wanted = range(max(0, block - capacity.QUIET_BLOCKS), block)
        missing = [n for n in wanted if n not in known]
        if missing:
            found = await asyncio.gather(*(self.chain.block_hash(n) for n in missing))
            known.update(zip(missing, found))
            self.hashes.clear()
            self.hashes.extend((n, known[n]) for n in wanted)
        return [known[n] for n in wanted]

    async def sit(self, test: capacity.Test) -> None:
        try:
            log.info("capacity test %d: started", test.block)
            # A claim close to expiring keeps going: losing it costs more than this test's score.
            pausing = self.may_pause()
            if pausing:
                self.embedder.pause()
            try:
                # A commitment sent at the deadline block itself would come too late.
                stop_blocks = max(1, self.settings.capacity_stop_blocks)
                raws = await self.embedded_until(test, test.deadline - stop_blocks)
            finally:
                if pausing:
                    self.embedder.resume()
            leaves = [capacity.leaf_hash(index, raw) for index, raw in enumerate(raws)]
            levels = capacity.tree_levels(leaves) if leaves else [[b"\0" * 32]]
            tree_root = levels[-1][0]
            count = len(raws)
            log.info(
                "capacity test %d: %d texts, %d characters",
                test.block,
                count,
                capacity.chars_of(test.seed, self.hotkey, count),
            )
            if not count:
                return
            root = capacity.commit_root(test.seed, self.hotkey, count, tree_root)
            if await self.committed(test, capacity.make_tag(test, count), root):
                await self.revealed(test, raws, levels, tree_root, count)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("capacity test %d failed", test.block)

    async def embedded_until(self, test: capacity.Test, stop_block: int) -> list[bytes]:
        """Vectors of texts 0..n-1, as many in order as were done when the head reached the stop block."""
        done: dict[int, bytes] = {}
        retry: deque[int] = deque()
        size = self.embedder.batch
        turn = {"next": 0}

        async def work() -> None:
            while True:
                start = retry.popleft() if retry else turn["next"]
                if start == turn["next"]:
                    turn["next"] += size
                texts = [
                    capacity.text_at(test.seed, self.hotkey, i)
                    for i in range(start, start + size)
                ]
                try:
                    vectors = await self.embedder.embed_now(texts)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                    log.warning("capacity test %d: a batch failed: %r", test.block, exc)
                    retry.append(start)
                    await asyncio.sleep(0.5)
                    continue
                for offset, vector in enumerate(vectors):
                    done[start + offset] = encode_vector(vector)

        workers = [
            asyncio.create_task(work())
            for _ in range(self.settings.capacity_in_flight * self.embedder.servers)
        ]
        # Real tasks are held up meanwhile, so a stalled node or a stopping miner must end this too.
        reached = asyncio.create_task(self.wait_for_head(stop_block))
        stopped = asyncio.create_task(self.stopping.wait())
        limit = (stop_block - test.block + 1) * capacity.SECONDS_PER_BLOCK
        try:
            await asyncio.wait(
                {reached, stopped}, timeout=limit, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (reached, stopped, *workers):
                task.cancel()
            await asyncio.gather(reached, stopped, *workers, return_exceptions=True)
        raws: list[bytes] = []
        while len(raws) in done:
            raws.append(done[len(raws)])
        return raws

    async def wait_for_head(self, block: int) -> None:
        async with self.head_moved:
            await self.head_moved.wait_for(lambda: self.head >= block)

    async def committed(self, test: capacity.Test, tag: bytes, root: bytes) -> bool:
        """Sends the commitment again if a new block does not show it, until the deadline; a busy pool drops calls."""
        blocks_left = test.deadline - self.head + 1
        give_up = time.monotonic() + blocks_left * capacity.SECONDS_PER_BLOCK
        while (
            self.head < test.deadline
            and not self.stopping.is_set()
            and time.monotonic() < give_up
        ):
            try:
                await self.chain.commit(tag, root)
            except Exception as exc:
                log.warning(
                    "capacity test %d: sending the commitment failed: %r",
                    test.block,
                    exc,
                )
            try:
                await asyncio.wait_for(self.wait_for_head(self.head + 1), COMMIT_WAIT_S)
            except asyncio.TimeoutError:
                pass
            found = await self.chain.commitment(self.hotkey)
            if found is not None and found.tag == tag and found.root == root:
                log.info(
                    "capacity test %d: committed in block %d", test.block, found.block
                )
                return True
        found = await self.chain.commitment(self.hotkey, test.deadline)
        if found is not None and found.tag == tag and found.root == root:
            return True
        log.warning("capacity test %d: the commitment missed the deadline", test.block)
        return False

    async def revealed(
        self, test, raws: list[bytes], levels, tree_root: bytes, count: int
    ) -> None:
        blocks_left = test.reveal_by - self.head + 1
        give_up = time.monotonic() + blocks_left * capacity.SECONDS_PER_BLOCK
        while await self.chain.finalized() < test.sample_block:
            if self.stopping.is_set():
                return
            if time.monotonic() > give_up:
                log.warning(
                    "capacity test %d: the chain stopped finalizing", test.block
                )
                return
            await asyncio.sleep(FINALITY_POLL_S)
        if await self.chain.block_hash(test.block) != test.seed:
            log.warning(
                "capacity test %d: its block left the chain, nothing counts", test.block
            )
            return
        sample_hash = await self.chain.block_hash(test.sample_block)
        picked = capacity.sample_indices(sample_hash, self.hotkey, count)
        items = {
            index: (raws[index], capacity.tree_proof(levels, index)) for index in picked
        }
        body = capacity.encode_reveal(tree_root, count, items)
        for attempt in range(1, REVEAL_ATTEMPTS + 1):
            try:
                await self.upload_reveal(test, body)
                log.info(
                    "capacity test %d: revealed %d vectors", test.block, len(items)
                )
                return
            except (TaskApiError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("capacity test %d: revealing failed: %s", test.block, exc)
                if attempt < REVEAL_ATTEMPTS:
                    await asyncio.sleep(REVEAL_RETRY_S * attempt)

    async def upload_reveal(self, test: capacity.Test, body: bytes) -> None:
        link = await self.api.post(REVEAL_PATH, {"test_block": test.block})
        if self.http is None:
            self.http = aiohttp.ClientSession(timeout=UPLOAD_TIMEOUT)
        headers = {"Content-Type": link["content_type"]}
        async with self.http.put(
            URL(link["url"], encoded=True), data=body, headers=headers
        ) as answer:
            answer.raise_for_status()
        await self.api.post(REVEAL_PATH, {"test_block": test.block, "uploaded": True})

    async def aclose(self) -> None:
        if self.sitting is not None:
            self.sitting.cancel()
            await asyncio.gather(self.sitting, return_exceptions=True)
        if self.http is not None:
            await self.http.close()
