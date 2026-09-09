from __future__ import annotations

from . import rounds


async def open_round(core, urls: list[rounds.Url], batch_target: int = rounds.BATCH_TARGET):
    round_ = rounds.open_round(urls, core.seeds, batch_target)
    core.rounds[round_.round_id] = round_
    return round_


async def reveal_and_fill(core, round_id: str) -> int:
    round_ = core.rounds[round_id]
    if round_.revealed:
        return 0
    seed = await core.seeds.seed_for(round_.seed_block)
    if seed is None:
        return 0

    order = rounds.reveal(round_, seed)
    payloads = {}
    for batch_id, batch in round_.batches.items():
        payloads[batch_id] = {
            "hosts": batch.hosts,
            "url_count": len(batch.urls),
            "urls": [{"host": u.host, "url": u.url} for u in batch.urls],
            "crawl_delay": {u.host: u.crawl_delay for u in batch.urls},
        }
    filled = await core.queue.fill(round_id, order, payloads)
    core.current = round_id
    return filled


async def close_round(core, round_id: str) -> str:
    import time

    round_ = core.rounds[round_id]
    round_.closed_at = time.time()
    return core.log.anchor(round_id)


async def reclaim_expired(core) -> list[tuple[str, str]]:
    reclaimed = []
    for task_id in await core.queue.expired():
        holder = await core.queue.holder(task_id) or ""
        seq = await core.queue.requeue(task_id)
        if holder:
            core.budgets.penalise(holder, task_id, "lease_expired")
            core.log.record(
                core.current or "", holder, 0.0, "reclaimed",
                core.receipt({"task_id": task_id, "outcome": "reclaimed", "cause": "expired"}),
                task_id=task_id, seq=seq,
            )
        reclaimed.append((task_id, holder))
    return reclaimed
