from __future__ import annotations

import asyncio
import logging
import time

from app.canonical import canonicalize

from . import rounds
from .budget import COVERAGE_GATE, HOUR, STRIKE_REASONS, STRIKE_WINDOW_H
from .validations import Decision, build_report, decide

log = logging.getLogger("task_api")


class StorageDown(Exception):
    pass


class LeaseLost(Exception):
    pass


def open_round_key(round_id: str) -> str:
    return f"round:{round_id}:open"


async def open_round(
    core, urls: list[rounds.Url], batch_target: int = rounds.BATCH_TARGET
):
    # Two spellings of one page would race for the same key.
    unique = list({canonicalize(u.url): u for u in urls}.values())
    target = await core.seeds.target_block()
    round_ = await asyncio.to_thread(rounds.open_round, unique, target, batch_target)
    await core.db(core.rounds.save, round_)
    return round_


async def reveal_pending(core) -> int:
    pending = await core.db(core.rounds.unrevealed)
    if not pending:
        return 0
    current = await core.seeds.current_block()
    filled = 0
    for round_ in pending:
        if round_.seed_block > current:
            continue
        seed = await core.seeds.seed_for(round_.seed_block)
        if seed is None:
            continue
        order = rounds.reveal(round_, seed)
        # Saved first, so a crash cannot queue the round twice.
        await core.db(core.rounds.save, round_)
        if order:
            payloads = {
                batch_id: {
                    "url_count": len(batch.urls),
                    "urls": [u.url for u in batch.urls],
                }
                for batch_id, batch in round_.batches.items()
            }
            await core.redis.sadd(open_round_key(round_.round_id), *order)
            filled += await core.queue.fill(round_.round_id, order, payloads)
        core.current = round_.round_id
    return filled


async def close_finished(core) -> list[str]:
    closed = []
    for round_id in await core.db(core.rounds.open_revealed):
        if await core.redis.scard(open_round_key(round_id)):
            continue
        await core.db(core.log.anchor, round_id)
        await core.db(core.rounds.close, round_id, time.time())
        closed.append(round_id)
    return closed


async def finish_task(core, round_id: str, task_id: str) -> None:
    await core.redis.srem(open_round_key(round_id), task_id)


async def reclaim_expired(core) -> list[tuple[str, str]]:
    reclaimed = []
    for task_id in await core.queue.expired():
        payload = await core.queue.payload(task_id) or {}
        found = await core.queue.reclaim(task_id)
        if found is None:
            continue
        holder, seq = found
        if holder:
            await core.db(
                core.budgets.record_coverage,
                holder,
                len(set(payload.get("urls", []))),
                0,
            )
            await core.db(core.budgets.penalise, holder, task_id, "lease_expired")
            await core.record(
                payload.get("round_id") or core.current or "",
                holder,
                0.0,
                "reclaimed",
                seq,
                task_id=task_id,
                cause="expired",
            )
        reclaimed.append((task_id, holder))
    return reclaimed


async def requeue(core, task_id: str, job: dict, cause: str) -> None:
    attempts = job.get("attempts", 0) + 1
    if attempts >= core.max_attempts:
        seq = await core.queue.next_seq()
        await core.record(
            job["round_id"],
            job["miner"],
            0.0,
            "dropped",
            seq,
            task_id=task_id,
            cause=cause,
        )
        await finish_task(core, job["round_id"], task_id)
        return
    payload = {
        "url_count": len(job["urls"]),
        "urls": job["urls"],
        "round_id": job["round_id"],
        "position": job.get("position", 0),
        "rank": job.get("rank", job.get("position", 0)),
        "attempts": attempts,
    }
    seq = await core.queue.restore(task_id, payload)
    await core.record(
        job["round_id"],
        job["miner"],
        0.0,
        "reclaimed",
        seq,
        task_id=task_id,
        cause=cause,
    )


async def write_report(core, report: dict) -> None:
    try:
        await core.pages.put_json(report["report_key"], report)
    except Exception:
        log.exception("could not write the report for %s", report["task_id"])
        report["report_key"] = ""


async def void_task(core, task_id: str, job: dict, validator: str, reason: str) -> dict:
    await requeue(core, task_id, job, "void")
    report = build_report(
        task_id, job, validator, {"verdict": "void", "reason": reason}
    )
    await write_report(core, report)
    await core.db(core.validations.record, report)
    return report


async def conclude_validation(
    core, task_id: str, job: dict, decision: Decision, holder: str
) -> dict:
    vote, result = decision.vote, decision.vote["result"]
    verdict = vote["verdict"]
    report = build_report(task_id, job, vote["validator"], result, decision.votes)
    try:
        await core.pages.put_json(report["report_key"], report)
    except Exception:
        log.exception("storage failed while scoring %s", task_id)
        raise StorageDown(task_id) from None

    publish = None
    if verdict == "pass" and result.get("matched"):
        publish = {
            **{
                name: job.get(name, "")
                for name in ("task_id", "round_id", "miner", "key", "etag")
            },
            "urls": job["urls"],
            "completed_at": job["completed_at"],
            "lease_ttl": core.lease_ttl,
        }
    if await core.validation.finalize(task_id, holder, publish) is None:
        await delete_quietly(core.pages, report["report_key"])
        raise LeaseLost(task_id)

    miner, assigned = job["miner"], len(set(job["urls"]))
    credited = vote["credited"] if verdict == "pass" else 0
    if verdict != "void":
        await core.db(core.budgets.record_coverage, miner, assigned, result["returned"])
    if verdict == "fail":
        budget = (
            await core.db(core.budgets.penalise, miner, task_id, "verification_failed")
        ).budget
    elif credited:
        ramp = credited >= COVERAGE_GATE * assigned
        budget = (
            await core.db(core.budgets.reward, miner, task_id, credited, ramp)
        ).budget
    else:
        budget = (await core.db(core.budgets.get, miner)).budget
    if decision.agreed or decision.disagreed:
        await core.db(
            core.validations.record_audit, decision.agreed, decision.disagreed
        )
    await core.db(core.validations.record, report, result.get("urls"))
    if verdict == "fail" and result.get("reason") in STRIKE_REASONS:
        since = time.time() - STRIKE_WINDOW_H * HOUR
        judged = await core.db(core.validations.judged_since, miner, since)
        await core.db(core.budgets.strike, miner, result["reason"], task_id, judged)

    if verdict == "pass":
        await finish_task(core, job["round_id"], task_id)
    else:
        await requeue(core, task_id, job, verdict)
    return {
        "task_id": task_id,
        "verdict": verdict,
        "credited": credited,
        "miner_budget": budget,
    }


async def return_expired_validations(core) -> list[str]:
    returned = []
    for task_id in await core.validation.expired():
        tries = await core.validation.give_back(task_id)
        if not tries:
            continue
        returned.append(task_id)
        if tries >= core.validation.max_tries:
            settled = await core.validation.settle(task_id)
            if settled is not None:
                await void_task(core, task_id, settled.job, "", "validators_lapsed")
    return returned


async def conclude_overdue_audits(core) -> list[str]:
    concluded = []
    for task_id in await core.validation.overdue_audits():
        job = await core.validation.job(task_id)
        votes = await core.validation.votes(task_id)
        if job is None or not votes:
            continue
        try:
            await conclude_validation(
                core, task_id, job, decide(votes, overdue=True), ""
            )
        except (StorageDown, LeaseLost):
            continue
        concluded.append(task_id)
    return concluded


async def return_expired_publishes(core) -> list[str]:
    returned = []
    for task_id in await core.publish.expired():
        outcome = await core.publish.give_back(task_id)
        if outcome < 0:
            log.error(
                "task=%s failed to publish %d times; set aside",
                task_id,
                core.publish.max_tries,
            )
        elif outcome:
            returned.append(task_id)
    return returned


async def delete_quietly(storage, key: str) -> None:
    try:
        await storage.delete(key)
    except Exception:
        log.warning("could not delete %s from %s", key, storage.bucket)
