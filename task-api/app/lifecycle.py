from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from app.canonical import canonicalize

from desearch.manifest import FIELDS as MANIFEST_FIELDS
from desearch.manifest import OPEN_LIST_KEY
from desearch.manifest import payload as manifest_payload

from . import queues, rounds
from .budget import COVERAGE_GATE, CRAWL, EMBED, HOUR, STRIKE_REASONS, STRIKE_WINDOW_H
from .embeddings import DONE, DROPPED
from .validations import Decision, build_report, decide, utc_day

EMBED_FIELDS = ("model", "texts", "chars", "input_key", "input_sha256", "pages")
EMBED_ROUND_INPUTS = 200
FINALIZE_LOCK_S = 300

log = logging.getLogger("task_api")


class StorageDown(Exception):
    pass


class ClaimLost(Exception):
    pass


class AlreadySettled(Exception):
    pass


def open_round_key(round_id: str) -> str:
    return f"round:{round_id}:open"


def manifest_key(frozen_key: str) -> str:
    return frozen_key.removesuffix(".parquet") + ".manifest.json"


def signed_manifest(core, job: dict) -> dict:
    """What the miner was given and when it was frozen, signed by the task API."""
    manifest = {
        name: job[name] for name in MANIFEST_FIELDS if job.get(name) is not None
    }
    manifest["signer"] = core.key.ss58_address
    manifest["signature"] = core.key.sign(manifest_payload(manifest)).hex()
    return manifest


async def publish_open(core, now: float | None = None) -> bool:
    """Writes the open uploads, with their signed manifests, where validators read them."""
    uploads = []
    for task_id in await core.validation.open_ids():
        job = await core.validation.job(task_id)
        if job is not None and job.get("manifest"):
            uploads.append(job["manifest"])
    listed = tuple(m["key"] for m in uploads)
    if listed == core.open_listed:
        return False
    listing = {
        "generated_at": now or time.time(),
        "signer": core.key.ss58_address,
        "uploads": uploads,
    }
    try:
        await core.storage.put_json(OPEN_LIST_KEY, listing, cache_control="no-store")
    except Exception as exc:
        log.warning("could not publish the open list: %r", exc)
        return False
    core.open_listed = listed
    return True


async def open_round(
    core, urls: list[rounds.Url], batch_target: int = rounds.BATCH_TARGET
):
    # Two spellings of one page would race for the same key.
    unique = list({canonicalize(u.url): u for u in urls}.values())
    target = await core.seeds.target_block()
    round_ = await asyncio.to_thread(rounds.open_round, unique, target, batch_target)
    await core.db(core.rounds.save, round_)
    return round_


async def open_embed_rounds(core) -> rounds.Round | None:
    """Turns the publisher's waiting inputs into one embed round, one batch per input."""
    if not core.embed_tasks:
        return None
    waiting = await core.redis.lrange(queues.EMBED_INPUTS, 0, EMBED_ROUND_INPUTS - 1)
    if not waiting:
        return None
    batches = []
    for entry in map(json.loads, waiting):
        if not await core.db(core.embeddings.missing, entry["pages"], core.embed_model):
            continue
        pages = [
            {name: page[name] for name in ("page_key", "content_sha1", "url")}
            for page in entry["pages"]
        ]
        extra = {name: entry[name] for name in EMBED_FIELDS if name in entry}
        batches.append(
            rounds.Batch(
                uuid.uuid4().hex[:16],
                [rounds.Url(page["host"], page["url"]) for page in entry["pages"]],
                {**extra, "model": core.embed_model, "pages": pages},
            )
        )
    round_ = None
    if batches:
        target = await core.seeds.target_block()
        round_ = rounds.open_batches(batches, target, kind=EMBED)
        await core.db(core.rounds.save, round_)
        for batch in batches:
            await core.db(
                core.embeddings.queue,
                batch.extra["pages"],
                core.embed_model,
                batch.batch_id,
            )
    # Trimmed only once the round is saved, so a crash re-reads the inputs.
    await core.redis.ltrim(queues.EMBED_INPUTS, len(waiting), -1)
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
        rounds.reveal(round_, seed)
        # Saved first, so a crash cannot queue the round twice.
        await core.db(core.rounds.save, round_)
        filled += await fill_round(core, round_)
    return filled


async def fill_round(core, round_: rounds.Round) -> int:
    """Queues a revealed round's tasks; it counts as filled only once Redis holds them."""
    if round_.order:
        payloads = {
            batch_id: {
                **batch.extra,
                "url_count": len(batch.urls),
                "urls": [u.url for u in batch.urls],
            }
            for batch_id, batch in round_.batches.items()
        }
        await core.redis.sadd(open_round_key(round_.round_id), *round_.order)
        await core.tasks[round_.kind].fill(round_.round_id, round_.order, payloads)
    await core.db(core.rounds.mark_filled, round_.round_id, time.time())
    core.current[round_.kind] = round_.round_id
    return len(round_.order)


async def fill_missing(core) -> list[str]:
    """Rounds revealed before Redis took their tasks get them now, not closed unserved."""
    filled = []
    for round_ in await core.db(core.rounds.unfilled):
        if await core.redis.scard(open_round_key(round_.round_id)):
            await core.db(core.rounds.mark_filled, round_.round_id, time.time())
            continue
        await fill_round(core, round_)
        filled.append(round_.round_id)
    return filled


async def close_finished(core) -> list[str]:
    closed = []
    for round_id in await core.db(core.rounds.open_revealed):
        if await core.redis.scard(open_round_key(round_id)):
            continue
        await core.db(core.log.anchor, round_id)
        await core.db(core.rounds.close, round_id, time.time())
        # Later refusals must not land in a log whose root is anchored.
        for kind, current in list(core.current.items()):
            if current == round_id:
                core.current[kind] = ""
        closed.append(round_id)
    return closed


async def finish_task(core, round_id: str, task_id: str) -> None:
    await core.redis.srem(open_round_key(round_id), task_id)


async def reclaim_expired(core) -> list[tuple[str, str]]:
    reclaimed = []
    for task_id in await core.tasks[CRAWL].expired():
        payload = await core.payload(task_id) or {}
        kind = payload.get("kind", CRAWL)
        found = await core.tasks[kind].reclaim(task_id)
        if found is None:
            continue
        holder, seq = found
        if holder:
            if kind == CRAWL:
                await core.db(
                    core.budgets.record_coverage,
                    holder,
                    len(set(payload.get("urls", []))),
                    0,
                )
            await core.db(core.budgets.penalise, holder, task_id, "claim_expired", kind)
            await core.record(
                payload.get("round_id") or core.current.get(kind, ""),
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
    kind = job.get("kind", CRAWL)
    attempts = job.get("attempts", 0) + 1
    if attempts >= core.max_attempts:
        seq = await core.next_seq()
        if kind == EMBED:
            await core.db(core.embeddings.finalize, task_id, job["model"], DROPPED)
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
        **{name: job[name] for name in EMBED_FIELDS if name in job},
        "kind": kind,
        "url_count": len(job["urls"]),
        "urls": job["urls"],
        "round_id": job["round_id"],
        "position": job.get("position", 0),
        "rank": job.get("rank", job.get("position", 0)),
        "attempts": attempts,
    }
    seq = await core.tasks[kind].restore(task_id, payload)
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


def quorum(electorate: int) -> int:
    """More than half of the validators in play; one alone finalizes."""
    return max(1, electorate // 2 + 1)


def lock_key(task_id: str) -> str:
    return f"scoring:{task_id}"


async def finalize_due(core, now: float | None = None) -> list[str]:
    """Finalizes every open upload that is ready, oldest first."""
    now = now or time.time()
    finalized = []
    for task_id in await core.validation.open_ids():
        job = await core.validation.job(task_id)
        if job is None:
            continue
        lock = lock_key(task_id)
        if not await core.redis.set(lock, "janitor", nx=True, ex=FINALIZE_LOCK_S):
            continue
        try:
            outcome = await finalize_task(core, task_id, job, now)
        finally:
            await core.redis.delete(lock)
        if outcome is not None:
            finalized.append(task_id)
    return finalized


async def finalize_task(core, task_id: str, job: dict, now: float) -> dict | None:
    """Finalizes once every active validator voted, or at the deadline with a quorum."""
    voters = await core.validation.voters(task_id)
    active = await core.validation.active(now) | voters
    due = now >= job.get("deadline", 0)
    if not voters or (not due and not active <= voters):
        return None
    if len(voters) < quorum(len(active)):
        if not due:
            return None
        lapsed = await core.validation.finalize(task_id)
        if lapsed is not None:
            await void_task(core, task_id, lapsed.job, "", "no_quorum")
        return {"task_id": task_id, "verdict": "void", "credited": 0}
    votes = await core.validation.votes(task_id)
    try:
        return await conclude_validation(
            core, task_id, job, decide(votes, overdue=True)
        )
    except (StorageDown, ClaimLost):
        return None


async def conclude_validation(
    core, task_id: str, job: dict, decision: Decision
) -> dict:
    """Accounts are finalized once per upload, in one transaction, before Redis lets go."""
    finalized = await core.db(core.validations.final_verdict, task_id, job["key"])
    if finalized is not None:
        return await finish_finalized(core, task_id, job, finalized)

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
        publish = publish_job(core, task_id, job, vote, decision.agreed)
    try:
        budget = await core.db(
            finalize_accounts, core, task_id, job, decision, report, publish
        )
    except AlreadySettled:
        finalized = await core.db(core.validations.final_verdict, task_id, job["key"])
        return await finish_finalized(core, task_id, job, finalized)

    await close_upload(core, task_id, job, verdict, publish)
    return {
        "task_id": task_id,
        "verdict": verdict,
        "credited": vote["credited"] if verdict == "pass" else 0,
        "miner_budget": budget,
    }


def finalize_accounts(
    core,
    task_id: str,
    job: dict,
    decision: Decision,
    report: dict,
    publish: dict | None,
) -> int:
    """Every SQLite effect of one verdict, committed together; returns the miner's budget."""
    vote, result = decision.vote, decision.vote["result"]
    verdict, kind = vote["verdict"], job.get("kind", CRAWL)
    miner, assigned = job["miner"], len(set(job["urls"]))
    credited = vote["credited"] if verdict == "pass" else 0
    with core.sqlite.batch():
        if not core.validations.finalize(
            task_id, job["key"], verdict, credited, publish
        ):
            raise AlreadySettled(task_id)
        if kind == CRAWL and verdict != "void":
            core.budgets.record_coverage(miner, assigned, result["returned"])
        if verdict == "fail":
            budget = core.budgets.penalise(
                miner, task_id, "verification_failed", kind
            ).budget
        elif credited:
            # An embed pass is all or nothing, so every one grows the budget.
            ramp = kind == EMBED or credited >= COVERAGE_GATE * assigned
            budget = core.budgets.reward(miner, task_id, credited, ramp, kind).budget
        else:
            budget = core.budgets.get(miner, kind).budget
        if publish and kind == EMBED:
            core.embeddings.finalize(
                task_id, job["model"], DONE, publish["vectors_key"]
            )
        if decision.agreed or decision.disagreed:
            core.validations.record_audit(decision.agreed, decision.disagreed)
        core.validations.record(report, result.get("urls"))
        if verdict == "fail" and result.get("reason") in STRIKE_REASONS:
            since = time.time() - STRIKE_WINDOW_H * HOUR
            judged = core.validations.judged_since(miner, since, kind)
            core.budgets.strike(miner, result["reason"], task_id, judged, kind)
    return budget


def publish_job(
    core, task_id: str, job: dict, vote: dict, agreed: list[str] = ()
) -> dict:
    kind, result = job.get("kind", CRAWL), vote["result"]
    publish = {
        **{
            name: job.get(name, "")
            for name in ("task_id", "round_id", "miner", "key", "etag")
        },
        "kind": kind,
        "validator": vote["validator"],
        "validators": sorted(set(agreed) | {vote["validator"]}),
        "urls": job["urls"],
        "completed_at": job["completed_at"],
        "claim_ttl": core.claim_ttl,
    }
    if kind == EMBED:
        publish |= {
            name: job[name] for name in ("model", "input_key", "pages", "texts")
        }
        publish["vectors_key"] = vectors_key(job["model"], task_id)
    else:
        publish["skip"] = rejected_urls(result)
    return publish


def rejected_urls(result: dict) -> list[str]:
    """Rows a check failed stay out of the corpus even though the task passed."""
    mismatched = {
        sample["url"]
        for sample in result.get("samples", [])
        if sample["outcome"] == "mismatched"
    }
    return sorted(mismatched | set(result.get("rejected", [])))


async def close_upload(
    core, task_id: str, job: dict, verdict: str, publish: dict | None
) -> None:
    """Closes a finalized upload in Redis and moves the task on."""
    if await core.validation.finalize(task_id, publish) is None:
        log.warning("task=%s was closed before its final_verdict was recorded", task_id)
        raise ClaimLost(task_id)
    if verdict == "pass":
        await finish_task(core, job["round_id"], task_id)
    else:
        await requeue(core, task_id, job, verdict)


async def finish_finalized(core, task_id: str, job: dict, finalized: dict) -> dict:
    await close_upload(core, task_id, job, finalized["verdict"], finalized["publish"])
    kind = job.get("kind", CRAWL)
    return {
        "task_id": task_id,
        "verdict": finalized["verdict"],
        "credited": finalized["credited"],
        "miner_budget": (await core.db(core.budgets.get, job["miner"], kind)).budget,
    }


def vectors_key(model: str, task_id: str) -> str:
    return f"vectors/model={model}/dt={utc_day()}/task={task_id}.parquet"


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
