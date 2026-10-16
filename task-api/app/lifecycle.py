from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import logging
import time
import uuid

from app.canonical import canonicalize

from desearch import credit
from desearch.manifest import FIELDS as MANIFEST_FIELDS
from desearch.manifest import OPEN_LIST_KEY
from desearch.manifest import payload as manifest_payload

from . import outcomes, queues, rounds, sampling
from .budget import (
    COVERAGE_GATE,
    CRAWL,
    EMBED,
    FULL_PENALTY_LOCKOUT_H,
    HOSTILE_LOCKOUT_H,
    HOUR,
    STRIKE_REASONS,
    STRIKE_WINDOW_H,
)
from .embeddings import DONE, DROPPED
from .validations import (
    ERROR_OUTCOMES,
    NO_MAJORITY,
    Decision,
    build_report,
    decide,
    utc_day,
)

EMBED_FIELDS = ("model", "texts", "chars", "input_key", "input_sha256", "pages")
EMBED_ROUND_INPUTS = 200
FINALIZE_LOCK_S = 300
SETTLE_AT_ONCE = 16
SLOW_ENQUEUE_S = 2.0
# A check takes about a minute; an upload with less time left would be finalized before its votes land.
CHECKABLE_LEFT_S = 180
UNCHECKED = "unchecked"
REPORTED_ROWS = "reported_rows"
# Uploads with no counts in their report, or from a locked-out hotkey, are all checked.
UNREPORTED, LOCKED = "unreported", "locked"

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
    at = now or time.time()
    open_ids = await core.validation.open_ids()
    known = core.open_manifests
    for task_id in set(known) - set(open_ids):
        del known[task_id]
    for task_id in open_ids:
        if task_id not in known:
            job = await core.validation.job(task_id)
            if job is None or not job.get("manifest"):
                continue
            known[task_id] = (job["manifest"], job.get("deadline", math.inf))
        manifest, deadline = known[task_id]
        if deadline - at >= CHECKABLE_LEFT_S:
            uploads.append(manifest)
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


def pack_round(urls: list[rounds.Url], target: int) -> rounds.Round:
    # Two spellings of one page would race for the same key.
    unique = list({canonicalize(u.url): u for u in urls}.values())
    return rounds.open_round(unique, target)


async def open_round(core, urls: list[rounds.Url]):
    started = time.monotonic()
    target = await core.seeds.target_block()
    blocked = time.monotonic()
    round_ = await asyncio.to_thread(pack_round, urls, target)
    packed = time.monotonic()
    await core.db(core.rounds.save, round_)
    took = time.monotonic() - started
    if took > SLOW_ENQUEUE_S:
        log.warning(
            "enqueueing %d urls took %.1fs (block %.1fs, pack %.1fs, save %.1fs)",
            len(urls),
            took,
            blocked - started,
            packed - blocked,
            time.monotonic() - packed,
        )
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
    current = await core.seeds.current_block()
    filled = 0
    for round_ in await core.db(core.rounds.unrevealed, current):
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


async def reclaim_expired(core, now: float | None = None) -> list[tuple[str, str]]:
    reclaimed = []
    for task_id in await core.tasks[CRAWL].expired(now):
        payload = await core.payload(task_id) or {}
        kind = payload.get("kind", CRAWL)
        expiry = await core.tasks[kind].claim_expiry(task_id)
        found = await core.tasks[kind].reclaim(task_id, now)
        if found is None:
            continue
        holder, seq = found
        # Claimed before this process started: the miner may have tried to finish while we were down.
        forgiven = expiry is not None and expiry <= core.started_at + core.tasks[kind].claim_ttl
        if holder:
            if not forgiven:
                await core.db(lapse, core, holder, task_id, payload, "claim_expired")
            await core.record(
                payload.get("round_id") or core.current.get(kind, ""),
                holder,
                0.0,
                "reclaimed",
                seq,
                task_id=task_id,
                cause="restart" if forgiven else "expired",
            )
        reclaimed.append((task_id, holder))
    return reclaimed


def lapse(core, hotkey: str, task_id: str, payload: dict, cause: str) -> int:
    """A claim that ended without an upload: its URLs, the budget and a strike; returns the budget."""
    kind = payload.get("kind", CRAWL)
    assigned = len(set(payload.get("urls", [])))
    with core.sqlite.batch():
        if kind == CRAWL:
            core.budgets.record_coverage(hotkey, assigned, 0)
            core.budgets.credit(hotkey, -assigned, kind)
        budget = core.budgets.penalise(hotkey, task_id, cause, kind).budget
        strike(core, hotkey, cause, task_id, kind)
    return budget


def strike(core, miner: str, reason: str, task_id: str, kind: str) -> None:
    since = time.time() - STRIKE_WINDOW_H * HOUR
    judged = core.validations.judged_since(miner, since, kind)
    core.budgets.strike(miner, reason, task_id, judged, kind)


def crashed_on_most(votes: list[dict]) -> bool:
    """Checks that crashed twice on this upload alone, on most validators, point at the file."""
    crashed = sum(1 for vote in votes if vote["result"].get("crashed"))
    return crashed * 2 > len(votes)


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
        if kind == CRAWL:
            await report_outcomes(core, job["urls"], outcomes.DROPPED, task_id)
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


async def void_task(
    core, task_id: str, lapsed: queues.Finalized, validator: str, reason: str
) -> dict:
    await discard_upload(core, lapsed.job)
    await requeue(core, task_id, lapsed.job, "void")
    report = build_report(
        task_id, lapsed.job, validator, {"verdict": "void", "reason": reason}
    )
    await write_report(core, report)
    await core.db(core.validations.record, report)
    await core.db(core.validations.record_votes, report, lapsed.votes, decided=False)
    return report


def lock_key(task_id: str) -> str:
    return f"scoring:{task_id}"


async def settle_seeded(core) -> int:
    """Uploads whose seed block exists: a drawn share opens for validators, the rest finalize on the miner's own counts."""
    try:
        block = await core.seeds.current_block()
    except Exception as exc:
        log.warning("chain unavailable: %r", exc)
        return 0
    ids = await core.validation.seeded(block)
    found = await asyncio.gather(*(core.validation.job(task_id) for task_id in ids))
    jobs = [(task_id, job) for task_id, job in zip(ids, found) if job is not None]
    # Uploads frozen together share a seed block, so each block's hash is read once.
    blocks = sorted({job["seed_block"] for _, job in jobs})
    hashes = await asyncio.gather(
        *(core.seeds.seed_for(block) for block in blocks), return_exceptions=True
    )
    seeds = {
        block: seed for block, seed in zip(blocks, hashes) if isinstance(seed, str)
    }
    slots = asyncio.Semaphore(SETTLE_AT_ONCE)

    async def settle(task_id: str, job: dict) -> int:
        seed = seeds.get(job["seed_block"])
        if seed is None:
            return 0
        async with slots:
            lock = lock_key(task_id)
            if not await core.redis.set(lock, "seeded", nx=True, ex=FINALIZE_LOCK_S):
                return 0
            try:
                await settle_one(core, task_id, job, seed)
            except Exception:
                log.exception(
                    "task=%s could not be settled; it is tried again", task_id
                )
                return 0
            finally:
                await core.redis.delete(lock)
        return 1

    settled = sum(await asyncio.gather(*(settle(t, j) for t, j in jobs)))
    if settled:
        await publish_open(core)
    return settled


async def settle_one(core, task_id: str, job: dict, seed: str) -> None:
    reason = await pick_reason(core, task_id, job, seed)
    if reason is None:
        await finalize_unchecked(core, task_id, job, seeding=True)
        return
    if reason == sampling.RECHECK:
        await core.db(core.checks.took_recheck, job["miner"])
    await core.validation.open(task_id, {**job, "picked": reason})


async def pick_reason(core, task_id: str, job: dict, seed: str) -> str | None:
    kind, miner = job.get("kind", CRAWL), job["miner"]
    if kind == EMBED:
        return sampling.NEW
    if not (job.get("reported") or {}).get("rows"):
        return UNREPORTED
    if await core.db(core.budgets.locked_until, miner, kind) is not None:
        return LOCKED
    return sampling.pick_reason(
        sampling.draw(seed, task_id, job.get("etag", "")),
        await sampling.uploads_last_hour(core.redis, miner),
        await core.db(core.checks.passes, miner),
        await core.db(core.checks.recheck_left, miner),
        sampling.budget_share(
            core.check_share,
            core.checks_per_hour,
            await sampling.uploads_last_hour(core.redis, sampling.ALL),
        ),
    )


def unchecked_vote(job: dict, error_share: float) -> dict:
    """No validator checked it: the miner's own counts, within what it was assigned, errors at its confirmed share."""
    assigned = len(set(job["urls"]))
    reported = job.get("reported") or {}
    content = min(int(reported.get("ok", 0)), assigned)
    errors = min(int(reported.get("errors", 0)), assigned - content)
    returned = content + errors
    if returned < COVERAGE_GATE * assigned:
        verdict, reason, credited = "fail", "coverage", 0
    else:
        verdict, reason = "pass", UNCHECKED
        credited = content + round(errors * error_share)
    return {
        "validator": "",
        "verdict": verdict,
        "credited": credited,
        "result": {
            "verdict": verdict,
            "reason": reason,
            "returned": returned,
            "missing": assigned - returned,
            "error_rows": errors,
            "credited": credited,
            "samples": [],
            "rejected": [],
        },
    }


async def finalize_unchecked(
    core, task_id: str, job: dict, seeding: bool = False
) -> dict | None:
    share = await core.db(core.checks.error_share, job["miner"])
    decision = Decision("final", unchecked_vote(job, share), [])
    try:
        return await conclude_validation(core, task_id, job, decision, seeding)
    except (StorageDown, ClaimLost):
        return None


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
    """Finalizes once every active validator voted, or at the deadline on the votes it has; with none, on the miner's own counts."""
    voters = await core.validation.voters(task_id)
    active = await core.validation.active(now) | voters
    due = now >= job.get("deadline", 0)
    if not due and not (voters and active <= voters):
        return None
    if not voters:
        if job.get("kind", CRAWL) != CRAWL:
            return None
        return await finalize_unchecked(core, task_id, job)
    votes = await core.validation.votes(task_id)
    decision = held_to_report(job, decide(votes, overdue=True))
    try:
        return await conclude_validation(core, task_id, job, decision)
    except (StorageDown, ClaimLost):
        return None


def overstated(job: dict, result: dict) -> bool:
    reported = job.get("reported") or {}
    if not reported.get("rows"):
        return False
    counted = result.get("returned", 0) - result.get("error_rows", 0)
    return credit.overstated(reported.get("ok", 0), counted, len(set(job["urls"])))


def held_to_report(job: dict, decision: Decision) -> Decision:
    """Unchecked uploads are paid on the miner's report, so a checked one that overstates it fails."""
    vote = decision.vote
    if (
        job.get("kind", CRAWL) != CRAWL
        or vote["verdict"] != "pass"
        or not overstated(job, vote["result"])
    ):
        return decision
    result = {
        **vote["result"],
        "verdict": "fail",
        "reason": REPORTED_ROWS,
        "credited": 0,
    }
    return dataclasses.replace(
        decision, vote={**vote, "verdict": "fail", "credited": 0, "result": result}
    )


async def conclude_validation(
    core, task_id: str, job: dict, decision: Decision, seeding: bool = False
) -> dict:
    """Accounts are finalized once per upload, in one transaction, before Redis lets go."""
    finalized = await core.db(core.validations.final_verdict, task_id, job["key"])
    if finalized is not None:
        return await finish_finalized(core, task_id, job, finalized, seeding)

    vote, result = decision.vote, decision.vote["result"]
    verdict = vote["verdict"]
    report = build_report(task_id, job, vote["validator"], result, decision.votes)
    try:
        await core.pages.put_json(report["report_key"], report)
    except Exception:
        log.exception("storage failed while scoring %s", task_id)
        raise StorageDown(task_id) from None

    publish = None
    if verdict == "pass" and (
        result.get("matched") or result.get("reason") == UNCHECKED
    ):
        publish = publish_job(core, task_id, job, vote, decision.agreed)
    try:
        budget = await core.db(
            finalize_accounts, core, task_id, job, decision, report, publish
        )
    except AlreadySettled:
        finalized = await core.db(core.validations.final_verdict, task_id, job["key"])
        return await finish_finalized(core, task_id, job, finalized, seeding)

    for validator in decision.disagreed:
        if await core.db(core.validations.is_excluded, validator):
            await core.validation.leave(validator)
    await close_upload(core, task_id, job, verdict, publish, seeding)
    if job.get("picked") and job.get("kind", CRAWL) == CRAWL:
        await account_check(core, job, decision)
    return {
        "task_id": task_id,
        "verdict": verdict,
        "credited": vote["credited"] if verdict == "pass" else 0,
        "miner_budget": budget,
    }


async def account_check(core, job: dict, decision: Decision) -> None:
    """A pass marks where the hotkey last stood; a fail takes back what passed since then and checks its next uploads."""
    verdict, miner = decision.vote["verdict"], job["miner"]
    if verdict not in ("pass", "fail"):
        return
    samples = decision.vote["result"].get("samples", [])
    judged = sum(1 for sample in samples if sample["outcome"] in ERROR_OUTCOMES)
    unconfirmed = sum(
        1 for sample in samples if sample["outcome"] == "errors_unconfirmed"
    )
    since = await core.db(core.checks.last_pass_at, miner)
    await core.db(
        core.checks.record,
        miner,
        job["task_id"],
        verdict == "pass",
        job.get("completed_at") or time.time(),
        judged,
        unconfirmed,
    )
    if verdict == "pass":
        return
    await take_back(core, miner, since, "check_failed")
    await core.db(core.checks.start_recheck, miner)
    if await core.db(core.checks.fails_in_recent, miner) >= credit.FAILS_FOR_PENALTY:
        await full_penalty(core, miner, "repeated_fails")


async def take_back(
    core, miner: str, since: float, reason: str, checked_too: bool = False
) -> list[str]:
    """Un-credits and unpublishes a hotkey's passed crawl uploads completed after `since`: the unchecked ones, or every one."""

    def withdraw() -> list[tuple[str, int]]:
        with core.sqlite.batch():
            taken = core.validations.withdraw(miner, since, checked_too)
            core.budgets.credit(miner, -sum(credited for _, credited in taken), CRAWL)
        return taken

    taken = await core.db(withdraw)
    task_ids = [task_id for task_id, _ in taken]
    if task_ids:
        await core.publish.withdraw(task_ids, reason)
        log.warning(
            "miner=%s %d uploads taken back: %s", miner[:10], len(task_ids), reason
        )
    return task_ids


async def full_penalty(core, miner: str, reason: str) -> None:
    """The hotkey's last day of credit and pages, and a lockout."""
    now = time.time()
    await core.db(
        core.budgets.lock_out, miner, CRAWL, FULL_PENALTY_LOCKOUT_H, reason, now
    )
    await take_back(
        core, miner, now - credit.PENALTY_WINDOW_S, reason, checked_too=True
    )
    await core.db(core.budgets.wipe_credits, miner, now - credit.PENALTY_WINDOW_S)
    log.warning("miner=%s full penalty: %s", miner[:10], reason)


async def report_outcomes(core, urls: list[str], outcome: str, task_id: str) -> None:
    try:
        await outcomes.write(
            core.storage, core.redis, outcomes.rows_for(urls, outcome, task_id)
        )
    except Exception:
        log.exception("could not write %d outcomes for %s", len(urls), task_id)


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
            if kind == CRAWL and result.get("reason") in STRIKE_REASONS:
                core.budgets.credit(miner, -assigned, kind)
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
        core.validations.record_votes(
            report,
            decision.votes,
            decision.disagreed,
            decided=result.get("reason") != NO_MAJORITY,
        )
        if verdict == "fail" and result.get("reason") in STRIKE_REASONS:
            strike(core, miner, result["reason"], task_id, kind)
        if crashed_on_most(decision.votes):
            core.budgets.lock_out(miner, kind, HOSTILE_LOCKOUT_H, "hostile_upload")
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
        "validators": sorted(set(agreed) | ({vote["validator"]} - {""})),
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
    core,
    task_id: str,
    job: dict,
    verdict: str,
    publish: dict | None,
    seeding: bool = False,
) -> None:
    """Closes a finalized upload in Redis and moves the task on."""
    if await core.validation.finalize(task_id, publish, seeding) is None:
        log.warning("task=%s was closed before its final_verdict was recorded", task_id)
        raise ClaimLost(task_id)
    if verdict == "pass":
        await finish_task(core, job["round_id"], task_id)
    else:
        await discard_upload(core, job)
        await requeue(core, task_id, job, verdict)


async def discard_upload(core, job: dict) -> None:
    """Only a passed upload is published, and a kept file could be resubmitted by the next miner."""
    await delete_quietly(core.storage, job["key"])
    await delete_quietly(core.storage, manifest_key(job["key"]))


async def finish_finalized(
    core, task_id: str, job: dict, finalized: dict, seeding: bool = False
) -> dict:
    await close_upload(
        core, task_id, job, finalized["verdict"], finalized["publish"], seeding
    )
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
