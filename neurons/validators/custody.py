"""Which key validators follow: the task API's custody records, each checked against the chain and its attestation (docs/tee.md)."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Awaitable, Callable

import aiohttp
from bittensor.wallets import Keypair

from desearch import attestation
from desearch.attestation import canonical, sha256_hex
from desearch.trust import TASK_API_SIGNER
from neurons.validators import governance

CUSTODY_LATEST = "log/custody/latest.json"
# The task API's signer from before custody records, pinned by the release (block B): it alone may hand
# over at seq 0, and it signs the upload log up to that record's upload_log_seq.
LEGACY_SIGNER = TASK_API_SIGNER
READ_TIMEOUT = aiohttp.ClientTimeout(total=30.0)
# The running version's endorsement is read from the chain at most this often.
ENDORSE_EVERY_S = 600
APPOINT = "appoint"
HANDOVER = "handover"

log = logging.getLogger("validator")

Snapshots = Callable[[int | None], Awaitable[governance.Snapshot]]
Attest = Callable[..., Awaitable[attestation.Report]]


def custody_key(seq: int) -> str:
    return f"log/custody/seq/{seq:012d}.json"


def attestation_key(signer: str) -> str:
    return f"log/custody/attestations/{signer}.json"


def record_payload(record: dict) -> bytes:
    return canonical({k: v for k, v in record.items() if k != "signature"})


def record_hash(record: dict) -> str:
    return sha256_hex(canonical(record))


def signed_by(record: dict) -> str | None:
    """Who must have signed a record: an appointed CVM itself, or the signer handing over."""
    return (
        record.get("from") if record.get("kind") == HANDOVER else record.get("signer")
    )


def signature_valid(record: dict) -> bool:
    try:
        return Keypair(ss58_address=signed_by(record)).verify(
            record_payload(record), bytes.fromhex(record["signature"])
        )
    except Exception:
        return False


class FixedSigner:
    """One key for every log file and manifest, as before custody records."""

    def __init__(self, signer: str):
        self.signer = signer

    def for_log(self, seq: int) -> list[str]:
        return [self.signer] if self.signer else []

    def for_manifest(self, manifest: dict) -> list[str]:
        return [self.signer] if self.signer else []

    def for_journal(self, seq: int) -> list[str]:
        return [self.signer] if self.signer else []

    def every(self) -> list[str]:
        return [self.signer] if self.signer else []

    def __str__(self) -> str:
        return self.signer


def as_signers(signer):
    """A signer address or anything that answers for_log and for_manifest."""
    return FixedSigner(signer) if isinstance(signer, str) else signer


class Custody:
    """The custody chain this validator has accepted, read on from the uploads bucket."""

    def __init__(
        self,
        http: aiohttp.ClientSession,
        storage_url: str,
        path: str,
        snapshots: Snapshots,
        fallback: str = "",
        attest: Attest = attestation.verify,
        policy: attestation.Policy = attestation.RELEASE,
    ):
        self.http = http
        self.storage_url = storage_url.rstrip("/")
        self.path = path
        self.snapshots = snapshots
        self.fallback = fallback
        self.attest = attest
        self.policy = policy
        self.records: list[dict] = []
        self.fork: list[dict] | None = None
        self.endorsed = True
        self.endorsed_at = float("-inf")
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as file:
            kept = json.load(file)
        self.records = kept.get("records", [])
        self.fork = kept.get("fork")

    def save(self) -> None:
        temporary = f"{self.path}.tmp"
        with open(temporary, "w") as file:
            json.dump({"records": self.records, "fork": self.fork}, file)
        os.replace(temporary, self.path)

    @property
    def current(self) -> str | None:
        """The signer followed now; None after a fork or while its version lost its endorsement."""
        if self.fork or not self.records or not self.endorsed:
            return None
        return self.records[-1]["signer"]

    def __str__(self) -> str:
        if self.fork:
            return "nobody (custody forked)"
        if not self.records:
            return self.fallback or "nobody"
        return (
            self.current
            or f"nobody ({self.records[-1]['signer']} is no longer endorsed)"
        )

    def holders(self) -> list[tuple[str, dict | None]]:
        """Each signer before and in the chain, with the record that ended its turn."""
        found: list[tuple[str, dict | None]] = []
        if self.fallback:
            found.append((self.fallback, self.records[0] if self.records else None))
        for at, record in enumerate(self.records):
            after = self.records[at + 1] if at + 1 < len(self.records) else None
            found.append((record["signer"], after))
        return found

    def for_log(self, seq: int) -> list[str]:
        """Who may have signed upload log file `seq`: a turn ends at the next record's `upload_log_seq`."""
        if self.fork:
            return []
        found, start = [], 0
        for signer, ended_by in self.holders():
            if ended_by is None:
                if seq >= start and (signer == self.current or not self.records):
                    found.append(signer)
                continue
            end = ended_by.get("upload_log_seq")
            if end is not None:
                if start <= seq <= int(end):
                    found.append(signer)
                start = max(start, int(end) + 1)
        return found

    def for_journal(self, seq: int) -> list[str]:
        """Who may have signed log entries up to `seq`: a turn ends at the next record's `journal_seq`."""
        if self.fork:
            return []
        found, start = [], 0
        for signer, ended_by in self.holders():
            if ended_by is None:
                if seq >= start and (signer == self.current or not self.records):
                    found.append(signer)
                continue
            end = ended_by.get("journal_seq")
            if end is not None:
                if start <= seq <= int(end):
                    found.append(signer)
                start = max(start, int(end) + 1)
        return found

    def every(self) -> list[str]:
        """Every key that held the log, oldest first."""
        return [signer for signer, _ in self.holders()]

    def for_manifest(self, manifest: dict) -> list[str]:
        """Who may have signed a manifest: the current signer, or one whose turn ended after it was frozen."""
        if self.fork:
            return []
        frozen = int(manifest.get("frozen_block") or 0)
        found = []
        for signer, ended_by in self.holders():
            if ended_by is None:
                if signer == self.current or not self.records:
                    found.append(signer)
            elif frozen <= int(ended_by.get("block", -1)):
                found.append(signer)
        return found

    async def poll(self) -> bool:
        """Reads records written since the last poll; True when the signer changed."""
        before = self.current if self.records else self.fallback
        latest = await self.get(CUSTODY_LATEST)
        if latest is not None and not self.fork:
            await self.check_last()
            for seq in range(len(self.records), int(latest["seq"]) + 1):
                if self.fork:
                    break
                record = await self.get(custody_key(seq))
                if record is None or not await self.accept(record):
                    break
        if (
            self.records
            and not self.fork
            and time.time() - self.endorsed_at >= ENDORSE_EVERY_S
        ):
            await self.check_endorsed()
        after = self.current if self.records else self.fallback
        return before != after

    async def check_last(self) -> None:
        """A different record at a seq already accepted, valid on its own, is a fork."""
        if not self.records:
            return
        kept = self.records[-1]
        found = await self.get(custody_key(kept["seq"]))
        if found is None or record_hash(found) == record_hash(kept):
            return
        problem = await self.problem(found, self.records[:-1])
        if problem:
            log.warning(
                "custody record %d was replaced by one that does not hold: %s",
                kept["seq"],
                problem,
            )
            return
        self.fork = [kept, found]
        self.save()
        log.error(
            "custody forked at seq %d: two valid records, %s and %s; not following any signer",
            kept["seq"],
            record_hash(kept),
            record_hash(found),
        )

    async def accept(self, record: dict) -> bool:
        problem = await self.problem(record, self.records)
        if problem:
            log.error("custody record %s not accepted: %s", record.get("seq"), problem)
            return False
        self.records.append(record)
        self.save()
        log.info(
            "custody record %d accepted: %s now signs for version %s",
            record["seq"],
            record["signer"],
            record["compose_hash"],
        )
        return True

    async def problem(self, record: dict, chain: list[dict]) -> str | None:
        """Why a record does not continue `chain`, None when it does."""
        seq = len(chain)
        previous = chain[-1] if chain else None
        if record.get("seq") != seq:
            return f"seq {record.get('seq')} where {seq} comes next"
        if record.get("prev") != (record_hash(previous) if previous else None):
            return "prev does not name the last record"
        kind = record.get("kind")
        if kind == HANDOVER:
            # The first record may hand over from the signer before custody records, the release's own.
            holder = previous["signer"] if previous else self.fallback
            if not holder:
                return "a handover with nothing before it"
            if record.get("from") != holder:
                return f"handed over by {record.get('from')}, not by {holder}"
        elif kind != APPOINT:
            return f"kind {kind!r}"
        if not signature_valid(record):
            return f"not signed by {signed_by(record)}"
        problem = await self.approval_problem(record)
        if problem:
            return problem
        return await self.attestation_problem(record)

    async def approval_problem(self, record: dict) -> str | None:
        try:
            snapshot = await self.snapshots(int(record["block"]))
        except Exception as error:
            return f"the chain at block {record.get('block')} is unavailable: {error!r}"
        version = record["compose_hash"]
        if record["kind"] == APPOINT:
            if not governance.appointed(snapshot, version, record["signer"]):
                return f"{record['signer']} was not appointed for {version} at block {snapshot.block}"
        elif not governance.approved(snapshot, version):
            return f"version {version} was not approved at block {snapshot.block}"
        return None

    async def attestation_problem(self, record: dict) -> str | None:
        found = await self.get(attestation_key(record["signer"]))
        if found is None:
            return f"no attestation published for {record['signer']}"
        report = await self.attest(
            found,
            signer=record["signer"],
            compose_hash=record["compose_hash"],
            document_hash=record.get("attestation_sha256"),
            policy=self.policy,
        )
        if not report.ok:
            return "attestation: " + "; ".join(
                f"{c.name}: {c.detail}" for c in report.failures()
            )
        return None

    async def check_endorsed(self) -> None:
        try:
            snapshot = await self.snapshots(None)
        except Exception as error:
            log.warning("chain unavailable, keeping the last endorsement: %r", error)
            return
        self.endorsed_at = time.time()
        endorsed = governance.endorsed(snapshot, self.records[-1]["compose_hash"])
        if endorsed != self.endorsed:
            log.log(
                logging.INFO if endorsed else logging.ERROR,
                "version %s is %s endorsed by validators' stake",
                self.records[-1]["compose_hash"],
                "again" if endorsed else "no longer",
            )
        self.endorsed = endorsed

    async def get(self, key: str) -> dict | None:
        async with self.http.get(
            f"{self.storage_url}/{key}", timeout=READ_TIMEOUT
        ) as response:
            if response.status in (403, 404):
                return None
            response.raise_for_status()
            return await response.json(content_type=None)
