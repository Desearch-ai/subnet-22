#!/usr/bin/env python3
"""Check that a round was distributed honestly.

    python verify_round.py --api https://tasks.desearch.ai --round <round_id>

Needs no credentials and imports nothing from the server. The rule is reimplemented below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ALGORITHM = "desearch-serve-order-1"


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_hash(entries: list[dict]) -> str:
    ordered = sorted(entries, key=lambda entry: entry["batch_id"])
    return sha256(canonical({"algorithm": ALGORITHM, "batches": ordered}))


def serve_order(seed: str, batch_ids: list[str]) -> list[str]:
    return sorted(batch_ids, key=lambda b: (sha256(f"{seed}:{b}".encode()), b))


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return sha256(b"")
    level = [hashlib.sha256(leaf).digest() for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


GREEN, RED, GREY, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[0m"


def get(api: str, path: str) -> dict:
    with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=30) as response:
        return json.load(response)


def check(label: str, passed: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}")
    if detail:
        print(f"         {GREY}{detail}{RESET}")
    return passed


def verify(api: str, round_id: str, receipts: list[dict] | None = None) -> bool:
    round_ = get(api, f"/v1/rounds/{round_id}")
    log = get(api, f"/v1/rounds/{round_id}/log")

    if round_.get("closed_at") is None:
        print(
            f"{RED}Round is still open; contents are published when it closes.{RESET}"
        )
        return False

    manifest = round_["manifest"]
    seed = round_["seed"]
    print(f"\nRound {round_id}")
    print(
        f"  {GREY}{len(manifest)} batches, seed from block {round_['seed_block']}{RESET}\n"
    )

    ok = True

    ok &= check(
        "algorithm is the one this script implements",
        round_.get("algorithm") == ALGORITHM,
        f"published {round_.get('algorithm')!r}, expected {ALGORITHM!r}",
    )

    recomputed = manifest_hash(manifest)
    ok &= check(
        "manifest matches the hash committed before the seed existed",
        recomputed == round_["manifest_hash"],
        f"committed {round_['manifest_hash'][:32]}…\n         recomputed {recomputed[:32]}…",
    )

    expected = serve_order(seed, [entry["batch_id"] for entry in manifest])
    ok &= check(
        "published serve order is what the rule produces from this seed",
        expected == round_["serve_order"],
        f"first three should be {expected[:3]}",
    )

    ok &= check(
        "every issue was the earliest batch whose hosts were all free",
        *_replay(manifest, expected, log["entries"]),
    )

    root = merkle_root([canonical(line) for line in log["entries"]])
    ok &= check(
        "anchored root matches the published log",
        root == log["anchor_root"],
        f"anchored {log['anchor_root'][:32]}…\n         recomputed {root[:32]}…",
    )

    refusals = [line for line in log["entries"] if line["outcome"] == "refused"]
    ok &= check(
        "every refusal states a reason",
        all(line.get("refusal", {}).get("code") for line in refusals),
        f"{len(refusals)} refusals, codes: "
        + ", ".join(
            sorted({line.get("refusal", {}).get("code", "?") for line in refusals})
            or ["none"]
        ),
    )

    if receipts:
        ok &= check("every receipt you hold appears in the published log", *_receipts(receipts, log["entries"]))
    else:
        print(f"  {GREY}[ .. ] no receipts supplied -- the above only proves internal consistency{RESET}")

    print()
    print(
        f"{GREEN}Round verified.{RESET}" if ok else f"{RED}Verification FAILED.{RESET}"
    )
    return bool(ok)


def _receipts(receipts: list[dict], entries: list[dict]) -> tuple[bool, str]:
    issued = {(e["hotkey"], e.get("task_id")) for e in entries if e["outcome"] == "issued"}
    refused = {}
    for e in entries:
        if e["outcome"] == "refused":
            refused.setdefault(e["hotkey"], []).append(e.get("refusal", {}).get("code"))

    missing = []
    for receipt in receipts:
        body = receipt["body"]
        if body["outcome"] == "issued":
            if (body["hotkey"], body["task_id"]) not in issued:
                missing.append(f"issue of {body['task_id'][:12]}")
        elif body["outcome"] == "refused":
            if body["refusal"]["code"] not in refused.get(body["hotkey"], []):
                missing.append(f"refusal {body['refusal']['code']}")
    if missing:
        return False, f"{len(missing)} of your {len(receipts)} receipts are absent: {missing[:3]}"
    return True, f"all {len(receipts)} receipts present in the log"


def _replay(manifest: list[dict], order: list[str], entries: list[dict]) -> tuple[bool, str]:
    """Rebuild the queue and its host locks from the log alone.

    A batch may only be passed over when one of its hosts was already held by a batch still in
    flight. That is the single deviation from serve order the server is allowed, and replaying the
    log makes it checkable instead of merely claimed.

    Completion and reclamation have to be distinguishable in the log: a completed batch is gone,
    a reclaimed one returns to the queue at the position it started from.
    """
    entries = sorted(entries, key=lambda e: e.get("seq", 0))
    hosts = {entry["batch_id"]: set(entry["hosts"]) for entry in manifest}
    rank = {batch_id: i for i, batch_id in enumerate(order)}
    queued = set(order)
    locked: dict[str, str] = {}
    issues = reclaims = 0

    def by_rank(ids):
        return sorted(ids, key=lambda b: rank[b])

    for line in entries:
        task_id = line.get("task_id")
        outcome = line["outcome"]
        if outcome == "issued":
            if task_id not in queued:
                return False, f"issued {task_id}, which was not in the queue"
            for earlier in by_rank(b for b in queued if rank[b] < rank[task_id]):
                if not (hosts[earlier] & set(locked)):
                    return False, (
                        f"{task_id} (position {rank[task_id]}) issued while {earlier} "
                        f"(position {rank[earlier]}) waited with every host free"
                    )
            queued.discard(task_id)
            for host in hosts[task_id]:
                locked[host] = task_id
            issues += 1
        elif outcome in ("completed", "reclaimed") and task_id:
            for host in [h for h, owner in locked.items() if owner == task_id]:
                del locked[host]
            if outcome == "reclaimed":
                queued.add(task_id)
                reclaims += 1
    return True, (
        f"{issues} issues replayed against the published order, "
        f"{reclaims} reclaimed after expiry, every skip forced by a held host"
    )


def _diff(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--round", required=True)
    parser.add_argument(
        "--receipts",
        help="file of receipts you kept. Without these the checks above only prove the "
        "published round agrees with itself.",
    )
    args = parser.parse_args()
    receipts = json.loads(Path(args.receipts).read_text()) if args.receipts else None
    return 0 if verify(args.api, args.round, receipts) else 1


if __name__ == "__main__":
    sys.exit(main())
