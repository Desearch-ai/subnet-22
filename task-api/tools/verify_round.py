#!/usr/bin/env python3
"""Checks that a round was distributed honestly; needs no credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ALGORITHM = "desearch-serve-order-2"
RECEIPT_FIELDS = (
    "round_id",
    "hotkey",
    "requested_at",
    "outcome",
    "seq",
    "task_id",
    "refusal",
    "cause",
)


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_hash(entries: list[dict], seed_block: int) -> str:
    ordered = sorted(entries, key=lambda entry: entry["batch_id"])
    return sha256(
        canonical_json(
            {"algorithm": ALGORITHM, "seed_block": seed_block, "batches": ordered}
        )
    )


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


def receipt_body(round_id: str, entry: dict) -> dict:
    fields = {"round_id": round_id, **entry}
    return {
        name: fields[name] for name in RECEIPT_FIELDS if fields.get(name) is not None
    }


def signed_by(signer: str, body: dict, signature: str) -> bool | None:
    try:
        from bittensor_wallet import Keypair
    except ImportError:
        return None
    try:
        return Keypair(ss58_address=signer).verify(
            canonical_json(body), bytes.fromhex(signature)
        )
    except Exception:
        return False


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


def verify(
    api: str,
    round_id: str,
    receipts: list[dict] | None = None,
    received: dict[str, list[str]] | None = None,
    signer: str | None = None,
) -> bool:
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

    recomputed = manifest_hash(manifest, round_["seed_block"])
    ok &= check(
        "manifest AND seed block match the hash committed before that block existed",
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
        "batches were handed out in exactly the published order",
        *_replay(manifest, expected, log["entries"]),
    )

    root = merkle_root([canonical_json(line) for line in log["entries"]])
    ok &= check(
        "anchored root matches the published log",
        root == log["anchor_root"],
        f"anchored {log['anchor_root'][:32]}…\n         recomputed {root[:32]}…",
    )

    published_signer = round_.get("signer", "")
    if signer:
        ok &= check(
            "the API signs with the key you expected",
            published_signer == signer,
            f"published {published_signer}, expected {signer}",
        )
    else:
        print(
            f"  {GREY}[ .. ] no --signer given: signatures prove only that one key signed the log{RESET}"
        )
    ok &= check(
        "every log entry carries the API's signature",
        *_signatures(round_id, log["entries"], signer or published_signer),
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

    if received:
        ok &= check(
            "the URLs you were given match the commitment",
            *_batches(manifest, received),
        )

    if receipts:
        ok &= check(
            "every receipt you hold is signed and appears in the published log",
            *_receipts(round_id, receipts, log["entries"], signer or published_signer),
        )
    else:
        print(
            f"  {GREY}[ .. ] no receipts supplied -- the above only proves internal consistency{RESET}"
        )

    print()
    print(
        f"{GREEN}Round verified.{RESET}" if ok else f"{RED}Verification FAILED.{RESET}"
    )
    return bool(ok)


def _signatures(round_id: str, entries: list[dict], signer: str) -> tuple[bool, str]:
    for line in entries:
        verdict = signed_by(
            signer, receipt_body(round_id, line), line.get("receipt_sig", "")
        )
        if verdict is None:
            return False, "install bittensor-wallet to check signatures"
        if not verdict:
            return (
                False,
                f"entry seq {line.get('seq')} ({line['outcome']}) is not signed by {signer}",
            )
    return True, f"{len(entries)} entries signed by {signer}"


def _receipts(
    round_id: str, receipts: list[dict], entries: list[dict], signer: str
) -> tuple[bool, str]:
    logged = {canonical_json(receipt_body(round_id, line)) for line in entries}
    problems = []
    for receipt in receipts:
        body = receipt["body"]
        if signed_by(signer, body, receipt["signature"]) is False:
            problems.append(f"bad signature on {body['outcome']} seq {body.get('seq')}")
        elif canonical_json(body) not in logged:
            problems.append(
                f"{body['outcome']} seq {body.get('seq')} is missing from the log"
            )
    if problems:
        return (
            False,
            f"{len(problems)} of your {len(receipts)} receipts fail: {problems[:3]}",
        )
    return True, f"all {len(receipts)} receipts signed and present in the log"


def _replay(
    manifest: list[dict], order: list[str], entries: list[dict]
) -> tuple[bool, str]:
    entries = sorted(entries, key=lambda e: e.get("seq", 0))
    rank = {batch_id: i for i, batch_id in enumerate(order)}
    queued = list(order)
    holders: dict[str, set[str]] = defaultdict(set)
    issues = reclaims = 0

    for line in entries:
        task_id = line.get("task_id")
        if line["outcome"] == "issued":
            hotkey = line.get("hotkey", "")
            due = next((b for b in queued if hotkey not in holders[b]), None)
            if due is None:
                return (
                    False,
                    f"issued {task_id} when no queued batch was new to {hotkey}",
                )
            if due != task_id:
                return False, (
                    f"issued {task_id} (position {rank.get(task_id, '?')}) while "
                    f"{due} (position {rank[due]}) was next in line for {hotkey}"
                )
            queued.remove(task_id)
            holders[task_id].add(hotkey)
            issues += 1
        elif line["outcome"] == "reclaimed" and task_id:
            place = next(
                (i for i, b in enumerate(queued) if rank[b] > rank[task_id]),
                len(queued),
            )
            queued.insert(place, task_id)
            reclaims += 1
    return (
        True,
        f"{issues} issues in exact serve order, {reclaims} returned to the queue",
    )


def _batches(manifest: list[dict], received: dict[str, list[str]]) -> tuple[bool, str]:
    committed = {e["batch_id"]: e["urls_hash"] for e in manifest}
    for task_id, urls in received.items():
        if task_id not in committed:
            return False, f"{task_id} was never in the manifest"
        if sha256(canonical_json(urls)) != committed[task_id]:
            return (
                False,
                f"the URLs you were given for {task_id} are not the ones committed",
            )
    return True, f"{len(received)} batches match the URLs committed at round open"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--round", required=True)
    parser.add_argument(
        "--signer", help="the ss58 key the API publishes for its receipts"
    )
    parser.add_argument(
        "--batches",
        help="file of {task_id: [urls]} you were given, to check they match the commitment",
    )
    parser.add_argument(
        "--receipts",
        help="file of receipts you kept, one JSON object per line as the miner writes them",
    )
    args = parser.parse_args()
    receipts = (
        [
            json.loads(line)
            for line in Path(args.receipts).read_text().splitlines()
            if line.strip()
        ]
        if args.receipts
        else None
    )
    batches = json.loads(Path(args.batches).read_text()) if args.batches else None
    return 0 if verify(args.api, args.round, receipts, batches, args.signer) else 1


if __name__ == "__main__":
    sys.exit(main())
