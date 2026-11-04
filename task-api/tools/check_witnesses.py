#!/usr/bin/env python3
"""Turns relayed requests the task API's log lacks into censorship proofs; needs no credentials.

Witness records come from a miner's own WITNESSES_FILE (one JSON object per line, as the miner keeps the ones
relays sign for it) or from a validator's relay (`--relay`, its public `/v1/witness/requests`). Each one the log
owes an entry, forwarded at least `--after-blocks` blocks before the chain's head, is looked up by its request
hash in the log the API signs with `--signer`; one it lacks is printed as a proof, one JSON object per line.

    python task-api/tools/check_witnesses.py --api https://api-22.desearch.ai --signer 5F... \\
        --witnesses witnesses.jsonl --network finney

Exits 1 when it found a proof, 0 otherwise. `--proofs FILE` checks proofs someone else made instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp  # noqa: E402

from desearch import witness  # noqa: E402
from desearch.witness import CHECK_AFTER_BLOCKS, ApiLog  # noqa: E402


def read_lines(path: str) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


async def relay_records(http: aiohttp.ClientSession, relay: str) -> list[dict]:
    found, after = [], 0
    while True:
        async with http.get(
            f"{relay.rstrip('/')}/v1/witness/requests",
            params={"after": after, "limit": 500},
        ) as r:
            r.raise_for_status()
            page = (await r.json())["records"]
        if not page:
            return found
        found += page
        after = page[-1]["id"]


async def head_block(network: str) -> int:
    import bittensor as bt

    subtensor = await bt.Subtensor(network)
    try:
        return int(await subtensor.block())
    finally:
        await subtensor.close()


async def check(args) -> int:
    async with aiohttp.ClientSession() as http:
        if args.proofs:
            proofs = read_lines(args.proofs)
            log = (
                ApiLog(http, args.api, args.signer)
                if args.api and args.signer
                else None
            )
            bad = 0
            for proof in proofs:
                genuine = witness.check_proof(proof, args.after_blocks)
                missing = await log.included(proof["request"]) is False if log else None
                ok = genuine and missing is not False
                bad += not ok
                print(
                    json.dumps(
                        {
                            "request": proof.get("request"),
                            "genuine": genuine,
                            "still_missing": missing,
                        }
                    )
                )
            return 1 if bad else 0

        records = read_lines(args.witnesses) if args.witnesses else []
        for relay in args.relay:
            records += await relay_records(http, relay)
        head = (
            args.head_block
            if args.head_block is not None
            else await head_block(args.network)
        )
        log = ApiLog(http, args.api, args.signer)
        found = 0
        for record in records:
            proof = await witness.censorship_proof(
                record, head, log.included, args.after_blocks, args.api
            )
            if proof:
                found += 1
                print(json.dumps(proof, sort_keys=True))
        print(
            f"{len(records)} witness records, {found} censorship proofs at block {head}",
            file=sys.stderr,
        )
        return 1 if found else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api",
        default="https://api-22.desearch.ai",
        help="the task API whose log is checked",
    )
    parser.add_argument(
        "--signer", required=True, help="the ss58 key the log must be signed with"
    )
    parser.add_argument(
        "--witnesses", help="a file of witness records, one JSON object per line"
    )
    parser.add_argument(
        "--relay",
        action="append",
        default=[],
        help="a validator relay to read records from",
    )
    parser.add_argument(
        "--network",
        default="finney",
        help="the chain whose head decides which records are due",
    )
    parser.add_argument(
        "--head-block",
        type=int,
        help="the head block to judge by, instead of asking the chain",
    )
    parser.add_argument("--after-blocks", type=int, default=CHECK_AFTER_BLOCKS)
    parser.add_argument(
        "--proofs", help="check a file of proofs instead of making them"
    )
    args = parser.parse_args()
    if not (args.witnesses or args.relay or args.proofs):
        parser.error("give --witnesses, --relay or --proofs")
    return asyncio.run(check(args))


if __name__ == "__main__":
    sys.exit(main())
