#!/usr/bin/env python3
"""Writes a validator's approval of task API versions as its commitment on chain; prints it unless --commit."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurons.validators import governance  # noqa: E402


def approval_of(args) -> governance.Approval:
    versions = (args.version, args.next) if args.next else (args.version,)
    approval = governance.Approval(versions, args.appoint)
    if governance.parse(governance.text_of(approval)) != approval:
        raise SystemExit(
            "not a valid approval: compose hashes are 64 lowercase hex characters,"
            " --appoint takes an ss58 address and cannot go with --next"
        )
    return approval


def commitment_info(text: str) -> dict:
    return {
        "fields": [
            {f"Raw{len(field)}": "0x" + field.hex()}
            for field in governance.raw_fields(text)
        ]
    }


async def commit(args, info: dict) -> None:
    import bittensor as bt
    from bittensor.wallets import Wallet

    wallet = Wallet(
        name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path
    )
    async with bt.Client(args.network) as client:
        call = bt.calls.Commitments.set_commitment(args.netuid, info)
        result = await client.submit_call(call, wallet, signer="hotkey")
        print(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="the compose hash approved")
    parser.add_argument(
        "--next", help="a second compose hash, the successor of --version"
    )
    parser.add_argument(
        "--appoint",
        help="the ss58 of the one CVM of --version to appoint (genesis, recovery)",
    )
    parser.add_argument("--netuid", type=int, default=22)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="sign and send the commitment with the hotkey",
    )
    parser.add_argument(
        "--network", help="the chain to commit on; required with --commit"
    )
    parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument(
        "--wallet.path", dest="wallet_path", default="~/.bittensor/wallets"
    )
    args = parser.parse_args()
    text = governance.text_of(approval_of(args))
    info = commitment_info(text)
    print(f"commitment on netuid {args.netuid}: {text}")
    print(f"{len(text.encode())} bytes in {len(info['fields'])} Raw field(s): {info}")
    print("It replaces this hotkey's current commitment on the subnet.")
    if not args.commit:
        print("Dry run: pass --commit --network <chain> to send it.")
        return 0
    if not args.network:
        parser.error("--commit needs --network")
    asyncio.run(commit(args, info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
