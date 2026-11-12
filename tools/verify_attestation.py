#!/usr/bin/env python3
"""Checks that the task API runs an approved version in a TEE and signs with the key it shows; needs no credentials."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import socket
import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from desearch import attestation  # noqa: E402
from neurons.validators import custody as custody_module  # noqa: E402
from neurons.validators import governance  # noqa: E402

TIMEOUT_S = 30


def fetch_json(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "sn22-verify-attestation"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def served_spki(url: str) -> str | None:
    """sha256 of the public key in the certificate the API serves, None over plain HTTP."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        return None
    context = ssl.create_default_context()
    raw = socket.create_connection((parts.hostname, parts.port or 443), TIMEOUT_S)
    with context.wrap_socket(raw, server_hostname=parts.hostname) as connection:
        return attestation.spki_sha256(connection.getpeercert(binary_form=True))


def show(report: attestation.Report) -> None:
    for check in report.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")


def policy_of(args) -> attestation.Policy:
    return attestation.Policy(allow_any_os=args.allow_any_os, simulated=args.simulated)


async def check_api(args) -> bool:
    base = args.api.rstrip("/")
    nonce = args.nonce or (secrets.token_hex(32) if args.fresh else None)
    url = f"{base}/v1/attestation" + (f"?nonce={nonce}" if nonce else "")
    found = fetch_json(url)
    report = await attestation.verify(
        found,
        signer=args.signer,
        compose_hash=args.compose_hash,
        nonce=nonce,
        policy=policy_of(args),
    )
    document = report.document or {}
    try:
        config = fetch_json(f"{base}/v1/config")
        matches = attestation.config_sha256(config) == document.get("config_sha256")
        report.add(
            "config",
            matches,
            (
                "/v1/config is what the quote binds"
                if matches
                else "/v1/config differs from the bound config_sha256"
            ),
        )
    except Exception as error:
        report.add("config", False, f"/v1/config unavailable: {error!r}")
    try:
        spki = served_spki(base)
    except Exception as error:
        report.add("tls", False, f"the served certificate could not be read: {error!r}")
    else:
        if spki is None:
            report.add("tls", True, "not checked: plain HTTP")
        else:
            bound = document.get("tls_spki_sha256")
            report.add("tls", spki == bound, f"served key {spki}, bound {bound}")
    print(f"Attestation at {url}")
    show(report)
    print(
        f"  compose hash {report.compose_hash}\n  signer       {report.signer}\n  app id       {report.app_id}"
    )
    return report.ok


async def check_file(args) -> bool:
    found = json.loads(Path(args.file).read_text())
    report = await attestation.verify(
        found,
        signer=args.signer,
        compose_hash=args.compose_hash,
        policy=policy_of(args),
    )
    print(f"Attestation in {args.file}")
    show(report)
    print(f"  compose hash {report.compose_hash}\n  signer       {report.signer}")
    return report.ok


async def check_custody(args) -> bool:
    """The whole custody chain from the bucket, as a validator checks it."""
    print(f"Custody chain at {args.storage_url}")
    logging.basicConfig(
        level=logging.INFO, format="  %(levelname)s %(message)s", stream=sys.stdout
    )
    async with aiohttp.ClientSession() as http:
        if args.network:
            import bittensor as bt

            subtensor = await bt.Subtensor(args.network)
            snapshots = lambda block: governance.read_snapshot(
                subtensor, args.netuid, block
            )  # noqa: E731
        else:
            subtensor = None
            snapshots = None
        with tempfile.TemporaryDirectory() as folder:
            followed = Unchecked(
                http,
                args.storage_url,
                f"{folder}/custody.json",
                snapshots,
                fallback=args.legacy_signer,
                policy=policy_of(args),
            )
            try:
                await followed.poll()
            finally:
                if subtensor is not None:
                    await subtensor.close()
            latest = await followed.get(custody_module.CUSTODY_LATEST)
    expected = int(latest["seq"]) + 1 if latest else 0
    for record in followed.records:
        print(
            f"  ok   {record['seq']}: {record['kind']} -> {record['signer']} (version {record['compose_hash']})"
        )
    if not args.network:
        print(
            "  approvals were not checked: pass --network to read validators' commitments"
        )
    if followed.fork:
        print(
            "  FAIL the chain forked: "
            + ", ".join(custody_module.record_hash(r) for r in followed.fork)
        )
    ok = (
        not followed.fork
        and len(followed.records) == expected
        and followed.current is not None
    )
    print(
        f"  {'ok  ' if ok else 'FAIL'} {len(followed.records)} of {expected} records hold; following {followed}"
    )
    return ok


class Unchecked(custody_module.Custody):
    """The validator's custody checks, without the chain when no network is given."""

    async def approval_problem(self, record: dict) -> str | None:
        if self.snapshots is None:
            return None
        return await super().approval_problem(record)

    async def check_endorsed(self) -> None:
        if self.snapshots is not None:
            await super().check_endorsed()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", help="the task API, e.g. https://api-22.desearch.ai")
    parser.add_argument("--file", help="an attestation saved as JSON")
    parser.add_argument(
        "--storage-url", help="the uploads bucket, to check the custody chain"
    )
    parser.add_argument(
        "--network",
        help="a chain endpoint, to check approvals (an archive node for old records)",
    )
    parser.add_argument("--netuid", type=int, default=22)
    parser.add_argument(
        "--legacy-signer",
        default=custody_module.LEGACY_SIGNER,
        help="the signer before custody records, the only one that may hand over first",
    )
    parser.add_argument("--signer", help="the ss58 the attestation must bind")
    parser.add_argument("--compose-hash", help="the compose hash it must run")
    parser.add_argument("--nonce", help="hex the API must put in a fresh quote")
    parser.add_argument(
        "--fresh", action="store_true", help="ask for a fresh quote over a random nonce"
    )
    parser.add_argument(
        "--allow-any-os", action="store_true", help="testnets: accept any OS image"
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="the dstack simulator: skip the quote signature",
    )
    args = parser.parse_args()
    if not (args.api or args.file or args.storage_url):
        parser.error("give --api, --file or --storage-url")
    results = []
    if args.file:
        results.append(asyncio.run(check_file(args)))
    if args.api:
        results.append(asyncio.run(check_api(args)))
    if args.storage_url:
        results.append(asyncio.run(check_custody(args)))
    ok = all(results)
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
