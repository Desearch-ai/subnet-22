#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local.miners import Client, Fabricator, Hoarder, Honest, Omitter, PollSpammer  # noqa: E402

API = "http://127.0.0.1:8099"
HOSTS = 60
URLS_PER_HOST = 40
WORK_SECONDS = 8.0


def sample_urls() -> list[dict]:
    urls = []
    for i in range(HOSTS):
        host = f"site{i:02d}.example"
        delay = 1.0 if i % 5 else 5.0
        count = URLS_PER_HOST * (4 if i < 5 else 1)
        for j in range(count):
            urls.append(
                {"host": host, "url": f"https://{host}/page/{j}", "crawl_delay": delay}
            )
    random.shuffle(urls)
    return urls


def start_server() -> subprocess.Popen:
    env = dict(
        os.environ,
        TASK_API_REDIS="redis://localhost:6379/15",
        TASK_API_REGISTRY="local",
        TASK_API_SEEDS="local",
        TASK_API_SEED_DELAY="1",
        TASK_API_LEASE_TTL="6",
        TASK_API_POLL_RATE="10",
        TASK_API_BUDGETS=str(ROOT / "local" / "scratch-budgets.db"),
        TASK_API_LOG=str(ROOT / "local" / "scratch-log.db"),
    )
    for leftover in ("scratch-budgets.db", "scratch-log.db"):
        (ROOT / "local" / leftover).unlink(missing_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            "8099",
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )


async def wait_ready(timeout: float = 30) -> None:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=2) as http:
        while time.time() < deadline:
            try:
                if (await http.get(f"{API}/v1/health")).status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError("API did not come up")


async def main() -> int:
    import redis.asyncio as aioredis

    scratch = aioredis.from_url("redis://localhost:6379/15", decode_responses=True)
    await scratch.flushdb()
    await scratch.aclose()

    server = start_server()
    try:
        await wait_ready()
        async with httpx.AsyncClient(timeout=30) as http:
            urls = sample_urls()
            opened = (
                await http.post(
                    f"{API}/v1/admin/rounds/open",
                    json={"urls": urls, "batch_target": 120},
                )
            ).json()
            round_id = opened["round_id"]
            print(f"\nRound {round_id}")
            print(f"  committed manifest hash  {opened['manifest_hash'][:32]}…")
            print(f"  seed block               {opened['seed_block']}  (not yet known)")
            print(
                f"  batches                  {opened['batches']} from {len(urls)} URLs\n"
            )

            filled = 0
            for _ in range(30):
                filled = (
                    await http.post(f"{API}/v1/admin/rounds/{round_id}/fill")
                ).json()["filled"]
                if filled:
                    break
                await asyncio.sleep(0.5)
            revealed = (await http.get(f"{API}/v1/rounds/{round_id}")).json()
            print(f"  seed revealed            {revealed['seed'][:32]}…")
            print(f"  queue filled             {filled} batches in serve order\n")

            miners = [
                Honest(Client(API, "//honest-1"), WORK_SECONDS),
                Honest(Client(API, "//honest-2"), WORK_SECONDS),
                Hoarder(Client(API, "//hoarder"), WORK_SECONDS),
                Omitter(Client(API, "//omitter"), WORK_SECONDS),
                Fabricator(Client(API, "//fabricator"), WORK_SECONDS),
                PollSpammer(Client(API, "//spammer"), WORK_SECONDS),
            ]
            print(f"  {len(miners)} miners working for {WORK_SECONDS:.0f}s…\n")
            await asyncio.gather(*(m.run() for m in miners))

            for miner in miners:
                print("   ", miner.report())

            print()
            for miner in miners:
                view = (await http.get(f"{API}/v1/miners/{miner.client.hotkey}")).json()
                causes = ", ".join(t["cause"] for t in view["transitions"][-4:]) or "—"
                print(
                    f"    {miner.name:<12} budget={view['budget']:<3} "
                    f"in_flight={view['in_flight']:<3} last causes: {causes}"
                )

            health = (await http.get(f"{API}/v1/health")).json()
            print("\n    coverage gate (85%):")
            by_hotkey = {m.client.hotkey: m.name for m in miners}
            for hotkey, c in sorted(
                health["coverage"].items(), key=lambda kv: by_hotkey.get(kv[0], "")
            ):
                mark = "eligible" if c["eligible"] else "EARNS NOTHING"
                print(
                    f"    {by_hotkey.get(hotkey, hotkey[:10]):<12} "
                    f"assigned={c['assigned']:<5} returned={c['returned']:<5} "
                    f"coverage={c['coverage']:.0%}  {mark}"
                )

            root = (await http.post(f"{API}/v1/admin/rounds/{round_id}/close")).json()
            print(f"\n  round closed, anchored root {root['anchor_root'][:32]}…")

        import json as _json
        async with httpx.AsyncClient(timeout=30) as http:
            (ROOT / "local" / "dump-round.json").write_text(
                _json.dumps((await http.get(f"{API}/v1/rounds/{round_id}")).json()))
            (ROOT / "local" / "dump-log.json").write_text(
                _json.dumps((await http.get(f"{API}/v1/rounds/{round_id}/log")).json()))
        receipts = [r for m in miners for r in m.receipts]
        (ROOT / "local" / "receipts.json").write_text(_json.dumps(receipts))
        print(f"  miners kept {len(receipts)} signed receipts")

        print("\n" + "=" * 78)
        print(
            "Now verifying as a miner would, with no access to the server's internals:"
        )
        print("=" * 78)
        sys.stdout.flush()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "verify_round.py"),
                "--api",
                API,
                "--round",
                round_id,
                "--receipts",
                str(ROOT / "local" / "receipts.json"),
            ]
        )
        return result.returncode
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
