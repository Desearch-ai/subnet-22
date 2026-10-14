"""Starts everything a miner needs to test against, feeds the queue, and reports what happens."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from desearch.client import TaskApiClient
from desearch.kinds import CRAWL

from sandbox import local
from sandbox.local import ADMIN_URI, ROOT, Ports
from sandbox.urls import host_of

REFILL_TASKS = 10
LOW_WATER_TASKS = 3
HELD_RECENTLY_S = 30
REVEAL_WAIT_S = 20.0
POLL_S = 3.0
SUMMARY_EVERY_S = 60.0
START_TIMEOUT_S = 60.0
# A confirmed error is a page nobody could load, not a problem with the miner.
FINE = ("matched", "errors_confirmed")
# Kept in the log file, left out of the terminal.
QUIET = ("Blocks left until next epoch", "{'netuid'")


def describe(task: dict, urls: list[dict], upload: str = "", details: str = "") -> str:
    head = (
        f"task {task['task_id']} miner {task['miner'][:8]}: {task['verdict'].upper()}"
        f" ({task['reason']}), paid {task['credited']} of {task['returned']} rows,"
        f" {task['matched']} matched / {task['mismatched']} mismatched"
        f" / {task['unverifiable']} unverifiable"
    )
    problems = [
        f"  {u['url']}: {u.get('outcome') or 'rejected'}"
        + (f", {u['why']}" if u.get("why") else "")
        for u in urls
        if u.get("sampled") and (u.get("outcome") not in FINE or u.get("rejected"))
    ]
    return "\n".join([head, *problems, *where(upload, details)])


def where(upload: str, details: str) -> list[str]:
    lines = [f"  upload:  {upload}"] if upload else []
    return lines + ([f"  details: {details}"] if details else [])


@dataclass
class MinerStats:
    """Totals per miner; the rate counts only work checked after the first verdict."""

    first_at: float
    tasks: int = 0
    passed: int = 0
    returned: int = 0
    paid: int = 0
    first_returned: int = 0

    def add(self, task: dict) -> None:
        if not self.tasks:
            self.first_returned = task["returned"]
        self.tasks += 1
        self.passed += task["verdict"] == "pass"
        self.returned += task["returned"]
        self.paid += task["credited"]

    def line(self, miner: str, budget: int | None, now: float) -> str:
        since_first = self.returned - self.first_returned
        rate = (
            f"{since_first / ((now - self.first_at) / 60):.0f}/min"
            if since_first and now > self.first_at
            else "rate after the next task"
        )
        return (
            f"  {miner[:8]}: {self.tasks} tasks checked, {self.passed} passed,"
            f" {self.returned} pages returned ({rate}),"
            f" {self.paid} rows paid ({100 * self.paid / max(self.returned, 1):.0f}%),"
            f" budget {budget if budget is not None else '?'}"
        )


class Sandbox:
    """Crawl tasks from a URL source; EmbedSandbox swaps in embed inputs and its own reports."""

    kind = CRAWL

    def __init__(self, ports: Ports, run_dir: Path, source, task_size: int):
        self.task_size = task_size
        self.ports = ports
        self.run_dir = run_dir
        self.source = source
        self.processes: dict[str, subprocess.Popen] = {}
        self.stopping = asyncio.Event()
        self.stats: dict = {}
        self.seen: set[str] = set()
        self.fed_at = 0.0
        self.exhausted = False
        self.log_offsets: dict[str, int] = {}
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "tasks").mkdir(parents=True, exist_ok=True)

    def start(self, name: str, command: list[str], env: dict[str, str]) -> None:
        self.processes[name] = subprocess.Popen(
            command,
            stdout=(self.run_dir / "logs" / f"{name}.log").open("ab"),
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, self.stopping.set)
        genesis = time.time()
        try:
            await self.start_services(genesis)
            self.start(
                "validator", self.validator_command(genesis), self.validator_env()
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as http:
                await self.feed(http)
                self.banner()
                await self.watch(http)
        except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as error:
            print(f"the sandbox could not start: {error}", flush=True)
            return 1
        finally:
            await self.shutdown()
        return 0

    def validator_command(self, genesis: float) -> list[str]:
        return [
            sys.executable,
            "-m",
            "sandbox.validator",
            "--port",
            str(self.ports.api),
            "--run-dir",
            str(self.run_dir),
            "--genesis",
            str(genesis),
        ]

    def validator_env(self) -> dict[str, str]:
        return {**os.environ, "PYTHONUNBUFFERED": "1"}

    def api_env(self, genesis: float) -> dict[str, str]:
        return local.api_env(self.ports, genesis, self.task_size)

    async def start_services(self, genesis: float) -> None:
        image = await asyncio.to_thread(local.build_image)
        await asyncio.to_thread(local.start_storage, self.ports, self.run_dir)
        await wait_until(self.storage_ready, "storage")
        await wait_until(self.redis_ready, "Redis")
        await asyncio.to_thread(local.create_buckets, self.ports)
        await asyncio.to_thread(
            local.start_api, image, self.api_env(genesis), self.run_dir / "api"
        )
        self.start("api", local.follow_logs(local.API), dict(os.environ))
        await self.wait_healthy()

    async def storage_ready(self) -> bool:
        url = f"http://127.0.0.1:{self.ports.minio}/minio/health/live"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as http:
            try:
                async with http.get(url) as response:
                    return response.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return False

    async def redis_ready(self) -> bool:
        return await asyncio.to_thread(local.redis_ready, self.ports)

    async def wait_healthy(self, timeout: float = START_TIMEOUT_S) -> None:
        deadline = time.time() + timeout
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as http:
            while time.time() < deadline:
                if self.processes["api"].poll() is not None:
                    raise RuntimeError(
                        f"the task API exited; see {self.run_dir / 'logs' / 'api.log'}"
                    )
                if await get_json(http, f"{self.ports.api_url}/v1/health"):
                    return
                await asyncio.sleep(1)
        raise TimeoutError("the task API did not become healthy")

    async def feed(self, http: aiohttp.ClientSession) -> None:
        """Keeps a few tasks waiting, the way the mainnet feeder keeps the queue topped up."""
        if self.exhausted or time.time() - self.fed_at < REVEAL_WAIT_S:
            return
        health = await get_json(http, f"{self.ports.api_url}/v1/health")
        waiting = (health.get("queue_depth") or {}).get("crawl", 0)
        if waiting >= LOW_WATER_TASKS and not await self.only_held_tasks_left(http):
            return
        urls = await self.source.take(self.task_size * REFILL_TASKS)
        if not urls:
            self.exhausted = True
            print("every URL has been queued", flush=True)
            return
        async with TaskApiClient(self.ports.api_url, ADMIN_URI, timeout=120) as admin:
            enqueued = await admin.post(
                "/v1/admin/enqueue",
                {"urls": [{"host": host_of(u), "url": u} for u in urls]},
            )
        self.fed_at = time.time()
        print(f"queued {len(urls)} URLs as {enqueued['batches']} tasks", flush=True)

    async def only_held_tasks_left(self, http: aiohttp.ClientSession) -> bool:
        """A miner is never given a task it held before, so after a restart those can block it."""
        events = await get_json(
            http, f"{self.ports.api_url}/v1/events?outcome=refused&limit=1"
        )
        last = (events.get("events") or [{}])[0]
        return (last.get("refusal") or {}).get("code") == "ALREADY_HELD" and (
            time.time() - last.get("at", 0) < HELD_RECENTLY_S
        )

    def banner(self) -> None:
        print(
            f"\nTask API: {self.ports.api_url}\n"
            "Run your miner with TASK_API_URL set to that address. Any wallet works"
            " here, no registration needed. Verdicts print below as uploads are"
            " checked, with a summary every minute; Ctrl-C stops everything.\n",
            flush=True,
        )

    async def watch(self, http: aiohttp.ClientSession) -> None:
        started = time.time()
        summary_at = started + SUMMARY_EVERY_S
        while not self.stopping.is_set():
            await asyncio.wait(
                [asyncio.ensure_future(self.stopping.wait())], timeout=POLL_S
            )
            for name in ("api", "validator"):
                if self.processes[name].poll() is not None:
                    print(
                        f"the {name} exited; see {self.run_dir / 'logs' / (name + '.log')}",
                        flush=True,
                    )
                    return
            self.echo_logs()
            await self.feed(http)
            await self.report_verdicts(http, started)
            if time.time() >= summary_at:
                await self.summary(http)
                summary_at = time.time() + SUMMARY_EVERY_S

    async def report_verdicts(self, http: aiohttp.ClientSession, since: float) -> None:
        listing = await get_json(
            http, f"{self.ports.api_url}/v1/tasks?since={since}&kind={self.kind}"
        )
        for task in reversed(listing.get("tasks", [])):
            if task["task_id"] in self.seen:
                continue
            self.seen.add(task["task_id"])
            self.stats.setdefault(task["miner"], MinerStats(time.time())).add(task)
            detail = await get_json(
                http, f"{self.ports.api_url}/v1/tasks/{task['task_id']}"
            )
            details = self.run_dir / "tasks" / f"{task['task_id']}.json"
            details.write_text(
                json.dumps({**task, "urls": detail.get("urls", [])}, indent=1)
            )
            print(
                describe(
                    task, detail.get("urls", []), self.upload_link(task), str(details)
                ),
                flush=True,
            )

    def upload_link(self, task: dict) -> str:
        key = task.get("upload_key")
        return f"{self.ports.storage_url}/{key}" if key else ""

    def echo_logs(self) -> None:
        """The validator's log and the API's warnings, as they are written."""
        for name in ("validator", "api"):
            path = self.run_dir / "logs" / f"{name}.log"
            if not path.exists():
                continue
            with path.open("rb") as log:
                log.seek(self.log_offsets.get(name, 0))
                fresh = log.read()
                self.log_offsets[name] = log.tell()
            for line in fresh.decode("utf-8", "replace").splitlines():
                if line.strip() and not any(quiet in line for quiet in QUIET):
                    print(f"{name:>9} | {line}", flush=True)

    async def summary(self, http: aiohttp.ClientSession) -> None:
        health = await get_json(http, f"{self.ports.api_url}/v1/health")
        waiting = (health.get("queue_depth") or {}).get(self.kind, "?")
        checking = health.get("validation_depth", "?")
        lines = [f"summary: {waiting} tasks waiting, {checking} uploads being checked"]
        now = time.time()
        for miner, stats in self.stats.items():
            about = await get_json(http, f"{self.ports.api_url}/v1/miners/{miner}")
            budget = ((about.get("pools") or {}).get(self.kind) or {}).get("budget")
            lines.append(stats.line(miner, budget, now))
        print("\n".join(lines), flush=True)

    async def shutdown(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.time() + 30
        while time.time() < deadline and any(
            p.poll() is None for p in self.processes.values()
        ):
            await asyncio.sleep(0.5)
        for process in self.processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        await asyncio.to_thread(local.stop_containers)
        await asyncio.to_thread(shutil.rmtree, self.run_dir / "minio", True)


async def get_json(http: aiohttp.ClientSession, url: str) -> dict:
    try:
        async with http.get(url) as response:
            if response.status != 200:
                return {}
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return {}


async def wait_until(ready, what: str, timeout: float = START_TIMEOUT_S) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if await ready():
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"{what} did not start within {timeout:.0f}s")
