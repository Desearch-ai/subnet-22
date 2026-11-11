"""Run `python -m crawler site` (or `desearch-bot crawl-sites` with --rust) against the replay server and write what it did as JSON."""

from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def shim(engine: str, port: int, argv: list[str]) -> int:
    sys.path.insert(0, engine)
    import aiohttp
    import aiohttp.abc

    from crawler import __main__ as cli
    from crawler import net

    class Replay(aiohttp.abc.AbstractResolver):
        async def resolve(self, host, port_=0, family=socket.AF_INET):
            flags = socket.AI_NUMERICHOST
            return [
                {
                    "hostname": host,
                    "host": "127.0.0.1",
                    "port": port,
                    "family": socket.AF_INET,
                    "proto": 0,
                    "flags": flags,
                }
            ]

        async def close(self) -> None:
            return None

    def session(headers=None):
        return aiohttp.ClientSession(
            headers={"User-Agent": net.USER_AGENT, **(headers or {})},
            timeout=net.TIMEOUT,
            auto_decompress=True,
            connector=aiohttp.TCPConnector(limit=2000, resolver=Replay(), ssl=False),
        )

    net.session = session
    return cli.main(argv)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def records(out: Path) -> tuple[list[dict], int]:
    import zstandard

    found, files = [], sorted(out.glob("pages-*.jsonl.zst"))
    for f in files:
        with open(f, "rb") as raw:
            for line in io.TextIOWrapper(
                zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8"
            ):
                if line.strip():
                    rec = json.loads(line)
                    rec.pop("fetched_at")
                    found.append(rec)
    return found, len(files)


def run_one(
    name: str, run: dict, fixture: Path, replay_bin: str, side: list[str], work: Path
) -> dict:
    d = work / name
    d.mkdir()
    (d / "hosts.txt").write_text("".join(f"{line}\n" for line in run["hosts_file"]))
    args = ["--out", str(d / "out"), *run["args"]]
    if run.get("skip"):
        (d / "skip.txt").write_text("".join(f"{u}\n" for u in run["skip"]))
        args += ["--skip", str(d / "skip.txt")]
    if run.get("set"):
        (d / "set.json").write_text(json.dumps(run["set"]))
        args += ["--set", str(d / "set.json")]
    if run.get("seeds"):
        (d / "seeds.txt").write_text("".join(f"{u}\n" for u in run["seeds"]))
        args += ["--seeds-file", str(d / "seeds.txt")]
    if run.get("found"):
        args += ["--found", str(d / "found.txt")]
    port = free_port()
    log, ca = d / "hits.jsonl", d / "ca.pem"
    server = subprocess.Popen(
        [replay_bin, str(fixture), str(port), str(log), str(ca)],
        stdout=subprocess.PIPE,
        text=True,
    )
    server.stdout.readline()
    started = time.monotonic()
    try:
        if side[0] == "python":
            cmd = [
                sys.executable,
                __file__,
                "--shim",
                side[1],
                str(port),
                "site",
                "--hosts-file",
                str(d / "hosts.txt"),
                *args,
            ]
        else:
            cmd = [
                side[1],
                "crawl-sites",
                str(d / "hosts.txt"),
                *args,
                "--test-resolve-all",
                f"127.0.0.1:{port}",
                "--test-ca-cert",
                str(ca),
            ]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    finally:
        server.kill()
    if done.returncode:
        raise SystemExit(f"{side[0]} run {name} failed:\n{done.stderr[-3000:]}")
    lines = [
        json.loads(line)
        for line in (done.stdout + done.stderr).splitlines()
        if line.startswith("{")
    ]
    total = next(line["total"] for line in lines if "total" in line)
    hosts = {
        line["host"]: {k: v for k, v in line.items() if k != "host"}
        for line in lines
        if "host" in line
    }
    hits = [json.loads(line) for line in log.read_text().splitlines()]
    requests: dict[str, list[str]] = {}
    times: dict[str, list[float]] = {}
    for h in hits:
        requests.setdefault(h["host"], []).append(h["target"])
        if h["target"] != "/robots.txt":
            times.setdefault(h["host"], []).append(h["t"])
    gaps = {
        host: round(min((b - a for a, b in zip(ts, ts[1:])), default=0.0), 3)
        for host, ts in times.items()
    }
    recs, files = records(d / "out")
    found = (
        sorted((d / "found.txt").read_text().splitlines()) if run.get("found") else None
    )
    agents = sorted({h["agent"] for h in hits})
    return {
        "total": total,
        "hosts": hosts,
        "requests": requests,
        "gaps": gaps,
        "records": recs,
        "files": files,
        "found": found,
        "agents": agents,
        "seconds": round(time.monotonic() - started, 1),
    }


def main() -> int:
    if sys.argv[1] == "--shim":
        return shim(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    engine, replay_bin, fixture, out = sys.argv[1:5]
    side = (
        ["rust", sys.argv[6]]
        if len(sys.argv) > 6 and sys.argv[5] == "--rust"
        else ["python", engine]
    )
    runs = json.loads(Path(fixture).read_text())["runs"]
    with tempfile.TemporaryDirectory() as work:
        result = {
            name: run_one(name, run, Path(fixture), replay_bin, side, Path(work))
            for name, run in runs.items()
        }
    Path(out).write_text(
        json.dumps(
            {"side": side[0], "runs": result},
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
