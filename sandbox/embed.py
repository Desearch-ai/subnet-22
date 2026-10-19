"""Embed mode: the test set's texts as embed inputs, and a model server that answers with their reference vectors."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
import pyarrow.parquet as pq
from aiohttp import web

from desearch.embedding import INPUT_SCHEMA, EmbeddingModel, model_named, write_parquet
from desearch.kinds import EMBED
from neurons.validators.tasks import UPLOAD_CHANGED

from sandbox import local
from sandbox.local import PAGES_BUCKET, ROOT, UPLOADS_BUCKET, Ports
from sandbox.run import CHANGED_NOTE, Sandbox, get_json, where
from sandbox.urls import host_of

DATASET = "desearch/sn22-sandbox-embed"
REVISION = "b033676920d293a8e750ffb743e1bc7dcd309aba"
FILE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
CACHE = ROOT / "sandbox" / "cache" / "embed" / REVISION
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
# Mainnet's cap on the characters of one input.
MAX_INPUT_CHARS = 3_000_000
EMBED_INPUTS = "embed:inputs"
PARQUET = "application/vnd.apache.parquet"
QUEUE_TARGET = 10
CLAIM_LIMIT_S = 300


def text_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


@dataclass(frozen=True)
class Input:
    """Whole pages of the set, as one embed task's texts."""

    pages: tuple[list[dict], ...]
    texts: int
    chars: int

    def rows(self) -> list[dict]:
        return [row for page in self.pages for row in page]


class EmbedSet:
    """The test texts grouped by page, and each text's reference vector by the sha256 of the text."""

    def __init__(
        self,
        model: EmbeddingModel,
        pages: list[list[dict]],
        references: dict[bytes, bytes],
    ):
        self.model = model
        self.pages = pages
        self.references = references
        self.rows = {row["text_id"]: row for page in pages for row in page}

    @classmethod
    def read(cls, folder: Path) -> EmbedSet:
        manifest = json.loads((folder / "manifest.json").read_text())
        model = model_named(manifest["model"])
        if manifest["dims"] != model.dims:
            raise ValueError(
                f"{folder} has {manifest['dims']} dims, {model.name} has {model.dims}"
            )
        pages: list[list[dict]] = []
        references: dict[bytes, bytes] = {}
        for name in manifest["files"]:
            table = pq.read_table(folder / name)
            given = table.column("reference").to_pylist()
            for row, raw in zip(table.select(INPUT_SCHEMA.names).to_pylist(), given):
                if len(raw) != model.dims * 4:
                    raise ValueError(
                        f"{name}: the reference of {row['text_id']} is not {model.dims} float32 values"
                    )
                references[text_digest(row["text"])] = raw
                if pages and pages[-1][0]["page_key"] == row["page_key"]:
                    pages[-1].append(row)
                else:
                    pages.append([row])
        return cls(model, pages, references)

    def reference(self, text: str) -> np.ndarray | None:
        raw = self.references.get(text_digest(text))
        return None if raw is None else np.frombuffer(raw, dtype="<f4")

    def inputs(self, task_size: int, max_chars: int = MAX_INPUT_CHARS) -> list[Input]:
        """The pages in order, task_size to an input, an input closed early before it would pass max_chars."""
        made: list[Input] = []
        held: list[list[dict]] = []
        texts = chars = 0
        for page in self.pages:
            size = sum(len(row["text"]) for row in page)
            if held and (len(held) == task_size or chars + size > max_chars):
                made.append(Input(tuple(held), texts, chars))
                held, texts, chars = [], 0, 0
            held.append(page)
            texts += len(page)
            chars += size
        if held:
            made.append(Input(tuple(held), texts, chars))
        return made


async def download_set(folder: Path = CACHE) -> Path:
    """The public test set at the pinned revision, downloaded once into folder."""
    async with aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT) as http:
        manifest = await download(http, "manifest.json", folder)
        for name in json.loads(manifest.read_text())["files"]:
            await download(http, name, folder)
    return folder


async def download(http: aiohttp.ClientSession, name: str, folder: Path) -> Path:
    target = folder / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    async with http.get(FILE_URL + name) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            async for chunk in response.content.iter_chunked(1 << 20):
                out.write(chunk)
    partial.rename(target)
    print(f"downloaded {name} from {DATASET}", flush=True)
    return target


def input_key(now: float) -> str:
    return f"embed/inputs/dt={time.strftime('%Y-%m-%d', time.gmtime(now))}/{uuid.uuid4().hex}.parquet"


def input_body(given: Input) -> bytes:
    return write_parquet(given.rows(), INPUT_SCHEMA)


def input_entry(given: Input, key: str, body: bytes, now: float) -> dict:
    """What the planner hands the task API for one input."""
    return {
        "input_key": key,
        "input_sha256": hashlib.sha256(body).hexdigest(),
        "written_at": now,
        "texts": given.texts,
        "chars": given.chars,
        "pages": [
            {
                "page_key": page[0]["page_key"],
                "content_sha1": page[0]["content_sha1"],
                "url": page[0]["url"],
                "host": host_of(page[0]["url"]),
            }
            for page in given.pages
        ],
    }


def openai_error(status: int, message: str) -> web.Response:
    kind = "NotFoundError" if status == 404 else "BadRequestError"
    return web.json_response(
        {"error": {"message": message, "type": kind, "code": status}}, status=status
    )


def reference_app(texts: EmbedSet) -> web.Application:
    """An OpenAI-style model server that knows only the set's texts and answers with their reference vectors."""
    model = texts.model

    async def models(request: web.Request) -> web.Response:
        served = {
            "id": model.name,
            "object": "model",
            "max_model_len": model.max_tokens,
        }
        return web.json_response({"object": "list", "data": [served]})

    async def embeddings(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except ValueError:
            return openai_error(400, "the body is not JSON")
        if body.get("model") != model.name:
            return openai_error(404, f"The model `{body.get('model')}` does not exist.")
        given = body.get("input")
        given = [given] if isinstance(given, str) else given
        if not isinstance(given, list) or not all(isinstance(t, str) for t in given):
            return openai_error(400, "input must be a text or a list of texts")
        encoding = body.get("encoding_format") or "float"
        if encoding not in ("float", "base64"):
            return openai_error(400, f"unknown encoding_format {encoding!r}")
        data = []
        for index, text in enumerate(given):
            vector = texts.reference(text)
            if vector is None:
                return openai_error(
                    400, f"input {index} is not a text of the sandbox's embed set"
                )
            embedding = (
                base64.b64encode(vector.tobytes()).decode()
                if encoding == "base64"
                else vector.tolist()
            )
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        return web.json_response(
            {
                "object": "list",
                "model": model.name,
                "data": data,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/embeddings", embeddings)
    return app


async def serve_reference(texts: EmbedSet, port: int) -> web.AppRunner:
    runner = web.AppRunner(reference_app(texts), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def describe(
    task: dict,
    report: dict,
    texts: int,
    chars: int | None,
    upload: str = "",
    details: str = "",
) -> str:
    """One embed upload's verdict: the sample, the size, the time it took, and every sampled text that did not match."""
    result = {**task, **report}
    head = f"task {task['task_id']} miner {task['miner'][:8]}: {task['verdict'].upper()} ({task['reason']})"
    if result.get("sampled"):
        head += f", {result['matched']}/{result['sampled']} sampled texts matched"
        if result.get("malformed"):
            head += f", {result['malformed']} malformed"
        lowest = result.get("min_similarity", "?")
        if lowest is not None:
            head += f", lowest similarity {lowest}"
    elif task["reason"] == "wrong_size":
        head += ", the upload's size is not one vector for each text"
    elif task["reason"] == UPLOAD_CHANGED:
        head += f", {CHANGED_NOTE}"
    size = f"  {texts:,} texts" + (
        f", {chars:,} characters" if chars is not None else ""
    )
    if task.get("claimed_at") and task.get("completed_at"):
        took = task["completed_at"] - task["claimed_at"]
        size += f", uploaded {took:.1f}s after the claim (limit {CLAIM_LIMIT_S}s)"
    samples = report.get("samples", [])
    mismatched = sorted(
        (s for s in samples if s.get("outcome") == "mismatched"),
        key=lambda s: s["similarity"],
    )
    problems = [
        f"  {origin(s)}: similarity {s['similarity']:.6f}" for s in mismatched
    ] + [
        f"  {origin(s)}: not a finite vector of length 1"
        for s in samples
        if s.get("outcome") == "malformed"
    ]
    return "\n".join([head, size, *problems, *where(upload, details)])


def origin(sample: dict) -> str:
    """The page and position of a sampled text, or its id when the set does not hold it."""
    parts = (
        sample.get("url", sample["text_id"]),
        sample.get("kind"),
        sample.get("position"),
    )
    return " ".join(str(part) for part in parts if part is not None)


@dataclass
class EmbedStats:
    """Totals per miner; the rates count only work checked after the first verdict."""

    first_at: float
    tasks: int = 0
    passed: int = 0
    texts: int = 0
    chars: int = 0
    first_texts: int = 0
    first_chars: int = 0

    def add(self, verdict: str, texts: int, chars: int) -> None:
        if not self.tasks:
            self.first_texts, self.first_chars = texts, chars
        self.tasks += 1
        self.passed += verdict == "pass"
        self.texts += texts
        self.chars += chars

    def line(self, miner: str, budget: int | None, now: float) -> str:
        minutes = (now - self.first_at) / 60
        later = self.texts - self.first_texts
        if later and minutes > 0:
            texts = f"{later / minutes:,.0f}/min"
            chars = f"{(self.chars - self.first_chars) / minutes:,.0f}/min"
        else:
            texts = chars = "rate after the next task"
        return (
            f"  {miner[:8]}: {self.tasks} tasks checked, {self.passed} passed,"
            f" {self.texts:,} texts ({texts}), {self.chars:,} characters ({chars}),"
            f" budget {budget if budget is not None else '?'}"
        )


class EmbedSandbox(Sandbox):
    """Embed tasks made from the test set, checked by the real validator against the set's reference vectors."""

    kind = EMBED

    def __init__(self, ports: Ports, run_dir: Path, texts: EmbedSet, task_size: int):
        super().__init__(ports, run_dir, None, task_size)
        self.texts = texts
        self.inputs = texts.inputs(task_size)
        self.cycle = itertools.cycle(self.inputs)
        self.input_of = {
            row["text_id"]: given for given in self.inputs for row in given.rows()
        }
        self.handed = 0
        self.reference: web.AppRunner | None = None
        self.s3 = None

    def api_env(self, genesis: float) -> dict[str, str]:
        return {
            **local.api_env(self.ports, genesis, self.task_size, embed=True),
            "TASK_API_EMBED_QUEUE_TARGET": str(QUEUE_TARGET),
        }

    def validator_command(self, genesis: float) -> list[str]:
        return [*super().validator_command(genesis), "--embed"]

    def validator_env(self) -> dict[str, str]:
        return {
            **super().validator_env(),
            "EMBED_API_URL": self.ports.reference_url,
            "EMBED_API_KEY": "",
        }

    async def start_services(self, genesis: float) -> None:
        print(
            f"{len(self.texts.rows):,} texts of {len(self.texts.pages):,} pages,"
            f" handed out as {len(self.inputs)} inputs of up to {self.task_size} pages",
            flush=True,
        )
        self.reference = await serve_reference(self.texts, self.ports.reference)
        await super().start_services(genesis)
        self.s3 = local.s3_client(self.ports)

    async def feed(self, http: aiohttp.ClientSession) -> None:
        """Hands over inputs while the API has room, the way the planner does."""
        room = await get_json(http, f"{self.ports.api_url}/v1/room?kind={EMBED}")
        for _ in range(int(room.get("room_tasks") or 0)):
            await asyncio.to_thread(self.hand_over, next(self.cycle))
            self.handed += 1
            if self.handed == len(self.inputs):
                print(
                    "handed over every text of the set; it goes round again,"
                    " each input under a new key",
                    flush=True,
                )

    def hand_over(self, given: Input) -> None:
        body = input_body(given)
        now = time.time()
        key = input_key(now)
        self.s3.put_object(
            Bucket=UPLOADS_BUCKET, Key=key, Body=body, ContentType=PARQUET
        )
        entry = input_entry(given, key, body, now)
        local.push_redis(self.ports, EMBED_INPUTS, json.dumps(entry))

    def banner(self) -> None:
        print(
            f"\nTask API: {self.ports.api_url}\n"
            "Run your embed miner with TASK_API_URL set to that address and EMBED_API_URL"
            " to your model server. Any wallet works here, no registration needed."
            " Verdicts print below as uploads are checked against the set's reference"
            " vectors, with a summary every minute; Ctrl-C stops everything.\n",
            flush=True,
        )

    def read_report(self, key: str) -> dict:
        try:
            found = self.s3.get_object(Bucket=PAGES_BUCKET, Key=key)
            return json.loads(found["Body"].read())
        except Exception:
            return {}

    async def report_verdicts(self, http: aiohttp.ClientSession, since: float) -> None:
        listing = await get_json(
            http, f"{self.ports.api_url}/v1/tasks?since={since}&kind={EMBED}"
        )
        for task in reversed(listing.get("tasks", [])):
            if task["task_id"] in self.seen:
                continue
            self.seen.add(task["task_id"])
            report = await asyncio.to_thread(
                self.read_report, task.get("report_key", "")
            )
            samples = [
                {**sample, **self.text_of(sample)}
                for sample in report.get("samples", [])
            ]
            report = {**report, "samples": samples}
            given = next(
                (
                    self.input_of[s["text_id"]]
                    for s in samples
                    if s["text_id"] in self.input_of
                ),
                None,
            )
            texts = given.texts if given else task.get("texts", 0)
            chars = given.chars if given else task.get("chars")
            self.stats.setdefault(task["miner"], EmbedStats(time.time())).add(
                task["verdict"], texts, chars or 0
            )
            details = self.run_dir / "tasks" / f"{task['task_id']}.json"
            details.write_text(
                json.dumps({**task, **report, "texts": texts, "chars": chars}, indent=1)
            )
            print(
                describe(
                    task, report, texts, chars, self.upload_link(task), str(details)
                ),
                flush=True,
            )

    def text_of(self, sample: dict) -> dict:
        """Where a sampled text comes from, and the text itself when it did not match."""
        row = self.texts.rows.get(sample.get("text_id", ""))
        if row is None:
            return {}
        found = {"url": row["url"], "kind": row["kind"], "position": row["index"]}
        if sample.get("outcome") in ("mismatched", "malformed"):
            found["text"] = row["text"]
        return found

    async def shutdown(self) -> None:
        await super().shutdown()
        if self.reference is not None:
            await self.reference.cleanup()
