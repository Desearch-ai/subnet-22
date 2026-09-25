import asyncio

from bittensor_wallet import Keypair

from desearch.manifest import payload
from neurons.validators.crawl import CrawlValidator
from neurons.validators.tasks import DownloadFailed

API_KEY = Keypair.create_from_uri("//task-api-test")
SIGNER = API_KEY.ss58_address
STORAGE = "https://files.example"


class Api:
    hotkey = "v"

    async def post(self, path: str, body: dict | None = None) -> dict:
        return {}


def manifest(task_id: str, seed_block: int = 5, kind: str = "crawl", **extra) -> dict:
    made = {
        "task_id": task_id,
        "kind": kind,
        "round_id": "r",
        "miner": "m",
        "key": f"submitted/dt=2026-09-25/task={task_id}/m-1-abcd1234.parquet",
        "urls": ["https://a/1", "https://a/2"],
        "frozen_block": seed_block - 10,
        "seed_block": seed_block,
        "completed_at": 1.0,
        "deadline": 2.0,
        **extra,
    }
    made["signer"] = SIGNER
    made["signature"] = API_KEY.sign(payload(made)).hex()
    return made


class Listed(CrawlValidator):
    """A checker whose open list and chain are given, so nothing is fetched."""

    def __init__(self, uploads, api=None, current_block: int = 10, signer=SIGNER):
        super().__init__(
            api or Api(),
            None,
            None,
            storage_url=STORAGE + "/",
            seeds=self.seed_for,
            signer=signer,
        )
        self.uploads = uploads
        self.current_block = current_block

    async def open_list(self) -> list[dict]:
        if isinstance(self.uploads, Exception):
            raise self.uploads
        return list(self.uploads)

    async def seed_for(self, block: int) -> str | None:
        return None if block > self.current_block else f"seed-{block}"


class ListedEmbed(Listed):
    kinds = ("crawl", "embed")


def test_the_oldest_listed_upload_whose_seed_block_has_passed_is_taken():
    checker = Listed([manifest("late", seed_block=20), manifest("ready", seed_block=5)])
    job = asyncio.run(checker.next_job())
    assert job["task_id"] == "ready" and job["seed"] == "seed-5"
    assert job["download_url"] == f"{STORAGE}/{job['key']}"
    assert "input" not in job and job["urls"] == ["https://a/1", "https://a/2"]


def test_uploads_of_other_kinds_or_already_on_this_validators_plate_are_skipped():
    checker = Listed(
        [
            manifest("embed", kind="embed"),
            manifest("flying"),
            manifest("later"),
            manifest("done"),
        ]
    )
    checker.in_flight.add("flying")
    checker.defer("later")
    checker.reported["done"] = 1e12
    assert asyncio.run(checker.next_job()) is None

    checker.uploads.append(manifest("fresh"))
    assert asyncio.run(checker.next_job())["task_id"] == "fresh"


def test_a_manifest_the_task_api_did_not_sign_is_not_worked_on():
    forged = {**manifest("t"), "urls": ["https://evil/"]}
    other_key = {**manifest("u"), "signer": SIGNER}
    other_key["signature"] = (
        Keypair.create_from_uri("//impostor").sign(payload(other_key)).hex()
    )
    assert asyncio.run(Listed([forged, other_key]).next_job()) is None
    assert (
        asyncio.run(Listed([forged, manifest("good")]).next_job())["task_id"] == "good"
    )


def test_an_unreachable_open_list_means_no_work_rather_than_a_crash():
    assert asyncio.run(Listed(DownloadFailed("open list: HTTP 503")).next_job()) is None
    assert asyncio.run(Listed(ValueError("not json")).next_job()) is None


def test_an_embed_upload_brings_the_link_to_its_input():
    listed = manifest(
        "e",
        kind="embed",
        model="m",
        texts=3,
        chars=10,
        input_key="embed-inputs/e.parquet",
        input_sha256="ab" * 32,
    )
    job = asyncio.run(ListedEmbed([listed]).next_job())
    assert job["input"] == {
        "url": f"{STORAGE}/embed-inputs/e.parquet",
        "sha256": "ab" * 32,
    }
    assert asyncio.run(Listed([listed]).next_job()) is None, (
        "a crawl checker leaves embeds alone"
    )
