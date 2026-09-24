import asyncio

from desearch.client import TaskApiError
from neurons.validators.crawl import CrawlValidator


class Api:
    hotkey = "v"

    def __init__(self, failing: str):
        self.failing = failing
        self.reasons = []

    async def post(self, path: str, body: dict | None = None) -> dict:
        self.reasons.append(body["reason"])
        if body["reason"] == self.failing:
            raise TaskApiError(502, "storage is down")
        return {}


def test_a_missing_upload_that_cannot_be_checked_is_handed_back_plainly():
    api = Api(failing="missing")
    asyncio.run(CrawlValidator(api, None, None).hand_back("t", "missing"))
    assert api.reasons == ["missing", "download"]


def test_other_hand_backs_are_not_retried():
    api = Api(failing="provider")
    asyncio.run(CrawlValidator(api, None, None).hand_back("t", "provider"))
    assert api.reasons == ["provider"]
