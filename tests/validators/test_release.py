import asyncio

from desearch.client import TaskApiError
from neurons.validators.crawl import CrawlValidator
from tests.validators.test_worklist import Listed, manifest


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


class Refusing:
    hotkey = "v"

    def __init__(self, status: int, detail: str):
        self.error = TaskApiError(status, detail)

    async def post(self, path: str, body: dict | None = None) -> dict:
        raise self.error


class Crashing(Listed):
    async def check(self, job: dict) -> dict | None:
        raise RuntimeError("boom")


def test_a_verdict_for_an_upload_finalized_meanwhile_is_no_fault_and_stays_ours():
    checker = CrawlValidator(Refusing(409, "no such open upload"), None, None)
    counted = asyncio.run(checker.submit_verdict("t", {"verdict": "pass"}))
    assert counted is False and checker.trouble is None
    assert "t" in checker.reported, "not offered to us again"


def test_a_task_the_checker_crashes_on_is_put_off_and_counted_against_it():
    checker = Crashing([manifest("t")], Refusing(409, ""))
    asyncio.run(checker.run(asyncio.Event(), idle_exit=1))
    assert "t" in checker.deferred and checker.failures == 1


def test_a_lost_upload_is_reported_and_the_rest_is_tried_again_later():
    api = Api(failing="")
    checker = CrawlValidator(api, None, None)
    asyncio.run(checker.hand_back("t", "missing"))
    asyncio.run(checker.hand_back("u", "download"))

    assert api.reasons == ["missing"]
    assert set(checker.deferred) == {"t", "u"}


def test_a_failed_report_of_a_lost_upload_does_not_raise():
    api = Api(failing="missing")
    checker = CrawlValidator(api, None, None)
    asyncio.run(checker.hand_back("t", "missing"))
    assert api.reasons == ["missing"] and "t" in checker.deferred


def test_a_checker_in_trouble_says_why_and_recovers():
    checker = CrawlValidator(Api(failing=""), None, None)
    assert checker.trouble is None

    for _ in range(3):
        checker.provider_failed()
    assert "provider" in checker.trouble
    checker.provider_worked()
    assert checker.trouble is None

    for _ in range(3):
        checker.scoring_failed()
    assert "could not be scored" in checker.trouble
    checker.scoring_worked()
    assert checker.trouble is None


def test_a_validator_the_api_refuses_is_in_trouble_until_it_is_served_again():
    checker = CrawlValidator(Api(failing=""), None, None)
    checker.note_refusal(TaskApiError(502, "storage is down"))
    assert checker.trouble is None

    checker.note_refusal(
        TaskApiError(403, "this validator disagreed with too many audits")
    )
    assert "refuses this validator" in checker.trouble
    checker.refused = None
    assert checker.trouble is None
