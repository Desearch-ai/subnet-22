import pytest

from desearch.miner_config import SearchType
from desearch.protocol import SearchMode
from neurons.validators.clients.validator_service_client import (
    ValidatorServiceClient,
    _enum_value,
)


class _Resp:
    status = 200

    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return {"uid": 7, "axon": {}}


class _Session:
    def __init__(self):
        self.captured = {}

    def post(self, url, json):
        self.captured.update(json)
        return _Resp(self.captured)


def test_enum_value_accepts_str_and_enum():
    assert _enum_value(None) is None
    assert _enum_value("ai_search") == "ai_search"
    assert _enum_value(SearchType.AI_SEARCH) == "ai_search"
    assert _enum_value(SearchMode.FAST) == "fast"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_type, mode, expect_type, expect_mode",
    [
        ("ai_search", "fast", "ai_search", "fast"),
        (SearchType.AI_SEARCH, SearchMode.DEEP, "ai_search", "deep"),
        ("x_search", None, "x_search", None),
    ],
)
async def test_organic_request_serializes_plain_strings(
    monkeypatch, search_type, mode, expect_type, expect_mode
):
    """The scraper validators pass plain strings; the client must not crash."""
    client = ValidatorServiceClient()
    session = _Session()

    async def fake_session(self):
        return session

    monkeypatch.setattr(
        ValidatorServiceClient, "session", property(fake_session), raising=False
    )

    from bittensor import AxonInfo

    monkeypatch.setattr(AxonInfo, "from_dict", staticmethod(lambda d: d))

    await client.get_random_miner(search_type=search_type, mode=mode)

    assert session.captured == {"search_type": expect_type, "mode": expect_mode}
