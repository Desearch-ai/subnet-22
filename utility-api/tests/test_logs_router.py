from datetime import datetime, timezone
from unittest.mock import AsyncMock

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.logs.router import router
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


def create_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def build_payload(**overrides):
    payload = {
        "query_kind": "organic",
        "search_type": "ai_search",
        "netuid": 22,
        "scoring_epoch_start": None,
        "miner_uid": 11,
        "miner_hotkey": "miner-hotkey",
        "miner_coldkey": "miner-coldkey",
        "validator_uid": 7,
        "validator_hotkey": "validator-hotkey",
        "validator_coldkey": "validator-coldkey",
        "request_query": "what is bittensor",
        "status_code": 200,
        "process_time": 1.23,
        "total_reward": None,
        "response_payload": {"completion": "response"},
        "reward_payload": None,
    }
    payload.update(overrides)
    return payload


def test_save_logs_inserts_batch():
    app = create_test_app()
    session = AsyncMock()
    session.execute.return_value = FakeResult(rowcount=2)

    async def override_session():
        yield session

    async def override_hotkey():
        return "validator-hotkey"

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_hotkey] = override_hotkey

    client = TestClient(app)

    response = client.post(
        "/logs",
        json={
            "logs": [build_payload(), build_payload(miner_uid=12, miner_hotkey="m2")]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"inserted": 2}
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


def test_save_logs_accepts_scoring_payload():
    app = create_test_app()
    session = AsyncMock()
    session.execute.return_value = FakeResult(rowcount=1)

    async def override_session():
        yield session

    async def override_hotkey():
        return "validator-hotkey"

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_hotkey] = override_hotkey

    client = TestClient(app)

    response = client.post(
        "/logs",
        json={
            "logs": [
                build_payload(
                    query_kind="scoring",
                    scoring_epoch_start=datetime(
                        2026, 3, 14, 10, 0, tzinfo=timezone.utc
                    ).isoformat(),
                    search_type="web_search",
                    total_reward=0.9,
                    reward_payload={
                        "total_reward": 0.9,
                        "components": {"search": 1.0},
                        "original_components": {"search": 0.7},
                        "validator_scores": {"search": {"11": 0.7}},
                        "penalties": {},
                        "event_slice": {"rewards": 0.9},
                    },
                )
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"inserted": 1}
