from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.dataset.enums import SearchType
from app.domains.logs.enums import QueryKind
from app.domains.logs.router import router
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSelectResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return FakeScalarResult(self._rows)


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


def build_log_row(**overrides):
    row = SimpleNamespace(
        id=uuid4(),
        created_at=datetime(2026, 3, 14, 10, 5, tzinfo=timezone.utc),
        query_kind=QueryKind.SCORING,
        search_type=SearchType.X_SEARCH,
        netuid=22,
        scoring_epoch_start=datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc),
        miner_uid=11,
        miner_hotkey="miner-hotkey",
        miner_coldkey="miner-coldkey",
        validator_uid=2,
        validator_hotkey="validator-2",
        validator_coldkey="validator-cold-2",
        request_query="Latest AI news",
        status_code=200,
        process_time=1.5,
        total_reward=0.4,
        response_payload={"query": "Latest AI news", "results": []},
        reward_payload={"total_reward": 0.4},
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def build_question_row(**overrides):
    row = SimpleNamespace(
        id=uuid4(),
        query="what is bittensor",
        search_types=[SearchType.X_SEARCH],
        ai_search_tools=None,
        source="desearch",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def test_save_logs_inserts_batch():
    app = create_test_app()
    session = AsyncMock()
    session.execute.side_effect = [
        FakeResult(rowcount=2),
        FakeSelectResult([]),
        FakeResult(rowcount=1),
    ]

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
    assert session.execute.await_count == 3
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


def test_save_logs_adds_questions_for_new_organic_search_types():
    app = create_test_app()
    session = AsyncMock()
    session.execute.side_effect = [
        FakeResult(rowcount=2),
        FakeSelectResult([]),
        FakeResult(rowcount=2),
    ]

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
                    search_type="x_post_by_id",
                    request_query="189203918203",
                ),
                build_payload(
                    search_type="x_posts_by_urls",
                    request_query="https://x.com/test/status/1",
                ),
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"inserted": 2}
    assert session.execute.await_count == 3

    question_insert_stmt = session.execute.await_args_list[2].args[0]
    assert question_insert_stmt.table.name == "questions"


def test_save_logs_updates_existing_question_search_types():
    app = create_test_app()
    session = AsyncMock()
    session.execute.side_effect = [
        FakeResult(rowcount=1),
        FakeSelectResult([build_question_row()]),
        FakeResult(rowcount=1),
    ]

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
                    search_type="x_post_by_id",
                    request_query="what is bittensor",
                )
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"inserted": 1}
    assert session.execute.await_count == 3

    question_update_stmt = session.execute.await_args_list[2].args[0]
    assert question_update_stmt.table.name == "questions"


def test_get_scoring_logs_returns_grouped_validator_runs():
    app = create_test_app()
    session = AsyncMock()
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    session.execute.return_value = FakeSelectResult(
        [
            build_log_row(
                validator_uid=9,
                validator_hotkey="validator-9",
                total_reward=0.9,
                process_time=1.9,
            ),
            build_log_row(
                validator_uid=3,
                validator_hotkey="validator-3",
                total_reward=0.3,
                process_time=1.3,
            ),
            build_log_row(
                search_type=SearchType.WEB_SEARCH,
                request_query="Top websites about AI",
                validator_uid=4,
                validator_hotkey="validator-4",
                response_payload={"query": "Top websites about AI", "results": []},
                reward_payload={"total_reward": 0.7},
                total_reward=0.7,
            ),
        ]
    )

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(
        "/logs/scoring",
        params={
            "scoring_epoch_start": epoch_start.isoformat(),
            "miner_uid": 11,
        },
    )

    assert response.status_code == 200
    assert session.execute.await_count == 1

    payload = response.json()
    assert len(payload["groups"]) == 2

    x_group = payload["groups"][1]
    assert x_group["search_type"] == "x_search"
    assert x_group["request_query"] == "Latest AI news"
    assert x_group["validator_count"] == 2
    assert x_group["reward_min"] == 0.3
    assert x_group["reward_max"] == 0.9
    assert x_group["reward_avg"] == 0.6
    assert [log["validator_uid"] for log in x_group["logs"]] == [3, 9]

    web_group = payload["groups"][0]
    assert web_group["search_type"] == "web_search"
    assert web_group["request_query"] == "Top websites about AI"
    assert web_group["validator_count"] == 1


def test_get_scoring_logs_without_miner_uid_returns_all_miners_for_hour():
    app = create_test_app()
    session = AsyncMock()
    epoch_start = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    session.execute.return_value = FakeSelectResult(
        [
            build_log_row(
                miner_uid=11,
                miner_hotkey="miner-11",
                request_query="Latest AI news",
                search_type=SearchType.X_SEARCH,
            ),
            build_log_row(
                miner_uid=12,
                miner_hotkey="miner-12",
                request_query="Best web results",
                search_type=SearchType.WEB_SEARCH,
            ),
        ]
    )

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(
        "/logs/scoring",
        params={"scoring_epoch_start": epoch_start.isoformat()},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["groups"]) == 2
    assert [group["miner_uid"] for group in payload["groups"]] == [11, 12]
