from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.logs.enums import QueryKind, SearchType
from app.domains.logs.router import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

EPOCH_START = datetime.now(timezone.utc).replace(
    minute=0, second=0, microsecond=0
) - timedelta(hours=1)


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

    def all(self):
        return self._rows


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
        created_at=EPOCH_START + timedelta(minutes=5),
        query_kind=QueryKind.SCORING,
        search_type=SearchType.X_SEARCH,
        netuid=22,
        scoring_epoch_start=EPOCH_START,
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
        tools=["Twitter Search"],
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


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
    assert session.execute.await_count == 1
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


def test_get_scoring_logs_returns_grouped_validator_runs():
    app = create_test_app()
    session = AsyncMock()
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
            "scoring_epoch_start": EPOCH_START.isoformat(),
            "search_type": "x_search",
            "miner_uids": 11,
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
    assert x_group["logs"][0]["tools"] == ["Twitter Search"]
    assert "response_payload" not in x_group["logs"][0]
    assert "reward_payload" not in x_group["logs"][0]

    web_group = payload["groups"][0]
    assert web_group["search_type"] == "web_search"
    assert web_group["request_query"] == "Top websites about AI"
    assert web_group["validator_count"] == 1


def test_get_scoring_logs_without_miner_uid_returns_all_miners_for_hour():
    app = create_test_app()
    session = AsyncMock()
    session.execute.side_effect = [
        FakeSelectResult(
            [
                (EPOCH_START, 11, "Latest AI news", SearchType.X_SEARCH),
                (EPOCH_START, 12, "Best web results", SearchType.WEB_SEARCH),
            ]
        ),
        FakeSelectResult(
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
        ),
    ]

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(
        "/logs/scoring",
        params={
            "scoring_epoch_start": EPOCH_START.isoformat(),
            "search_type": "x_search",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["groups"]) == 2
    assert [group["miner_uid"] for group in payload["groups"]] == [11, 12]


def test_get_scoring_logs_filters_by_validator_uid():
    app = create_test_app()
    session = AsyncMock()
    session.execute.return_value = FakeSelectResult(
        [
            build_log_row(
                validator_uid=3,
                validator_hotkey="validator-3",
                total_reward=0.4,
            ),
        ]
    )

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(
        "/logs/scoring",
        params={"search_type": "x_search", "miner_uids": 11, "validator_uid": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["groups"]) == 1
    group = payload["groups"][0]
    assert [log["validator_uid"] for log in group["logs"]] == [3]

    final_stmt = session.execute.await_args_list[-1].args[0]
    compiled = str(final_stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "validator_uid = 3" in compiled


def test_get_scoring_logs_rejects_epoch_older_than_three_days():
    app = create_test_app()
    session = AsyncMock()

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(
        "/logs/scoring",
        params={
            "search_type": "ai_search",
            "scoring_epoch_start": (EPOCH_START - timedelta(days=4)).isoformat(),
        },
    )

    assert response.status_code == 422
    session.execute.assert_not_awaited()


def test_get_scoring_log_returns_payloads_and_siblings():
    app = create_test_app()
    session = AsyncMock()
    row = build_log_row(
        response_payload={
            "results": [],
            "axon": {"ip": "1.2.3.4", "port": 8091, "process_time": 1.5},
        }
    )
    sibling_id = uuid4()
    session.get.return_value = row
    session.execute.return_value = FakeSelectResult(
        [
            SimpleNamespace(
                id=row.id, validator_uid=2, status_code=200, total_reward=0.4
            ),
            SimpleNamespace(
                id=sibling_id, validator_uid=5, status_code=200, total_reward=0.6
            ),
        ]
    )

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    response = client.get(f"/logs/{row.id}")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=3600"

    payload = response.json()
    assert payload["request_query"] == "Latest AI news"
    assert payload["log"]["response_payload"]["axon"]["ip"] == "0.0.0.0"
    assert payload["log"]["reward_payload"] == {"total_reward": 0.4}
    assert [sibling["id"] for sibling in payload["siblings"]] == [
        str(row.id),
        str(sibling_id),
    ]


def test_get_scoring_log_hides_organic_and_expired_logs():
    app = create_test_app()
    session = AsyncMock()

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)

    for row in (
        build_log_row(query_kind=QueryKind.ORGANIC),
        build_log_row(created_at=EPOCH_START - timedelta(days=4)),
    ):
        session.get.return_value = row
        assert client.get(f"/logs/{row.id}").status_code == 404


def test_public_log_endpoints_are_rate_limited():
    app = create_test_app()
    session = AsyncMock()
    session.get.return_value = None

    async def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session

    client = TestClient(app)
    headers = {"CF-Connecting-IP": "203.0.113.7"}

    for _ in range(30):
        assert client.get(f"/logs/{uuid4()}", headers=headers).status_code == 404

    response = client.get(f"/logs/{uuid4()}", headers=headers)

    assert response.status_code == 429
    assert "retry-after" in response.headers
