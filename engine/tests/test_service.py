from datetime import date

import numpy as np
import pytest
from engine import service
from engine.tests.asgi import Client


class FakeEngine:
    """Ranks three fixed documents, so the service can be tested without the index."""

    def __init__(self):
        self.meta = [
            {"url": "https://a.example/1", "title": "A", "published": "2026-09-01"},
            {"url": "https://b.example/2", "title": "B", "published": "2026-01-05"},
            {"url": "https://c.example/3", "title": "C", "published": ""},
        ]
        self.pub = np.array([date(2026, 9, 1), date(2026, 1, 5), date(1970, 1, 1)])
        self.key_ix = {"a.example/1": 0, "b.example/2": 1, "c.example/3": 2}
        self.weights = None

    def arms(self, question, vector):
        return {
            "head": (np.arange(3), np.ones(3)),
            "chunk": (np.arange(3), np.array([0.61, 0.44, 0.3])),
        }, {}

    def term_coverage(self, question):
        return 0.75

    def fuse(self, arms, weights, date_boost, window):
        self.weights = weights
        return np.arange(3), np.ones(3)

    def text(self, doc):
        return f"body of {self.meta[doc]['title']}"


async def fake_embed(query: str) -> np.ndarray:
    return np.ones(4096, dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(service, "engine", lambda: engine)
    monkeypatch.setattr(service, "query_vectors", dict)
    monkeypatch.setattr(service, "access_key", lambda: "secret")
    monkeypatch.setattr(service, "embed", fake_embed)
    monkeypatch.setattr(service, "passage", lambda eng, doc, vector: f"passage {doc}")
    monkeypatch.setattr(service, "date_window", lambda query, years=False: None)
    client = Client(service.app)
    client.engine = engine
    return client


def test_search_returns_ranked_results_with_a_passage(client):
    response = client.post("/v1/search", json={"query": "who won", "count": 2})

    assert response.status_code == 200
    assert response.json()["confidence"] == {"top_score": 0.61, "term_coverage": 0.75}
    assert response.json()["results"] == [
        {
            "url": "https://a.example/1",
            "title": "A",
            "published_date": "2026-09-01",
            "highlights": ["passage 0"],
        },
        {
            "url": "https://b.example/2",
            "title": "B",
            "published_date": "2026-01-05",
            "highlights": ["passage 1"],
        },
    ]


def test_mode_selects_the_benchmarked_weights(client):
    client.post("/v1/search", json={"query": "q", "mode": "fast"})
    assert client.engine.weights == service.PROFILES["fast"]

    client.post("/v1/search", json={"query": "q"})
    assert client.engine.weights == service.PROFILES["balanced"]


def test_page_text_and_highlights_are_opt_in(client):
    response = client.post(
        "/v1/search",
        json={"query": "q", "count": 1, "highlights": False, "page_text": True},
    )

    assert response.json()["results"] == [
        {
            "url": "https://a.example/1",
            "title": "A",
            "published_date": "2026-09-01",
            "page_text": "body of A",
        }
    ]


def test_date_filter_drops_results_outside_the_window_and_undated_pages(client):
    response = client.post(
        "/v1/search", json={"query": "q", "start_date": "2026-06-01"}
    )

    assert [r["url"] for r in response.json()["results"]] == ["https://a.example/1"]


def test_undated_pages_survive_when_no_window_is_given(client):
    response = client.post("/v1/search", json={"query": "q"})

    assert response.json()["results"][2]["published_date"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "q", "count": 0},
        {"query": "q", "count": 51},
        {"query": "q", "mode": "deep"},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert client.post("/v1/search", json=body).status_code == 422


def test_remote_callers_need_the_access_key(client):
    remote = Client(service.app, client=("203.0.113.5", 4000))

    assert remote.post("/v1/search", json={"query": "q"}).status_code == 401
    assert (
        remote.post(
            "/v1/search", json={"query": "q"}, headers={"Access-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        remote.post(
            "/v1/search", json={"query": "q"}, headers={"Access-Key": "secret"}
        ).status_code
        == 200
    )


def test_confidence_is_zero_when_no_passage_matched(client, monkeypatch):
    monkeypatch.setattr(
        client.engine,
        "arms",
        lambda q, v: ({"chunk": (np.array([]), np.array([]))}, {}),
    )

    response = client.post("/v1/search", json={"query": "q"})

    assert response.json()["confidence"]["top_score"] == 0.0


def test_document_returns_the_indexed_page_by_url(client):
    response = client.post("/v1/document", json={"url": "https://www.a.example/1/"})

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://a.example/1",
        "title": "A",
        "text": "body of A",
    }


def test_document_outside_the_index_is_a_404(client):
    assert (
        client.post(
            "/v1/document", json={"url": "https://elsewhere.example/x"}
        ).status_code
        == 404
    )
