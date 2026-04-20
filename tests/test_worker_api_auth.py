"""
Integration test for worker API auth middleware.

Runs the real FastAPI app via TestClient — no network, no wallets,
no bittensor chain required. Tests the full HTTP request pipeline.

Run: python -m pytest tests/test_worker_api_auth.py -v
"""

import json

import pytest
from fastapi.testclient import TestClient
from substrateinterface import Keypair

from neurons.miners.api import app
from neurons.miners.worker_auth import init
from neurons.validators.worker_client import _sign_request

MINER_KP = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
VALIDATOR_KP = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


class _FakeHotkey:
    def __init__(self, keypair: Keypair):
        self.ss58_address = keypair.ss58_address
        self._keypair = keypair

    def sign(self, data: bytes) -> bytes:
        return self._keypair.sign(data)


@pytest.fixture(autouse=True, scope="module")
def _setup_auth():
    init(MINER_KP.ss58_address, "test")


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _signed_post(client: TestClient, endpoint: str, body_dict: dict, keypair=None):
    kp = keypair or VALIDATOR_KP
    body = json.dumps(body_dict).encode()
    headers = _sign_request(_FakeHotkey(kp), MINER_KP.ss58_address, body)
    headers["Content-Type"] = "application/json"
    return client.post(endpoint, content=body, headers=headers)


def test_no_auth_headers_returns_401(client):
    resp = client.post("/web/search", json={"query": "test"})
    assert resp.status_code == 401


def test_valid_auth_passes_middleware(client):
    resp = _signed_post(client, "/web/search", {"query": "bitcoin price"})
    # Endpoint may fail (no real search backend) but auth must pass (not 401)
    assert resp.status_code != 401


def test_tampered_body_returns_401(client):
    body = json.dumps({"query": "bitcoin"}).encode()
    headers = _sign_request(_FakeHotkey(VALIDATOR_KP), MINER_KP.ss58_address, body)
    headers["Content-Type"] = "application/json"

    tampered = json.dumps({"query": "ethereum"}).encode()
    resp = client.post("/web/search", content=tampered, headers=headers)
    assert resp.status_code == 401


def test_forged_signature_returns_401(client):
    attacker = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    body = json.dumps({"query": "test"}).encode()

    # Attacker signs, but we claim to be VALIDATOR_KP
    headers = _sign_request(_FakeHotkey(attacker), MINER_KP.ss58_address, body)
    headers["X-Hotkey"] = VALIDATOR_KP.ss58_address  # lie about identity
    headers["Content-Type"] = "application/json"

    resp = client.post("/web/search", content=body, headers=headers)
    assert resp.status_code == 401


def test_expired_nonce_returns_401(client):
    body = json.dumps({"query": "test"}).encode()
    headers = _sign_request(_FakeHotkey(VALIDATOR_KP), MINER_KP.ss58_address, body)
    headers["X-Nonce"] = "1000000000"  # year 2001
    headers["Content-Type"] = "application/json"

    resp = client.post("/web/search", content=body, headers=headers)
    assert resp.status_code == 401


def test_wrong_miner_hotkey_returns_401(client):
    """Request signed for a different miner than the one running."""
    other_miner = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    body = json.dumps({"query": "test"}).encode()

    headers = _sign_request(_FakeHotkey(VALIDATOR_KP), other_miner.ss58_address, body)
    headers["Content-Type"] = "application/json"

    resp = client.post("/web/search", content=body, headers=headers)
    assert resp.status_code == 401


def test_health_endpoint_no_auth_needed(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Miner worker API is running"


def test_twitter_search_auth_works(client):
    resp = _signed_post(client, "/twitter/search", {"query": "bitcoin"})
    assert resp.status_code != 401


def test_twitter_id_auth_works(client):
    resp = _signed_post(client, "/twitter/id", {"id": "123456"})
    assert resp.status_code != 401
