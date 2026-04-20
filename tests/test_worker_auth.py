"""
Tests for the worker API auth protocol.

Verifies the sign (validator) -> verify (miner) roundtrip works,
and that bad signatures / expired nonces / wrong hotkeys are rejected.

Run: python -m pytest tests/test_worker_auth.py -v
"""

import time

import pytest
from substrateinterface import Keypair

from neurons.miners.worker_auth import verify_signature
from neurons.validators.worker_client import _sign_request


@pytest.fixture()
def validator_keypair():
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


@pytest.fixture()
def miner_keypair():
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


class _FakeHotkey:
    """Mimics bt.Wallet().hotkey enough for _sign_request."""

    def __init__(self, keypair: Keypair):
        self.ss58_address = keypair.ss58_address
        self._keypair = keypair

    def sign(self, data: bytes) -> bytes:
        return self._keypair.sign(data)


def _sign_and_extract(validator_kp: Keypair, miner_hotkey: str, body: bytes) -> dict:
    fake_hotkey = _FakeHotkey(validator_kp)
    return _sign_request(fake_hotkey, miner_hotkey, body)


def test_roundtrip_success(validator_keypair, miner_keypair):
    body = b'{"query": "bitcoin news"}'
    headers = _sign_and_extract(validator_keypair, miner_keypair.ss58_address, body)

    verify_signature(
        validator_hotkey=headers["X-Hotkey"],
        miner_hotkey=miner_keypair.ss58_address,
        nonce=headers["X-Nonce"],
        signature_hex=headers["X-Signature"],
        body=body,
    )


def test_wrong_body_rejected(validator_keypair, miner_keypair):
    body = b'{"query": "bitcoin news"}'
    headers = _sign_and_extract(validator_keypair, miner_keypair.ss58_address, body)

    with pytest.raises(ValueError, match="Signature verification failed"):
        verify_signature(
            validator_hotkey=headers["X-Hotkey"],
            miner_hotkey=miner_keypair.ss58_address,
            nonce=headers["X-Nonce"],
            signature_hex=headers["X-Signature"],
            body=b'{"query": "tampered"}',
        )


def test_wrong_miner_hotkey_rejected(validator_keypair, miner_keypair):
    body = b'{"query": "test"}'
    headers = _sign_and_extract(validator_keypair, miner_keypair.ss58_address, body)
    other_miner = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    with pytest.raises(ValueError, match="Signature verification failed"):
        verify_signature(
            validator_hotkey=headers["X-Hotkey"],
            miner_hotkey=other_miner.ss58_address,
            nonce=headers["X-Nonce"],
            signature_hex=headers["X-Signature"],
            body=body,
        )


def test_expired_nonce_rejected(validator_keypair, miner_keypair):
    body = b'{"query": "test"}'
    headers = _sign_and_extract(validator_keypair, miner_keypair.ss58_address, body)
    headers["X-Nonce"] = str(int(time.time()) - 120)

    with pytest.raises(ValueError, match="Nonce expired"):
        verify_signature(
            validator_hotkey=headers["X-Hotkey"],
            miner_hotkey=miner_keypair.ss58_address,
            nonce=headers["X-Nonce"],
            signature_hex=headers["X-Signature"],
            body=body,
        )


def test_forged_signature_rejected(validator_keypair, miner_keypair):
    body = b'{"query": "test"}'
    attacker = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    forged_headers = _sign_and_extract(attacker, miner_keypair.ss58_address, body)

    with pytest.raises(ValueError, match="Signature verification failed"):
        verify_signature(
            validator_hotkey=validator_keypair.ss58_address,
            miner_hotkey=miner_keypair.ss58_address,
            nonce=forged_headers["X-Nonce"],
            signature_hex=forged_headers["X-Signature"],
            body=body,
        )
