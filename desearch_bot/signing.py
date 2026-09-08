"""Web Bot Auth: sign requests with Ed25519 so a host can prove the traffic is ours."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .sources import USER_AGENT

SIGNATURE_AGENT = "https://www.desearch.ai/crawler"
LABEL = "sig1"
TAG = "web-bot-auth"
LIFETIME = 300

DEFAULT_PORTS = {"https": ":443", "http": ":80"}


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _raw_public(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def authority(url: str) -> str:
    """The @authority component: host, lowercased, without userinfo or a default port."""
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2].lower()
    port = DEFAULT_PORTS.get(parts.scheme)
    return host[: -len(port)] if port and host.endswith(port) else host


def signature_base(components: list[tuple[str, str]], params: str) -> str:
    lines = [f'"{name}": {value}' for name, value in components]
    lines.append(f'"@signature-params": {params}')
    return "\n".join(lines)


def thumbprint(key: Ed25519PublicKey) -> str:
    """RFC 7638 JWK thumbprint, the keyid a verifier looks up in our directory."""
    jwk = {"crv": "Ed25519", "kty": "OKP", "x": _b64url(_raw_public(key))}
    canonical = json.dumps(jwk, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _b64url(hashlib.sha256(canonical).digest())


def public_jwk(key: Ed25519PublicKey) -> dict:
    return {
        "kid": thumbprint(key),
        "kty": "OKP",
        "crv": "Ed25519",
        "x": _b64url(_raw_public(key)),
    }


def generate() -> tuple[str, dict]:
    """A new private key as PEM, with the public JWK to publish in the directory."""
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    return pem, public_jwk(key.public_key())


class Signer:
    def __init__(self, key: Ed25519PrivateKey, agent: str = SIGNATURE_AGENT):
        self.key = key
        self.agent = f'"{agent}"'
        self.keyid = thumbprint(key.public_key())

    def headers(self, url: str) -> dict[str, str]:
        created = int(time.time())
        params = (
            f'("@authority" "signature-agent");created={created}'
            f';keyid="{self.keyid}";alg="ed25519"'
            f';expires={created + LIFETIME};tag="{TAG}"'
        )
        base = signature_base(
            [("@authority", authority(url)), ("signature-agent", self.agent)], params
        )
        signature = self.key.sign(base.encode("utf-8"))
        return {
            "Signature-Agent": self.agent,
            "Signature-Input": f"{LABEL}={params}",
            "Signature": f"{LABEL}=:{base64.b64encode(signature).decode('ascii')}:",
        }


def load(pem: str) -> Signer:
    key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("signing key must be Ed25519")
    return Signer(key)


def from_env() -> Signer | None:
    """Read the key from DESEARCH_SIGNING_KEY_FILE, or inline from DESEARCH_SIGNING_KEY."""
    path = os.environ.get("DESEARCH_SIGNING_KEY_FILE")
    if path:
        return load(Path(path).read_text())
    pem = os.environ.get("DESEARCH_SIGNING_KEY")
    return load(pem.replace("\\n", "\n")) if pem else None


def request_headers(url: str, signer: Signer | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if signer is not None:
        headers.update(signer.headers(url))
    return headers
