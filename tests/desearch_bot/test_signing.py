import base64

from cryptography.hazmat.primitives import serialization

from desearch_bot import signing

# RFC 9421 B.1.4, test-key-ed25519.
TEST_KEY = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIJ+DYvh6SEqVTm50DFtMDoQikTmiCqirVv9mWG9qfSnF
-----END PRIVATE KEY-----
"""


def _key():
    return serialization.load_pem_private_key(TEST_KEY.encode(), password=None)


def test_signature_base_matches_rfc_9421_ed25519_vector():
    components = [
        ("date", "Tue, 20 Apr 2021 02:07:55 GMT"),
        ("@method", "POST"),
        ("@path", "/foo"),
        ("@authority", "example.com"),
        ("content-type", "application/json"),
        ("content-length", "18"),
    ]
    params = (
        '("date" "@method" "@path" "@authority" "content-type" "content-length")'
        ';created=1618884473;keyid="test-key-ed25519"'
    )
    base = signing.signature_base(components, params)
    signature = base64.b64encode(_key().sign(base.encode())).decode()
    assert signature == (
        "wqcAqbmYJ2ji2glfAMaRy4gruYYnx2nEFN2HN6jrnDnQCK1u02Gb04v9EDgwUPiu4A0w6vuQv5lIp5WPpBKRCw=="
    )


def test_thumbprint_matches_rfc_7638():
    assert signing.thumbprint(_key().public_key()) == "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U"


def test_public_jwk_carries_the_raw_key():
    jwk = signing.public_jwk(_key().public_key())
    assert jwk["kty"] == "OKP" and jwk["crv"] == "Ed25519"
    assert jwk["x"] == "JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs"
    assert jwk["kid"] == signing.thumbprint(_key().public_key())


def test_authority_drops_default_ports_and_userinfo():
    assert signing.authority("https://Example.COM:443/robots.txt") == "example.com"
    assert signing.authority("http://example.com:80/") == "example.com"
    assert signing.authority("https://example.com:8443/") == "example.com:8443"
    assert signing.authority("https://user@example.com/") == "example.com"


def test_headers_verify_against_our_own_public_key():
    signer = signing.Signer(_key())
    headers = signer.headers("https://news.example.org/sitemap.xml")

    assert headers["Signature-Agent"] == '"https://www.desearch.ai/crawler"'
    label, _, params = headers["Signature-Input"].partition("=")
    assert label == "sig1"
    assert 'tag="web-bot-auth"' in params and 'alg="ed25519"' in params
    assert f'keyid="{signer.keyid}"' in params

    base = signing.signature_base(
        [("@authority", "news.example.org"), ("signature-agent", signer.agent)], params
    )
    raw = base64.b64decode(headers["Signature"].split(":", 1)[1].rstrip(":"))
    _key().public_key().verify(raw, base.encode())


def test_generated_key_round_trips():
    pem, jwk = signing.generate()
    assert signing.load(pem).keyid == jwk["kid"]


def test_request_headers_stay_unsigned_without_a_key(monkeypatch):
    monkeypatch.delenv("DESEARCH_SIGNING_KEY", raising=False)
    monkeypatch.delenv("DESEARCH_SIGNING_KEY_FILE", raising=False)
    assert signing.from_env() is None
    assert set(signing.request_headers("https://example.com/", None)) == {"User-Agent"}
