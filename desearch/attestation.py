"""Checks a task API attestation: the TDX quote, the measured OS and compose, and the key bound into it (docs/tee.md)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import yaml

ATTESTATION_KIND = "sn22-task-api-attestation"
ATTESTATION_VERSION = 1
REPORT_DATA_PREFIX = b"sn22-task-api-attestation-v1:"
RUNTIME_EVENT_TYPE = 0x08000001
TEE_TYPE_TDX = 0x81
HEADER_BYTES = 48
BODY_HEADER_BYTES = 6
TD_REPORT_BYTES = 584
# Offsets inside the TD report body (TDX 1.0, which TDX 1.5 bodies start with).
TD_ATTRIBUTES = slice(120, 128)
MRTD = slice(136, 184)
RTMRS = [slice(328 + 48 * i, 376 + 48 * i) for i in range(4)]
REPORT_DATA = slice(520, 584)
DEBUG_BIT = 0x01

# Release pins. Production values are filled in when the CVM is deployed; until then an empty pin
# fails closed, and only `Policy(allow_any_os=True)` (testnets, the dstack simulator) gets past it.
# Each entry is a dstack production OS image at the VM size the API runs with.
PRODUCTION_OS: tuple["OsMeasurement", ...] = ()
# sha256(PPID) of the machines allowed to run the API; empty allows any genuine TDX machine.
ALLOWED_DEVICES: frozenset[str] = frozenset()
# The only variables the deployer may set outside the measured compose: storage credentials and the
# hotkey that anchors the log (registered on the subnet, so it cannot be made inside the CVM).
CREDENTIAL_ENVS: frozenset[str] = frozenset(
    {
        "CF_R2_ACCESS_KEY_ID",
        "CF_R2_SECRET_ACCESS_KEY",
        "PAGES_S3_ACCESS_KEY_ID",
        "PAGES_S3_SECRET_ACCESS_KEY",
        "TASK_API_ANCHOR_KEY_URI",
    }
)
TCB_STATUSES: frozenset[str] = frozenset({"UpToDate"})
TCB_ADVISORIES: frozenset[str] = frozenset()
PUBLISHED_PORTS: frozenset[int] = frozenset({443})
# The compose must name these exactly, so the API checks approvals the way validators do.
PINNED_SETTINGS = {
    "TASK_API_KEY_SOURCE": "cvm",
    "TASK_API_CUSTODY": "1",
    "TASK_API_TEE_SIMULATED": "0",
    "TASK_API_CHAIN_FINALIZED": "1",
    "TASK_API_CHAIN_COUNT_ON": "0",
    "TASK_API_GOV_THRESHOLD": "0.5",
    "TASK_API_GOV_TIMELOCK_BLOCKS": "50400",
    "TASK_API_GOV_MIN_COLDKEYS": "3",
}
# The compose must give these at least this much: two endpoints agreeing on every chain read and anchor, and
# each upload locked for the day it can still be checked in (the API refuses to start with less too).
MINIMUM_SETTINGS = {
    "TASK_API_CHAIN_QUORUM": 2,
    "TASK_API_ANCHOR_QUORUM": 2,
    "TASK_API_UPLOAD_LOCK_S": 86400,
    "TASK_API_GOV_MIN_COLDKEYS": 3,
}
# ... and name these, so the log is anchored.
REQUIRED_SETTINGS = ("TASK_API_ANCHOR_NETWORK",)
FORBIDDEN_MOUNTS = ("/var/run/docker.sock", "/run/docker.sock")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class OsMeasurement:
    name: str
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str


@dataclass(frozen=True)
class Policy:
    os: tuple[OsMeasurement, ...] = PRODUCTION_OS
    devices: frozenset[str] = ALLOWED_DEVICES
    credentials: frozenset[str] = CREDENTIAL_ENVS
    statuses: frozenset[str] = TCB_STATUSES
    advisories: frozenset[str] = TCB_ADVISORIES
    ports: frozenset[int] = PUBLISHED_PORTS
    settings: dict = field(default_factory=lambda: dict(PINNED_SETTINGS))
    minimums: dict = field(default_factory=lambda: dict(MINIMUM_SETTINGS))
    required: tuple[str, ...] = REQUIRED_SETTINGS
    # Testnets and the dstack simulator only: any OS image passes.
    allow_any_os: bool = False
    # The dstack simulator's quotes carry a patched report data, so their signature cannot verify.
    simulated: bool = False


RELEASE = Policy()


@dataclass(frozen=True)
class Dcap:
    status: str
    advisory_ids: tuple[str, ...]
    ppid: bytes


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    document: dict | None = None
    compose_hash: str | None = None
    app_id: str | None = None
    instance_id: str | None = None
    device_id: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    @property
    def signer(self) -> str | None:
        return (self.document or {}).get("signer")

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]


@dataclass(frozen=True)
class Quote:
    version: int
    tee_type: int
    td_attributes: bytes
    mrtd: bytes
    rtmrs: tuple[bytes, bytes, bytes, bytes]
    report_data: bytes

    @property
    def debug(self) -> bool:
        return bool(self.td_attributes[0] & DEBUG_BIT)


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def report_data_of(document: dict) -> bytes:
    """The 64 bytes the API puts in its quote for this document."""
    return hashlib.sha512(REPORT_DATA_PREFIX + canonical(document)).digest()


def document_sha256(document: dict) -> str:
    """What a custody record's `attestation_sha256` names: the document without a nonce."""
    return sha256_hex(canonical({k: v for k, v in document.items() if k != "nonce"}))


def config_sha256(config: dict) -> str:
    """What the quote binds of `/v1/config`: everything but the pause, which is state, not a setting."""
    return sha256_hex(canonical({k: v for k, v in config.items() if k != "paused"}))


def parse_quote(raw: bytes) -> Quote:
    """A TDX quote v4 or v5, as far as its TD report body."""
    if len(raw) < HEADER_BYTES + TD_REPORT_BYTES:
        raise ValueError(f"a quote of {len(raw)} bytes is too short")
    version = int.from_bytes(raw[0:2], "little")
    tee_type = int.from_bytes(raw[4:8], "little")
    if tee_type != TEE_TYPE_TDX:
        raise ValueError(f"tee type {tee_type:#x} is not TDX")
    start = HEADER_BYTES
    if version == 5:
        body_type = int.from_bytes(raw[start : start + 2], "little")
        if body_type not in (2, 3, 4):
            raise ValueError(f"quote v5 body type {body_type} is not a TD report")
        start += BODY_HEADER_BYTES
    elif version != 4:
        raise ValueError(f"quote version {version} is not 4 or 5")
    body = raw[start : start + TD_REPORT_BYTES]
    if len(body) < TD_REPORT_BYTES:
        raise ValueError("the quote ends inside its TD report")
    return Quote(
        version=version,
        tee_type=tee_type,
        td_attributes=body[TD_ATTRIBUTES],
        mrtd=body[MRTD],
        rtmrs=tuple(body[r] for r in RTMRS),
        report_data=body[REPORT_DATA],
    )


def event_digest(event: dict) -> bytes:
    """A dstack runtime event's digest from its name and payload; any other event's as logged."""
    payload = bytes.fromhex(event.get("event_payload") or "")
    if int(event.get("event_type", 0)) != RUNTIME_EVENT_TYPE:
        return bytes.fromhex(event.get("digest") or "")
    name = event.get("event", "")
    if int(event.get("version") or 1) == 2:
        preimage = canonical(
            {"name": name, "payload": payload.hex(), "type": RUNTIME_EVENT_TYPE}
        )
    else:
        preimage = (
            RUNTIME_EVENT_TYPE.to_bytes(4, "little")
            + b":"
            + name.encode()
            + b":"
            + payload
        )
    return hashlib.sha384(preimage).digest()


def event_log_of(attestation: dict) -> list[dict]:
    found = attestation.get("event_log") or []
    return json.loads(found) if isinstance(found, str) else list(found)


def replay_rtmr3(events: list[dict]) -> bytes:
    rtmr = bytes(48)
    for event in events:
        if int(event.get("imr", -1)) == 3:
            rtmr = hashlib.sha384(rtmr + event_digest(event)).digest()
    return rtmr


def runtime_events(events: list[dict]) -> dict[str, list[bytes]]:
    """Each IMR 3 runtime event's payloads by name, in log order."""
    found: dict[str, list[bytes]] = {}
    for event in events:
        if (
            int(event.get("imr", -1)) == 3
            and int(event.get("event_type", 0)) == RUNTIME_EVENT_TYPE
        ):
            found.setdefault(event.get("event", ""), []).append(
                bytes.fromhex(event.get("event_payload") or "")
            )
    return found


def compose_problems(app_compose: dict, policy: Policy) -> list[str]:
    """What in the app compose a validator does not accept; empty when it passes."""
    problems = []
    if app_compose.get("runner") != "docker-compose":
        problems.append(f"runner is {app_compose.get('runner')!r}, not docker-compose")
    if app_compose.get("kms_enabled") is not True:
        problems.append("kms_enabled is not true")
    if app_compose.get("local_key_provider_enabled"):
        problems.append("local_key_provider_enabled is set")
    if app_compose.get("key_provider") not in (None, "kms"):
        problems.append(f"key_provider is {app_compose.get('key_provider')!r}")
    if app_compose.get("pre_launch_script"):
        problems.append("it has a pre_launch_script")
    if app_compose.get("init_script"):
        problems.append("it has an init_script")
    extra = sorted(set(app_compose.get("allowed_envs") or []) - policy.credentials)
    if extra:
        problems.append(f"allowed_envs lets the deployer set {', '.join(extra)}")
    try:
        services = (
            yaml.safe_load(app_compose.get("docker_compose_file") or "") or {}
        ).get("services") or {}
    except yaml.YAMLError as error:
        return problems + [f"docker_compose_file is not YAML: {error}"]
    if not services:
        problems.append("docker_compose_file has no services")
    pinned = {}
    for name, service in services.items():
        problems.extend(service_problems(name, service or {}, policy))
        pinned.update(environment_of(service or {}))
    for key, value in policy.settings.items():
        if not same_setting(pinned.get(key), value):
            problems.append(f"{key} is {pinned.get(key)!r}, the release pins {value!r}")
    for key, least in policy.minimums.items():
        try:
            enough = float(pinned.get(key, "")) >= least
        except ValueError:
            enough = False
        if not enough:
            problems.append(f"{key} is {pinned.get(key)!r}, the release needs at least {least}")
    for key in policy.required:
        if not str(pinned.get(key, "")).strip():
            problems.append(f"{key} is not set")
    return problems


def service_problems(name: str, service: dict, policy: Policy) -> list[str]:
    problems = []
    if "build" in service:
        problems.append(f"{name} is built, not pulled by digest")
    if not DIGEST.search(str(service.get("image", ""))):
        problems.append(
            f"{name}'s image {service.get('image')!r} is not pinned by @sha256"
        )
    if service.get("privileged"):
        problems.append(f"{name} is privileged")
    for key in ("network_mode", "pid", "ipc"):
        if str(service.get(key, "")) == "host":
            problems.append(f"{name} shares the host's {key}")
    if service.get("env_file"):
        problems.append(f"{name} reads an env_file")
    for volume in service.get("volumes") or []:
        source = (
            volume.get("source", "")
            if isinstance(volume, dict)
            else str(volume).split(":")[0]
        )
        if source in FORBIDDEN_MOUNTS:
            problems.append(f"{name} mounts the docker socket")
    for port in published_ports(service):
        if port not in policy.ports:
            problems.append(f"{name} publishes port {port}")
    for key, value in environment_of(service).items():
        if "${" in str(value) and key not in policy.credentials:
            problems.append(f"{name} takes {key} from the deployer, not the compose")
    return problems


def environment_of(service: dict) -> dict[str, str]:
    found = service.get("environment") or {}
    if isinstance(found, list):
        found = dict(
            item.split("=", 1) if "=" in item else (item, "") for item in found
        )
    return {str(k): "" if v is None else str(v) for k, v in found.items()}


def published_ports(service: dict) -> list[int]:
    ports = []
    for entry in service.get("ports") or []:
        if isinstance(entry, dict):
            published = entry.get("published", entry.get("target"))
        else:
            parts = str(entry).split("/")[0].split(":")
            published = parts[-2] if len(parts) >= 2 else parts[-1]
        try:
            ports.append(int(str(published).split("-")[0]))
        except ValueError:
            ports.append(-1)
    return ports


def same_setting(found: str | None, pinned: str) -> bool:
    if found is None:
        return False
    try:
        return float(found) == float(pinned)
    except ValueError:
        return found == pinned


def os_of(quote: Quote, policy: Policy) -> OsMeasurement | None:
    measured = (quote.mrtd.hex(), *(r.hex() for r in quote.rtmrs[:3]))
    for pinned in policy.os:
        if (pinned.mrtd, pinned.rtmr0, pinned.rtmr1, pinned.rtmr2) == measured:
            return pinned
    return None


async def dcap_qvl_verify(raw_quote: bytes) -> Dcap:
    """Intel's signature chain and TCB status, with collateral from Phala's PCCS."""
    import dcap_qvl

    found = await dcap_qvl.get_collateral_and_verify(raw_quote)
    return Dcap(
        status=str(found.status),
        advisory_ids=tuple(found.advisory_ids or ()),
        ppid=bytes(found.ppid or b""),
    )


Verifier = Callable[[bytes], Awaitable[Dcap]]


async def verify(
    attestation: dict,
    *,
    signer: str | None = None,
    compose_hash: str | None = None,
    document_hash: str | None = None,
    nonce: str | None = None,
    policy: Policy = RELEASE,
    dcap: Verifier = dcap_qvl_verify,
) -> Report:
    """Every check of docs/tee.md, each with its reason; `report.ok` only when all pass."""
    report = Report()
    try:
        raw = bytes.fromhex(attestation.get("quote") or "")
        quote = parse_quote(raw)
    except ValueError as error:
        report.add("quote", False, str(error))
        return report
    report.add("quote", True, f"TDX quote v{quote.version}")
    await check_platform(report, raw, quote, policy, dcap)
    check_os(report, quote, policy)
    events = event_log_of(attestation)
    replayed = replay_rtmr3(events)
    if not report.add(
        "rtmr3",
        replayed == quote.rtmrs[3],
        (
            "the event log replays to the quote's RTMR3"
            if replayed == quote.rtmrs[3]
            else f"the event log replays to {replayed.hex()}, the quote has {quote.rtmrs[3].hex()}"
        ),
    ):
        return report
    check_events(report, attestation, runtime_events(events), compose_hash, policy)
    check_document(report, attestation, quote, signer, document_hash, nonce)
    return report


async def check_platform(
    report: Report, raw: bytes, quote: Quote, policy: Policy, dcap
) -> None:
    report.add(
        "debug", not quote.debug, "debug is on" if quote.debug else "debug is off"
    )
    if policy.simulated:
        report.add("dcap", True, "not checked: simulator quote")
        report.add("device", True, "not checked: simulator quote")
        return
    try:
        found = await dcap(raw)
    except Exception as error:
        report.add("dcap", False, f"the quote does not verify: {error}")
        return
    unlisted = sorted(set(found.advisory_ids) - policy.advisories)
    report.add(
        "dcap",
        found.status in policy.statuses and not unlisted,
        f"TCB {found.status}"
        + (f", advisories {', '.join(unlisted)}" if unlisted else ""),
    )
    report.device_id = sha256_hex(found.ppid) if found.ppid else None
    if not policy.devices:
        report.add("device", True, "any device: the release has no allowlist")
    else:
        report.add(
            "device",
            report.device_id in policy.devices,
            f"device {report.device_id}",
        )


def check_os(report: Report, quote: Quote, policy: Policy) -> None:
    pinned = os_of(quote, policy)
    if pinned:
        report.add("os", True, pinned.name)
    elif policy.allow_any_os:
        report.add("os", True, f"not pinned (allowed): mrtd {quote.mrtd.hex()[:16]}…")
    elif not policy.os:
        report.add("os", False, "the release pins no OS image, so none is accepted")
    else:
        report.add(
            "os",
            False,
            f"mrtd {quote.mrtd.hex()} and its RTMRs are not a pinned OS image",
        )


def check_events(report, attestation, events, compose_hash, policy) -> None:
    hashes = events.get("compose-hash", [])
    if not report.add(
        "compose_hash", len(hashes) == 1, f"{len(hashes)} compose-hash events"
    ):
        return
    measured = hashes[0].hex()
    report.compose_hash = measured
    report.app_id = b"".join(events.get("app-id", [])[:1]).hex() or None
    report.instance_id = b"".join(events.get("instance-id", [])[:1]).hex() or None
    app_compose = attestation.get("app_compose") or ""
    if not report.add(
        "app_compose",
        sha256_hex(app_compose.encode()) == measured,
        "app_compose hashes to the measured compose hash",
    ):
        return
    if compose_hash is not None:
        report.add(
            "expected_compose", measured == compose_hash, f"compose hash {measured}"
        )
    try:
        problems = compose_problems(json.loads(app_compose), policy)
    except (json.JSONDecodeError, AttributeError) as error:
        problems = [f"app_compose is not a JSON object: {error}"]
    report.add("compose_policy", not problems, "; ".join(problems) or "passes")
    providers = events.get("key-provider", [])
    try:
        name = json.loads(providers[0]).get("name") if providers else None
    except (json.JSONDecodeError, AttributeError):
        name = None
    report.add("key_provider", name == "kms", f"key provider {name!r}")


def check_document(report, attestation, quote, signer, document_hash, nonce) -> None:
    document = attestation.get("document")
    if not isinstance(document, dict):
        report.add("document", False, "no document")
        return
    report.document = document
    report.add(
        "report_data",
        report_data_of(document) == quote.report_data,
        (
            "the quote binds the document"
            if report_data_of(document) == quote.report_data
            else "the quote's report data is not this document's"
        ),
    )
    report.add(
        "document",
        document.get("kind") == ATTESTATION_KIND
        and document.get("version") == ATTESTATION_VERSION,
        f"{document.get('kind')} v{document.get('version')}",
    )
    if signer is not None:
        report.add(
            "signer",
            document.get("signer") == signer,
            f"signer {document.get('signer')}",
        )
    if document_hash is not None:
        report.add(
            "document_hash",
            document_sha256(document) == document_hash,
            f"document {document_sha256(document)}",
        )
    if nonce is not None:
        report.add(
            "nonce", document.get("nonce") == nonce, f"nonce {document.get('nonce')}"
        )


def spki_sha256(certificate_der: bytes) -> str:
    """sha256 of a certificate's SubjectPublicKeyInfo, as `tls_spki_sha256` names it."""
    _, tbs_start, _ = der_children(certificate_der, 0)[0]
    fields = der_children(certificate_der, tbs_start)
    # The version is optional; the key follows serial, signature, issuer, validity and subject.
    skip = 1 if fields[0][0] == 0xA0 else 0
    _, start, end = fields[skip + 5]
    return sha256_hex(certificate_der[start:end])


def der_children(data: bytes, start: int) -> list[tuple[int, int, int]]:
    """The TLVs inside the constructed TLV at `start`, each as (tag, start, end)."""
    _, end, position = der_tlv(data, start)
    children = []
    while position < end:
        tag, child_end, _ = der_tlv(data, position)
        children.append((tag, position, child_end))
        position = child_end
    return children


def der_tlv(data: bytes, start: int) -> tuple[int, int, int]:
    """(tag, end of the TLV, start of its content) for the TLV at `start`."""
    tag = data[start]
    length = data[start + 1]
    position = start + 2
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[position : position + count], "big")
        position += count
    return tag, position + length, position
