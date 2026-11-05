#!/usr/bin/env python3
"""Builds app-compose.json from docker-compose.yml and prints its SHA-256, the compose hash.

The compose hash is what the TDX quote measures (the `compose-hash` event in RTMR3) and what validators
approve on chain, so this file is written the same way every time: keys sorted, no whitespace, UTF-8, as
dstack's own tools write it. Anyone can run it on a release's docker-compose.yml with the image digest CI
printed and must get the hash the running CVM attests.

    python3 task-api/deploy/dstack/app_compose.py --image-digest sha256:<64 hex>
    python3 task-api/deploy/dstack/app_compose.py --check        # the checks only, placeholders allowed

It refuses a compose that names an image by tag, takes any value from outside other than the storage
credentials, publishes a port other than the TLS one, or gives a container more than it needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "sn22-task-api"
# The only values the CVM is given from outside: credentials, encrypted to the CVM, never settings.
ALLOWED_ENVS = (
    "CF_R2_ACCESS_KEY_ID",
    "CF_R2_SECRET_ACCESS_KEY",
    "PAGES_S3_ACCESS_KEY_ID",
    "PAGES_S3_SECRET_ACCESS_KEY",
    "TASK_API_ANCHOR_KEY_URI",
)
API_IMAGE = "ghcr.io/desearch-ai/sn22-task-api"
IMAGE_PLACEHOLDER = "sha256:REPLACE_WITH_IMAGE_DIGEST"
TLS_PORT = 443
# Only the API may talk to dstack's agent, and only it and Redis share the socket volume.
DSTACK_SOCKET = "/var/run/dstack.sock"
FORBIDDEN = (
    (re.compile(r"^\s*privileged\s*:\s*true", re.M), "a privileged container"),
    (re.compile(r"^\s*(network_mode|pid|ipc|userns_mode)\s*:", re.M), "a host namespace"),
    (re.compile(r"^\s*(cap_add|devices|device_cgroup_rules|sysctls)\s*:", re.M), "extra capabilities or devices"),
    (re.compile(r"^\s*build\s*:", re.M), "an image built in the CVM"),
    (re.compile(r"^\s*env_file\s*:", re.M), "an env file"),
    (re.compile(r"docker\.sock"), "the Docker socket"),
)


def services(text: str) -> dict[str, str]:
    """Each service's block of the compose file, by name; enough YAML for the checks below."""
    blocks: dict[str, list[str]] = {}
    inside = False
    current = None
    for line in text.splitlines():
        if re.match(r"^\S", line):
            inside = line.startswith("services:")
            current = None
            continue
        if not inside:
            continue
        named = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if named:
            current = named.group(1)
            blocks[current] = []
        elif current:
            blocks[current].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def items(block: str, key: str) -> list[str]:
    """The list under a service's `key:`, e.g. its volumes or ports."""
    found, inside = [], False
    for line in block.splitlines():
        if re.match(rf"^    {key}:\s*$", line):
            inside = True
        elif inside and re.match(r"^      - ", line):
            found.append(line.strip()[2:].strip().strip('"'))
        elif inside and re.match(r"^ {0,4}\S", line):
            inside = False
    return found


def problems(text: str, *, final: bool) -> list[str]:
    # Comments say what they like; only what the compose does is checked.
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    found = []
    for line in re.findall(r"^\s*image\s*:\s*(\S+)", text, re.M):
        if not re.fullmatch(r"[^@\s]+@sha256:([0-9a-f]{64}|REPLACE_WITH_IMAGE_DIGEST)", line):
            found.append(f"image {line} is not pinned by @sha256: digest")
    if final and IMAGE_PLACEHOLDER in text:
        found.append("the API image digest is still a placeholder: pass --image-digest")
    if final and re.search(r"REPLACE_WITH_(?!IMAGE_DIGEST)", text):
        found.append("a REPLACE_WITH_ value is still a placeholder")
    referenced = set()
    for match in re.finditer(r"(?<!\$)\$\{?([A-Za-z_][A-Za-z0-9_]*)", text):
        referenced.add(match.group(1))
    if outside := referenced - set(ALLOWED_ENVS):
        found.append(f"values from outside the compose that are not credentials: {sorted(outside)}")
    if missing := set(ALLOWED_ENVS) - referenced:
        found.append(f"allowed credentials the compose never uses: {sorted(missing)}")
    if "TASK_API_KEY_URI" in text:
        found.append("TASK_API_KEY_URI: the key must be made inside the CVM")
    for pattern, what in FORBIDDEN:
        if pattern.search(text):
            found.append(f"forbidden: {what}")
    blocks = services(text)
    for name, block in blocks.items():
        for published in items(block, "ports"):
            if name != "tls" or published.split(":")[0] != str(TLS_PORT):
                found.append(f"service {name} publishes {published}; only tls may publish {TLS_PORT}")
        for volume in items(block, "volumes"):
            if volume.startswith("/") and not (name == "api" and volume == f"{DSTACK_SOCKET}:{DSTACK_SOCKET}"):
                found.append(f"service {name} mounts the host path {volume}; only the dstack socket into api")
        if DSTACK_SOCKET in block and name != "api":
            found.append(f"service {name} mounts the dstack socket; only api may")
    if "api" not in blocks or f"image: {API_IMAGE}@" not in blocks["api"]:
        found.append(f"the api service must run {API_IMAGE}@sha256:...")
    return found


def app_compose(text: str) -> dict:
    return {
        "manifest_version": 2,
        "name": NAME,
        "runner": "docker-compose",
        "docker_compose_file": text,
        "kms_enabled": True,
        "key_provider": "kms",
        "key_provider_id": "",
        "local_key_provider_enabled": False,
        "gateway_enabled": True,
        "public_logs": True,
        "public_sysinfo": True,
        "public_tcbinfo": True,
        "allowed_envs": list(ALLOWED_ENVS),
        "no_instance_id": False,
        "secure_time": True,
        # The gateway forwards only the TLS port and sends PROXY protocol to it, so the API sees who calls.
        "port_policy": {"restrict_mode": True, "ports": [{"port": TLS_PORT, "pp": True}]},
    }


def encode(compose: dict) -> bytes:
    return json.dumps(compose, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compose", type=Path, default=HERE / "docker-compose.yml")
    parser.add_argument("--image-digest", help="the API image digest CI printed, sha256:<64 hex>")
    parser.add_argument("--out", type=Path, default=HERE / "app-compose.json")
    parser.add_argument("--check", action="store_true", help="only check the compose; placeholders allowed")
    args = parser.parse_args()

    text = args.compose.read_text(encoding="utf-8")
    if args.image_digest:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
            parser.error("--image-digest must be sha256: and 64 lowercase hex digits")
        text = text.replace(IMAGE_PLACEHOLDER, args.image_digest)
    found = problems(text, final=not args.check)
    for problem in found:
        print(f"refused: {problem}", file=sys.stderr)
    if found:
        return 1
    if args.check:
        print("the compose passes the checks", file=sys.stderr)
        return 0
    encoded = encode(app_compose(text))
    args.out.write_bytes(encoded)
    print(hashlib.sha256(encoded).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
