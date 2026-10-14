"""Redis, storage and the task API on this machine, for a miner to run against."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = "redis:7-alpine"
MINIO_IMAGE = "quay.io/minio/minio:latest"
API_IMAGE = "desearch-sandbox-task-api"
API_DOCKERFILE = "task-api/Dockerfile.rust"
API_SOURCES = (
    "crates/desearch",
    "task-api/Cargo.toml",
    "task-api/Cargo.lock",
    "task-api/src",
)
MINIO_USER = "sandbox"
MINIO_PASSWORD = "sandbox-secret"
UPLOADS_BUCKET = "sandbox-uploads"
PAGES_BUCKET = "sandbox-pages"
REDIS = "sandbox-redis"
MINIO = "sandbox-minio"
API = "sandbox-api"
CONTAINERS = (API, REDIS, MINIO)
ENV_FILES = (
    ROOT / "neurons" / "validators" / ".env",
    ROOT / "neurons" / "miners" / ".env",
)
VALIDATOR_URI = "//sandbox/validator"
ADMIN_URI = "//sandbox/admin"
BLOCK_SECONDS = 1.0
TEMPO = 60


@dataclass(frozen=True)
class Ports:
    api: int = 18080

    @property
    def redis(self) -> int:
        return self.api + 1

    @property
    def minio(self) -> int:
        return self.api + 2

    @property
    def reference(self) -> int:
        return self.api + 3

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api}"

    @property
    def storage_url(self) -> str:
        return f"http://127.0.0.1:{self.minio}/{UPLOADS_BUCKET}"

    @property
    def reference_url(self) -> str:
        return f"http://127.0.0.1:{self.reference}/v1/embeddings"


def source_files(root: Path = ROOT) -> list[Path]:
    """The files the task API image is built from, as its Dockerfile copies them."""
    found = []
    for source in API_SOURCES:
        path = root / source
        if path.is_file():
            found.append(path)
            continue
        found += [
            file
            for file in path.rglob("*")
            if file.is_file()
            and "target" not in file.relative_to(path).parts
            and file.name != ".DS_Store"
        ]
    return sorted(found)


def image_tag(root: Path = ROOT) -> str:
    """The image named for its sources, so a change to them builds a new one."""
    digest = hashlib.sha256()
    for file in source_files(root):
        digest.update(file.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return f"{API_IMAGE}:{digest.hexdigest()[:12]}"


def image_exists(tag: str) -> bool:
    found = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    return found.returncode == 0


def build_image() -> str:
    """The task API's image, built from this checkout when there is none for its sources."""
    tag = image_tag()
    if image_exists(tag):
        return tag
    print(
        f"Building the task API image {tag} with Docker. The first build compiles it"
        " from source and takes several minutes; later starts reuse it.",
        flush=True,
    )
    subprocess.run(
        ["docker", "build", "-f", API_DOCKERFILE, "-t", tag, "."],
        cwd=ROOT,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
        check=True,
    )
    return tag


def start_storage(ports: Ports, run_dir: Path) -> None:
    """MinIO holds the network every container shares, so 127.0.0.1 means the same inside and out."""
    stop_containers()
    (run_dir / "minio").mkdir(parents=True, exist_ok=True)
    published = [
        f"127.0.0.1:{port}:{port}" for port in (ports.api, ports.redis, ports.minio)
    ]
    docker_run(
        MINIO,
        "512m",
        [
            *[flag for port in published for flag in ("-p", port)],
            "-e",
            f"MINIO_ROOT_USER={MINIO_USER}",
            "-e",
            f"MINIO_ROOT_PASSWORD={MINIO_PASSWORD}",
            "-v",
            f"{run_dir / 'minio'}:/data",
            MINIO_IMAGE,
            "server",
            "/data",
            "--address",
            f":{ports.minio}",
            "--quiet",
        ],
    )
    docker_run(
        REDIS,
        "256m",
        [
            "--network",
            f"container:{MINIO}",
            REDIS_IMAGE,
            "redis-server",
            "--port",
            str(ports.redis),
            "--protected-mode",
            "no",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
    )


def start_api(image: str, env: dict[str, str], data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    docker_run(
        API,
        "1g",
        [
            "--network",
            f"container:{MINIO}",
            # As the host user, so the API can write its data folder on a Linux host.
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{data}:/data",
            *[
                flag
                for name, value in env.items()
                for flag in ("-e", f"{name}={value}")
            ],
            image,
        ],
    )


def docker_run(name: str, memory: str, args: list[str]) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--memory",
            memory,
            "--memory-swap",
            memory,
            *args,
        ],
        check=True,
        capture_output=True,
    )


def follow_logs(name: str) -> list[str]:
    return ["docker", "logs", "-f", name]


def stop_containers() -> None:
    subprocess.run(["docker", "rm", "-f", *CONTAINERS], capture_output=True)


def redis_ready(ports: Ports) -> bool:
    found = subprocess.run(
        ["docker", "exec", REDIS, "redis-cli", "-p", str(ports.redis), "ping"],
        capture_output=True,
        text=True,
    )
    return found.stdout.strip() == "PONG"


def push_redis(ports: Ports, key: str, value: str) -> None:
    """RPUSH with the value on stdin: an input entry of 1,000 pages is too long for an argument."""
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            REDIS,
            "redis-cli",
            "-p",
            str(ports.redis),
            "-x",
            "RPUSH",
            key,
        ],
        input=value.encode(),
        check=True,
        capture_output=True,
    )


def storage_env(ports: Ports) -> dict[str, str]:
    return {
        "CF_R2_ENDPOINT": f"http://127.0.0.1:{ports.minio}",
        "CF_R2_ACCESS_KEY_ID": MINIO_USER,
        "CF_R2_SECRET_ACCESS_KEY": MINIO_PASSWORD,
        "CF_R2_BUCKET": UPLOADS_BUCKET,
        "CF_R2_PAGES_BUCKET": PAGES_BUCKET,
    }


def s3_client(ports: Ports):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{ports.minio}",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASSWORD,
        region_name="auto",
    )


def public_read(bucket: str) -> str:
    """Anyone may read objects, as on the public bucket validators read on mainnet."""
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket}/*"],
                }
            ],
        }
    )


def create_buckets(ports: Ports) -> None:
    s3 = s3_client(ports)
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for name in (UPLOADS_BUCKET, PAGES_BUCKET):
        if name not in existing:
            s3.create_bucket(Bucket=name)
    s3.put_bucket_policy(Bucket=UPLOADS_BUCKET, Policy=public_read(UPLOADS_BUCKET))


def api_env(
    ports: Ports, genesis: float, task_size: int, embed: bool = False
) -> dict[str, str]:
    """The task API's settings inside its container, where its data folder is /data."""
    return {
        **storage_env(ports),
        "TASK_API_REGISTRY": "local",
        "TASK_API_SEEDS": "local",
        "TASK_API_VALIDATOR_URIS": VALIDATOR_URI,
        "TASK_API_ADMIN_URIS": ADMIN_URI,
        "TASK_API_BLOCK_SECONDS": str(BLOCK_SECONDS),
        "TASK_API_GENESIS": str(genesis),
        "TASK_API_DATA": "/data",
        "TASK_API_REDIS": f"redis://127.0.0.1:{ports.redis}/0",
        "TASK_API_LISTEN": f"0.0.0.0:{ports.api}",
        "TASK_API_TASK_URLS": str(int(task_size)),
        "TASK_API_READS_PER_MINUTE": "6000",
        "TASK_API_LOG_READS_PER_MINUTE": "6000",
        # Every upload is checked, so a miner sees each verdict.
        "TASK_API_CHECK_SHARE": "1",
        # No publisher runs here to take the passed work, so its backlog never closes the queue.
        "TASK_API_MAX_BACKLOG_S": "0",
        "TASK_API_EMBED_TASKS": "1" if embed else "0",
    }
