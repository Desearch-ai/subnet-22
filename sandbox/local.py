"""Redis, storage and the task API on this machine, for a miner to run against."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = "redis:7-alpine"
MINIO_IMAGE = "quay.io/minio/minio:latest"
MINIO_USER = "sandbox"
MINIO_PASSWORD = "sandbox-secret"
UPLOADS_BUCKET = "sandbox-uploads"
PAGES_BUCKET = "sandbox-pages"
REDIS = "sandbox-redis"
MINIO = "sandbox-minio"
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
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api}"

    @property
    def storage_url(self) -> str:
        return f"http://127.0.0.1:{self.minio}/{UPLOADS_BUCKET}"


def start_containers(ports: Ports, run_dir: Path) -> None:
    stop_containers()
    (run_dir / "minio").mkdir(parents=True, exist_ok=True)
    docker_run(
        REDIS,
        "256m",
        f"127.0.0.1:{ports.redis}:6379",
        [REDIS_IMAGE, "redis-server", "--save", "", "--appendonly", "no"],
    )
    docker_run(
        MINIO,
        "512m",
        f"127.0.0.1:{ports.minio}:9000",
        [
            "-e",
            f"MINIO_ROOT_USER={MINIO_USER}",
            "-e",
            f"MINIO_ROOT_PASSWORD={MINIO_PASSWORD}",
            "-v",
            f"{run_dir / 'minio'}:/data",
            MINIO_IMAGE,
            "server",
            "/data",
            "--quiet",
        ],
    )


def docker_run(name: str, memory: str, port: str, args: list[str]) -> None:
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
            "-p",
            port,
            *args,
        ],
        check=True,
        capture_output=True,
    )


def stop_containers() -> None:
    subprocess.run(["docker", "rm", "-f", REDIS, MINIO], capture_output=True)


def storage_env(ports: Ports) -> dict[str, str]:
    return {
        "CF_R2_ENDPOINT": f"http://127.0.0.1:{ports.minio}",
        "CF_R2_ACCESS_KEY_ID": MINIO_USER,
        "CF_R2_SECRET_ACCESS_KEY": MINIO_PASSWORD,
        "CF_R2_BUCKET": UPLOADS_BUCKET,
        "CF_R2_PAGES_BUCKET": PAGES_BUCKET,
    }


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
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{ports.minio}",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASSWORD,
        region_name="auto",
    )
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for name in (UPLOADS_BUCKET, PAGES_BUCKET):
        if name not in existing:
            s3.create_bucket(Bucket=name)
    s3.put_bucket_policy(Bucket=UPLOADS_BUCKET, Policy=public_read(UPLOADS_BUCKET))


def api_env(ports: Ports, run_dir: Path, genesis: float) -> dict[str, str]:
    data = run_dir / "api"
    data.mkdir(parents=True, exist_ok=True)
    return {
        **{
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("CF_R2_", "TASK_API_", "SCRAPINGDOG"))
        },
        **storage_env(ports),
        "TASK_API_REGISTRY": "local",
        "TASK_API_SEEDS": "local",
        "TASK_API_VALIDATOR_URIS": VALIDATOR_URI,
        "TASK_API_ADMIN_URIS": ADMIN_URI,
        "TASK_API_BLOCK_SECONDS": str(BLOCK_SECONDS),
        "TASK_API_GENESIS": str(genesis),
        "TASK_API_DATA": str(data),
        "TASK_API_REDIS": f"redis://127.0.0.1:{ports.redis}/0",
        "TASK_API_READS_PER_MINUTE": "6000",
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
    }


def api_command(python: str, ports: Ports, task_size: int) -> list[str]:
    """The task API, with its task size set for this sandbox only."""
    start = (
        "import uvicorn, app.rounds as rounds;"
        f"rounds.TASK_URLS = {int(task_size)};"
        f"uvicorn.run('app.main:app', host='127.0.0.1', port={ports.api},"
        " log_level='warning')"
    )
    return [python, "-c", start]
