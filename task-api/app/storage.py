from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import cached_property, partial
from typing import NamedTuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PARQUET = "application/vnd.apache.parquet"
PARQUET_MAGIC = b"PAR1"
MISSING = {"404", "NoSuchKey", "NotFound"}
CHANGED = {"412", "PreconditionFailed"}


# Storage calls wait on the network, so they get their own threads and never queue behind other work.
CALLS = ThreadPoolExecutor(64, thread_name_prefix="r2")


class Changed(Exception):
    pass


class Stat(NamedTuple):
    size: int
    etag: str
    modified: datetime | None


async def _call(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(
        CALLS, partial(fn, *args, **kwargs)
    )


class Storage:
    def __init__(self, bucket: str | None = None, prefix: str | None = None):
        self.bucket = bucket or os.environ.get("CF_R2_BUCKET", "subnet-22")
        self.prefix = (
            os.environ.get("TASK_API_R2_PREFIX", "") if prefix is None else prefix
        )

    @cached_property
    def client(self):
        return boto3.session.Session().client(
            "s3",
            endpoint_url=os.environ.get("CF_R2_ENDPOINT"),
            aws_access_key_id=os.environ.get("CF_R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("CF_R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            # Bounded to finish inside the API's scoring grace.
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 3},
                max_pool_connections=64,
            ),
        )

    def path(self, key: str) -> str:
        return self.prefix + key

    async def check(self) -> None:
        await _call(self.client.head_bucket, Bucket=self.bucket)

    def presign_put(self, key: str, content_type: str, expires: int) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": self.path(key),
                "ContentType": content_type,
            },
            ExpiresIn=expires,
        )

    def presign_get(self, key: str, expires: int) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self.path(key)},
            ExpiresIn=expires,
        )

    async def stat(self, key: str) -> Stat | None:
        try:
            found = await _call(
                self.client.head_object, Bucket=self.bucket, Key=self.path(key)
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in MISSING:
                return None
            raise
        return Stat(
            int(found["ContentLength"]), found["ETag"], found.get("LastModified")
        )

    async def copy(
        self, src: str, dst: str, etag: str | None = None, into: Storage | None = None
    ) -> str:
        target = into or self
        extra = {"CopySourceIfMatch": etag} if etag else {}
        try:
            done = await _call(
                target.client.copy_object,
                Bucket=target.bucket,
                Key=target.path(dst),
                CopySource={"Bucket": self.bucket, "Key": self.path(src)},
                **extra,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in CHANGED or (code in MISSING and etag):
                raise Changed(src) from None
            raise
        return done["CopyObjectResult"]["ETag"]

    async def is_parquet(self, key: str, size: int) -> bool:
        """Only the magic bytes at both ends are read; nothing of the miner's is parsed here."""
        if size < 2 * len(PARQUET_MAGIC):
            return False
        head = await self.read_range(key, f"bytes=0-{len(PARQUET_MAGIC) - 1}")
        tail = await self.read_range(key, f"bytes=-{len(PARQUET_MAGIC)}")
        return head == PARQUET_MAGIC and tail == PARQUET_MAGIC

    async def read_range(self, key: str, byte_range: str) -> bytes:
        def read() -> bytes:
            found = self.client.get_object(
                Bucket=self.bucket, Key=self.path(key), Range=byte_range
            )
            return found["Body"].read()

        return await _call(read)

    async def put_json(self, key: str, obj, cache_control: str = "") -> None:
        await _call(
            self.client.put_object,
            Bucket=self.bucket,
            Key=self.path(key),
            Body=json.dumps(obj, sort_keys=True).encode(),
            ContentType="application/json",
            **({"CacheControl": cache_control} if cache_control else {}),
        )

    async def put_bytes(self, key: str, body: bytes, content_type: str) -> None:
        await _call(
            self.client.put_object,
            Bucket=self.bucket,
            Key=self.path(key),
            Body=body,
            ContentType=content_type,
        )

    def read_range_now(self, key: str, start: int, end: int) -> bytes:
        found = self.client.get_object(
            Bucket=self.bucket, Key=self.path(key), Range=f"bytes={start}-{end}"
        )
        return found["Body"].read()

    async def delete(self, key: str) -> None:
        await _call(self.client.delete_object, Bucket=self.bucket, Key=self.path(key))
