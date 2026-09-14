from __future__ import annotations

import asyncio
import json
import os
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PARQUET = "application/vnd.apache.parquet"
MISSING = {"404", "NoSuchKey", "NotFound"}
CHANGED = {"412", "PreconditionFailed"}


class Changed(Exception):
    pass


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
        await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)

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

    async def stat(self, key: str) -> tuple[int, str] | None:
        try:
            found = await asyncio.to_thread(
                self.client.head_object, Bucket=self.bucket, Key=self.path(key)
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in MISSING:
                return None
            raise
        return int(found["ContentLength"]), found["ETag"]

    async def copy(
        self, src: str, dst: str, etag: str | None = None, into: Storage | None = None
    ) -> str:
        target = into or self
        extra = {"CopySourceIfMatch": etag} if etag else {}
        try:
            done = await asyncio.to_thread(
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

    async def put_json(self, key: str, obj) -> None:
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=self.path(key),
            Body=json.dumps(obj, sort_keys=True).encode(),
            ContentType="application/json",
        )

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(
            self.client.delete_object, Bucket=self.bucket, Key=self.path(key)
        )
