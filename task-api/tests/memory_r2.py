from __future__ import annotations

import hashlib
import io
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from app.storage import Storage
from botocore.exceptions import ClientError

from tests.http_client import HttpClient

DOTENV = Path(__file__).resolve().parents[1] / ".env"
R2_KEYS = ("CF_R2_ENDPOINT", "CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY")


def _error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


class MemoryR2:
    def __init__(self):
        self.objects: dict[tuple[str, str], dict] = {}
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        if self.server is None:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), _presigned(self))
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def head_bucket(self, Bucket):
        return {}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        query = urlencode({"op": operation, "ct": Params.get("ContentType", "")})
        return f"{self.url}/{Params['Bucket']}/{quote(Params['Key'])}?{query}"

    def put_object(
        self,
        Bucket,
        Key,
        Body,
        ContentType="",
        Metadata=None,
        IfMatch=None,
        IfNoneMatch=None,
        CacheControl=None,
    ):
        current = self.objects.get((Bucket, Key))
        if (IfNoneMatch == "*" and current) or (
            IfMatch is not None and (not current or current["etag"] != IfMatch)
        ):
            raise _error("PreconditionFailed", 412, "PutObject")
        return {"ETag": self._store(Bucket, Key, bytes(Body), ContentType, Metadata)}

    def get_object(self, Bucket, Key, IfMatch=None):
        found = self._found(Bucket, Key, "GetObject")
        if IfMatch and found["etag"] != IfMatch:
            raise _error("PreconditionFailed", 412, "GetObject")
        return {
            "Body": io.BytesIO(found["body"]),
            "ETag": found["etag"],
            "Metadata": found["metadata"],
        }

    def head_object(self, Bucket, Key):
        found = self.objects.get((Bucket, Key))
        if found is None:
            raise _error("404", 404, "HeadObject")
        return {
            "ContentLength": len(found["body"]),
            "ETag": found["etag"],
            "Metadata": found["metadata"],
        }

    def copy_object(self, Bucket, Key, CopySource, CopySourceIfMatch=None):
        source = self._found(CopySource["Bucket"], CopySource["Key"], "CopyObject")
        if CopySourceIfMatch and source["etag"] != CopySourceIfMatch:
            raise _error("PreconditionFailed", 412, "CopyObject")
        etag = self._store(
            Bucket, Key, source["body"], source["type"], source["metadata"]
        )
        return {"CopyObjectResult": {"ETag": etag}}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)

    def list_objects_v2(self, Bucket, Prefix=""):
        keys = sorted(
            k for b, k in self.objects if b == Bucket and k.startswith(Prefix)
        )
        return {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}

    def _found(self, bucket: str, key: str, operation: str) -> dict:
        found = self.objects.get((bucket, key))
        if found is None:
            raise _error("NoSuchKey", 404, operation)
        return found

    def _store(self, bucket, key, body, content_type, metadata) -> str:
        etag = f'"{hashlib.md5(body).hexdigest()}"'
        self.objects[(bucket, key)] = {
            "body": body,
            "etag": etag,
            "type": content_type,
            "metadata": dict(metadata or {}),
        }
        return etag


class Backend:
    def __init__(self, real: bool):
        self.real = real
        self.memory = None if real else MemoryR2()
        self.prefix = f"_test/{uuid.uuid4().hex[:12]}/"
        self.made: list[Storage] = []

    def storage(self, bucket: str | None = None, prefix: str = "") -> Storage:
        storage = Storage(bucket=bucket, prefix=self.prefix + prefix)
        if self.memory is not None:
            storage.client = self.memory
        self.made.append(storage)
        return storage

    def http(self) -> HttpClient:
        return HttpClient()

    def cleanup(self) -> None:
        for storage in self.made:
            purge(storage)
        if self.memory is not None:
            self.memory.close()


def _presigned(r2: MemoryR2) -> type[BaseHTTPRequestHandler]:
    """A PUT must carry the signed content type, as on R2."""

    class Presigned(BaseHTTPRequestHandler):
        def do_PUT(self):
            bucket, key, query = self._target()
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if query.get("op") != "put_object":
                return self._reply(403)
            if self.headers.get("content-type") != query.get("ct"):
                return self._reply(403, b"SignatureDoesNotMatch")
            r2._store(bucket, key, body, query["ct"], {})
            self._reply(200)

        def do_GET(self):
            bucket, key, query = self._target()
            if query.get("op", "get_object") != "get_object":
                return self._reply(403)
            found = r2.objects.get((bucket, key))
            if found is None:
                return self._reply(404, b"NoSuchKey")
            self._reply(200, found["body"])

        def _target(self) -> tuple[str, str, dict[str, str]]:
            parts = urlsplit(self.path)
            bucket, _, key = unquote(parts.path).lstrip("/").partition("/")
            return bucket, key, {k: v[0] for k, v in parse_qs(parts.query).items()}

        def _reply(self, status: int, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_) -> None:
            pass

    return Presigned


def purge(storage: Storage) -> None:
    assert storage.prefix.startswith("_test/"), storage.prefix
    client, bucket = storage.client, storage.bucket
    listed = client.list_objects_v2(Bucket=bucket, Prefix=storage.prefix)
    keys = [{"Key": item["Key"]} for item in listed.get("Contents", [])]
    if keys:
        client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
    remaining = client.list_objects_v2(Bucket=bucket, Prefix=storage.prefix)
    assert remaining.get("KeyCount", 0) == 0, remaining.get("Contents")
