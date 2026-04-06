import asyncio
from pathlib import Path

from neurons.validators.storage.base import ObjectStorage


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    @property
    def provider_name(self) -> str:
        return "local"

    def _resolve_path(self, bucket: str, key: str) -> Path:
        root_path = self.root_path.resolve()
        file_path = (root_path / bucket / key).resolve()
        if root_path != file_path and root_path not in file_path.parents:
            raise ValueError(f"Storage path escapes local root: {file_path}")
        return file_path

    async def put_object(
        self,
        *,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        del content_type

        file_path = self._resolve_path(bucket, key)
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(file_path.write_bytes, data)

    async def get_object(self, *, bucket: str, key: str) -> bytes | None:
        file_path = self._resolve_path(bucket, key)
        if not file_path.exists():
            return None

        return await asyncio.to_thread(file_path.read_bytes)

    def build_bucket_locator(self, *, bucket: str) -> str:
        return f"{self.provider_name}:{bucket}"

    def resolve_bucket_locator(self, locator: str) -> str:
        prefix = f"{self.provider_name}:"
        if not locator.startswith(prefix):
            raise ValueError(f"Unsupported local bucket locator: {locator}")
        return locator[len(prefix) :]
