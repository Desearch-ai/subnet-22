import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re


BUCKET_COMPONENT_PATTERN = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class StorageObject:
    bucket: str
    key: str


class ObjectStorage(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def put_object(
        self,
        *,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_object(self, *, bucket: str, key: str) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    def build_bucket_locator(self, *, bucket: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def resolve_bucket_locator(self, locator: str) -> str:
        raise NotImplementedError


def sanitize_bucket_component(value: str, *, max_length: int = 24) -> str:
    cleaned = BUCKET_COMPONENT_PATTERN.sub("-", value.lower()).strip("-")
    if not cleaned:
        return "unknown"
    return cleaned[:max_length]


def build_validator_bucket_name(*, netuid: int, validator_uid: int, hotkey: str) -> str:
    hotkey_prefix = sanitize_bucket_component(hotkey[:8])
    hotkey_hash = hashlib.sha256(hotkey.encode("utf-8")).hexdigest()[:12]
    return (
        f"sn{int(netuid)}-validator-{int(validator_uid)}-"
        f"{hotkey_prefix}-{hotkey_hash}"
    )
