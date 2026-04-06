import os
from pathlib import Path

from neurons.validators.storage.base import ObjectStorage
from neurons.validators.storage.huggingface import HuggingFaceObjectStorage
from neurons.validators.storage.local import LocalObjectStorage


DEFAULT_STORAGE_PROVIDER = "local"
DEFAULT_LOCAL_STORAGE_PATH = (
    Path(__file__).resolve().parents[3] / "tmp" / "validator-storage"
)


def build_object_storage() -> ObjectStorage:
    provider = (
        os.environ.get("VALIDATOR_STORAGE_PROVIDER", DEFAULT_STORAGE_PROVIDER)
        .strip()
        .lower()
    )

    if provider == "local":
        root_path = os.environ.get("VALIDATOR_STORAGE_LOCAL_PATH")
        return LocalObjectStorage(root_path or DEFAULT_LOCAL_STORAGE_PATH)

    if provider in {"hf", "huggingface"}:
        namespace = os.environ.get("VALIDATOR_STORAGE_HF_NAMESPACE")
        if not namespace:
            raise RuntimeError(
                "VALIDATOR_STORAGE_HF_NAMESPACE must be set when "
                "VALIDATOR_STORAGE_PROVIDER=huggingface."
            )

        token = (
            os.environ.get("VALIDATOR_STORAGE_HF_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

        return HuggingFaceObjectStorage(namespace=namespace, token=token)

    raise ValueError(f"Unsupported validator storage provider: {provider}")
