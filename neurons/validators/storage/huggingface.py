import asyncio
from io import BytesIO
from pathlib import Path

from neurons.validators.storage.base import ObjectStorage


MISSING_HF_EXCEPTIONS = {
    "EntryNotFoundError",
    "LocalEntryNotFoundError",
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
}


class HuggingFaceObjectStorage(ObjectStorage):
    def __init__(
        self,
        *,
        namespace: str,
        token: str | None = None,
        repo_type: str = "dataset",
    ):
        from huggingface_hub import HfApi

        self.namespace = namespace
        self.token = token
        self.repo_type = repo_type
        self.api = HfApi(token=token)
        self._ensured_repos: set[str] = set()

    @property
    def provider_name(self) -> str:
        return "hf"

    def _ensure_repo(self, repo_id: str) -> None:
        if repo_id in self._ensured_repos:
            return

        self.api.create_repo(
            repo_id=repo_id,
            repo_type=self.repo_type,
            private=False,
            exist_ok=True,
            token=self.token,
        )
        self._ensured_repos.add(repo_id)

    async def put_object(
        self,
        *,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        del content_type

        repo_id = bucket
        await asyncio.to_thread(self._ensure_repo, repo_id)

        await asyncio.to_thread(
            self.api.upload_file,
            path_or_fileobj=BytesIO(data),
            path_in_repo=key,
            repo_id=repo_id,
            repo_type=self.repo_type,
            commit_message=f"Upload {key}",
            token=self.token,
        )

    async def get_object(self, *, bucket: str, key: str) -> bytes | None:
        from huggingface_hub import hf_hub_download

        repo_id = bucket

        try:
            file_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=key,
                repo_type=self.repo_type,
                token=self.token,
                force_download=True,
            )
        except Exception as exc:
            if exc.__class__.__name__ in MISSING_HF_EXCEPTIONS:
                return None
            raise

        return await asyncio.to_thread(Path(file_path).read_bytes)

    def build_bucket_locator(self, *, bucket: str) -> str:
        return f"{self.provider_name}:{self.namespace}/{bucket}"

    def resolve_bucket_locator(self, locator: str) -> str:
        prefix = f"{self.provider_name}:"
        if not locator.startswith(prefix):
            raise ValueError(f"Unsupported Hugging Face bucket locator: {locator}")
        repo_id = locator[len(prefix) :]
        if "/" not in repo_id:
            raise ValueError(f"Invalid Hugging Face repo locator: {locator}")
        return repo_id
