from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bittensor_wallet import Keypair, Wallet

from desearch.embedding import LOCAL_EMBEDDINGS
from desearch.fetch import FetchSettings

ENV_FILE = Path(__file__).resolve().parent / ".env"
TASK_API = "https://task-api.desearch.ai"


@dataclass(frozen=True)
class Settings(FetchSettings):
    task_api_url: str = TASK_API
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = "~/.bittensor/wallets"
    max_tasks: int = 4
    idle_exit: int = 0
    receipts_file: str = ""
    shutdown_grace: float = 45.0
    scrapingdog_api_key: str = ""
    scrapingdog_concurrency: int = 8
    extraction_threads: int = 4
    embed_model: str = "qwen3-embedding-8b"
    embed_api_url: str = LOCAL_EMBEDDINGS
    embed_api_key: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def get(name: str, default):
            value = env.get(name, "").strip()
            return type(default)(value) if value else default

        proxies = env.get("PROXY_URLS", "")
        return cls(
            task_api_url=get("TASK_API_URL", cls.task_api_url),
            wallet_name=get("WALLET_NAME", cls.wallet_name),
            wallet_hotkey=get("WALLET_HOTKEY", cls.wallet_hotkey),
            wallet_path=get("WALLET_PATH", cls.wallet_path),
            proxy_urls=tuple(p.strip() for p in proxies.split(",") if p.strip()),
            concurrency=get("CRAWL_CONCURRENCY", cls.concurrency),
            per_domain=get("CRAWL_CONCURRENCY_PER_DOMAIN", cls.per_domain),
            timeout=get("CRAWL_TIMEOUT", cls.timeout),
            user_agent=get("CRAWL_USER_AGENT", cls.user_agent),
            max_tasks=get("MAX_TASKS", cls.max_tasks),
            receipts_file=get("RECEIPTS_FILE", cls.receipts_file),
            scrapingdog_api_key=get("SCRAPINGDOG_API_KEY", cls.scrapingdog_api_key),
            scrapingdog_concurrency=get(
                "SCRAPINGDOG_CONCURRENCY", cls.scrapingdog_concurrency
            ),
            extraction_threads=get("EXTRACTION_THREADS", cls.extraction_threads),
            embed_model=get("EMBED_MODEL", cls.embed_model),
            embed_api_url=get("EMBED_API_URL", cls.embed_api_url),
            embed_api_key=get("EMBED_API_KEY", cls.embed_api_key),
        )

    def keypair(self) -> Keypair:
        wallet = Wallet(
            name=self.wallet_name, hotkey=self.wallet_hotkey, path=self.wallet_path
        )
        return wallet.hotkey
