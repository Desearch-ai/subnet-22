import os

from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "development")

DB_URL = os.getenv("DB_URL")
DB_POOL_RECYCLE_SECONDS = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))

NETUID = int(os.getenv("NETUID", "22"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "finney")


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


# Comma-separated list of validator public-API URLs, e.g.
# VALIDATOR_URLS="http://1.2.3.4:8005,http://5.6.7.8:8005"
VALIDATOR_URLS = _parse_csv(os.getenv("VALIDATOR_URLS", ""))

# Comma-separated CORS origins. Override locally to test from the dev frontend.
CORS_ALLOWED_ORIGINS = _parse_csv(
    os.getenv("CORS_ALLOWED_ORIGINS", "https://mining.desearch.ai")
)
