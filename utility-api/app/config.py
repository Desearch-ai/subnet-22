import os

from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "development")

DB_URL = os.getenv("DB_URL")
DB_POOL_RECYCLE_SECONDS = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))

NETUID = int(os.getenv("NETUID", "22"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "finney")
