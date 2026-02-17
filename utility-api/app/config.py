import os

from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")

NETUID = int(os.getenv("NETUID", "22"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "finney")
