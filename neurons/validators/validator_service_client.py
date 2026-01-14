from typing import Optional

import aiohttp
import bittensor as bt

from neurons.validators.env import VALIDATOR_SERVICE_PORT

VALIDATOR_SERVICE_URL = f"http://localhost:{VALIDATOR_SERVICE_PORT}"


class ValidatorServiceClient:
    def __init__(self):
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    @property
    async def session(self):
        """Get or create the session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def get_random_miner(self, uid: Optional[int] = None):
        """Fetch a random miner UID and axon."""
        session = await self.session

        async with session.post(
            f"{VALIDATOR_SERVICE_URL}/uid/random", json={"uid": uid}
        ) as response:
            if response.status == 200:
                data = await response.json()
                uid = data["uid"]
                axon = data["axon"]
                return uid, bt.AxonInfo.from_dict(axon)
            else:
                raise Exception(f"Failed to fetch UID: {response.status}")

    async def get_config(self):
        session = await self.session
        async with session.get(f"{VALIDATOR_SERVICE_URL}/config") as response:
            if response.status == 200:
                config_dict = await response.json()
                config = bt.Config()
                return config.fromDict(config_dict)
            else:
                raise Exception(f"Failed to fetch config: {response.status}")

    async def health_check(self):
        session = await self.session
        async with session.get(f"{VALIDATOR_SERVICE_URL}") as response:
            if response.status != 200:
                raise Exception(f"Health check failed: {response.status}")
