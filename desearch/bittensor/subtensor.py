import bittensor as bt
import random
from typing import Optional
from .metagraph import Metagraph


class Subtensor(bt.AsyncSubtensor):
    def __init__(self, **params):
        try:
            super().__init__(**params)
        except:
            pass

    async def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ):
        metagraph = Metagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        await metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    async def get_current_block(self):
        return 1000 + random.randint(0, 200)

    async def tempo(self, netuid: int, block: Optional[int] = None):
        return 200

    async def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ):
        return 0

    async def is_hotkey_registered(self, netuid, hotkey_ss58):
        return True
