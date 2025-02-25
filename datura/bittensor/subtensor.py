import bittensor as bt
import random
from typing import Optional
from .metagraph import Metagraph


class Subtensor(bt.subtensor):
    def __init__(self, **params):
        try:
            super().__init__(**params)
        except:
            pass

    def metagraph(self, netuid: int, lite: bool = True, block: Optional[int] = None):
        metagraph = Metagraph(
            network=self.chain_endpoint,
            netuid=netuid,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    def get_current_block(self):
        return 1000 + random.randint(0, 200)

    def tempo(self, netuid: int, block: Optional[int] = None):
        return 200

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ):
        return 0
