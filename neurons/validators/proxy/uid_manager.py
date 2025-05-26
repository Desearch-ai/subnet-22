from typing import List
import random
import bittensor as bt

from neurons.validators.weights import EMISSION_CONTROL_HOTKEY


class UIDManager:
    """
    UID manager class that chooses random miner UID from top miners
    UIDs are updated on metagraph resync
    """

    def __init__(
        self,
        wallet: bt.wallet,
        metagraph: bt.metagraph,
    ) -> None:
        self.wallet = wallet
        self.metagraph = metagraph
        self.max_miners_to_use = 120
        self.uids = []
        self.available_uids = []

    def resync(self, available_uids: List[int]):
        """
        Resync the state after metagraph resync
        """
        if not len(available_uids):
            return

        if EMISSION_CONTROL_HOTKEY:
            # Exclude emission control miner from organic requests
            emission_control_uid = next(
                (
                    neuron.uid
                    for neuron in self.metagraph.neurons
                    if neuron.hotkey == EMISSION_CONTROL_HOTKEY
                ),
                None,
            )

            available_uids = [
                uid for uid in available_uids if uid != emission_control_uid
            ]

        self.available_uids = available_uids

        self.top_uids = self.metagraph.I.argsort(descending=True)[
            : self.max_miners_to_use
        ]

        # Reuse uids from previous cycle if they are still in top 200 and available
        if len(self.uids):
            self.uids = [uid for uid in self.uids if uid in available_uids]

        # If no uids are
        if not len(self.uids):
            self.uids = [
                uid_tensor.item()
                for uid_tensor in self.top_uids
                if uid_tensor.item() in available_uids
            ]

    def get_miner_uid(self):
        """
        Get random miner UID from top 200 miners and remove it from the list
        """
        if len(self.uids) == 0:
            self.resync(self.available_uids)

        uid = random.choice(self.uids)
        self.uids.remove(uid)
        return uid
