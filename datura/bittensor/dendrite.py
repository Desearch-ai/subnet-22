import bittensor as bt
from neurons.miners.twitter_search_miner import TwitterSearchMiner
from neurons.miners.web_search_miner import WebSearchMiner
from datura.protocol import (
    TwitterURLsSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
    IsAlive,
)
from bittensor_wallet import Wallet


class Dendrite(bt.dendrite):
    def __init__(self, wallet=None):
        try:
            super().__init__(wallet)
        except:
            self.keypair = (
                wallet.hotkey if isinstance(wallet, Wallet) else wallet
            ) or Wallet().hotkey

            self.synapse_history: list = []

            self._session = None

        self.twitter_search_miner = TwitterSearchMiner(None)
        self.web_search_miner = WebSearchMiner(None)

    async def call(self, target_axon, synapse, timeout=12, deserialize=True):
        if isinstance(synapse, TwitterURLsSearchSynapse):
            print("MockDendrite--call twitter_search_miner.search_by_urls")
            return await self.twitter_search_miner.search_by_urls(synapse)

        if isinstance(synapse, TwitterIDSearchSynapse):
            print("MockDendrite--call twitter_search_miner.search_by_id")
            return await self.twitter_search_miner.search_by_id(synapse)

        if isinstance(synapse, TwitterSearchSynapse):
            print("MockDendrite--call twitter_search_miner.search")
            return await self.twitter_search_miner.search(synapse)

        if isinstance(synapse, WebSearchSynapse):
            print("MockDendrite--call web_search_miner.search")
            return await self.web_search_miner.search(synapse)

        if isinstance(synapse, IsAlive):
            print("MockDendrite--call is_alive")
            if target_axon.hotkey.startswith("hotkey"):
                synapse.completion = "True"
                synapse.dendrite.status_code = 200
            return synapse

        print("MockDendrite--call with super(), synapse=", synapse)
        return await super().call(target_axon, synapse, timeout, deserialize)
