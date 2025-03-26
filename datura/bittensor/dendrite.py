import bittensor as bt
import time
from datura.protocol import (
    TwitterURLsSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    WebSearchSynapse,
    IsAlive,
    ScraperStreamingSynapse,
)
from bittensor_wallet import Wallet
from .miner import Miner
from bittensor.core.synapse import Synapse
from aiohttp import ClientResponse
from unittest.mock import AsyncMock


class Dendrite(bt.dendrite):
    def __init__(self, wallet=None):
        from neurons.miners.twitter_search_miner import TwitterSearchMiner
        from neurons.miners.web_search_miner import WebSearchMiner
        from neurons.miners.scraper_miner import ScraperMiner

        try:
            super().__init__(wallet)
        except:
            self.keypair = (
                wallet.hotkey if isinstance(wallet, Wallet) else wallet
            ) or Wallet().hotkey

            self.synapse_history: list = []

            self._session = None

        self.miner = Miner()

        self.twitter_search_miner = TwitterSearchMiner(self.miner)
        self.web_search_miner = WebSearchMiner(self.miner)
        self.scraper_miner = ScraperMiner(self.miner)

    async def call(self, target_axon, synapse, timeout=12, deserialize=True):
        start_time = time.time()
        if isinstance(synapse, TwitterURLsSearchSynapse):
            bt.logging.info("MockDendrite--call twitter_search_miner.search_by_urls")
            synapse = await self.twitter_search_miner.search_by_urls(synapse)
            synapse.dendrite.process_time = str(time.time() - start_time)
            return synapse

        if isinstance(synapse, TwitterIDSearchSynapse):
            bt.logging.info("MockDendrite--call twitter_search_miner.search_by_id")
            synapse = await self.twitter_search_miner.search_by_id(synapse)
            synapse.dendrite.process_time = str(time.time() - start_time)
            return synapse

        if isinstance(synapse, TwitterSearchSynapse):
            bt.logging.info("MockDendrite--call twitter_search_miner.search")
            synapse = await self.twitter_search_miner.search(synapse)
            synapse.dendrite.process_time = str(time.time() - start_time)
            return synapse

        if isinstance(synapse, WebSearchSynapse):
            bt.logging.info("MockDendrite--call web_search_miner.search")
            synapse = await self.web_search_miner.search(synapse)
            synapse.dendrite.process_time = str(time.time() - start_time)
            return synapse

        if isinstance(synapse, IsAlive):
            bt.logging.info("MockDendrite--call is_alive")
            if target_axon.hotkey.startswith("hotkey"):
                synapse.completion = "True"
                synapse.dendrite.status_code = 200
            synapse.dendrite.process_time = str(time.time() - start_time)
            return synapse

        bt.logging.info("MockDendrite--call with super(), synapse=", synapse)
        return await super().call(target_axon, synapse, timeout, deserialize)

    async def call_stream(self, target_axon, synapse, timeout=12.0, deserialize=True):
        start_time = time.time()
        if isinstance(synapse, ScraperStreamingSynapse):
            responses = []

            async def mockSend(data):
                responses.append(data["body"])

            async def generateResponse():
                for data in responses:
                    yield data

            await self.scraper_miner.smart_scraper(synapse, mockSend)

            # Mock ClientResponse
            response = AsyncMock(spec=ClientResponse)
            response.content.iter_any = generateResponse
            response.__dict__["_raw_headers"] = {}
            response.status = 200
            response.headers = {}

            async for chunk in synapse.process_streaming_response(response):  # type: ignore
                yield chunk  # Yield each chunk as it's processed
            json_response = synapse.extract_response_json(response)
            # Process the server response
            self.process_server_response(response, json_response, synapse)
            synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

            self._log_incoming_response(synapse)

            # Log synapse event history
            self.synapse_history.append(Synapse.from_headers(synapse.to_headers()))

            # Return the updated synapse object after deserializing if requested
            if deserialize:
                yield synapse.deserialize()
            else:
                yield synapse

            return

        bt.logging.info("MockDendrite--call_stream with super(), synapse=", synapse)
        async for chunk in super().call_stream(
            target_axon, synapse, timeout, deserialize
        ):
            yield chunk
