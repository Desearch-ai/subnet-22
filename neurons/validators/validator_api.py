import itertools
from typing import Optional, Tuple

import bittensor as bt

from desearch.redis.redis_client import close_redis, initialize_redis
from neurons.validators.advanced_scraper_validator import AdvancedScraperValidator
from neurons.validators.basic_scraper_validator import BasicScraperValidator
from neurons.validators.basic_web_scraper_validator import BasicWebScraperValidator
from neurons.validators.validator_service_client import ValidatorServiceClient


class ValidatorAPI:
    """
    Validator API proxies organic requests to the appropriate scraper validators from API routes.
    Uses validator service to get random miner UID.
    """

    config: bt.Config
    dendrite_list: list[bt.Dendrite]
    dendrites: itertools.cycle
    advanced_scraper_validator: "AdvancedScraperValidator"
    x_scraper_validator: "BasicScraperValidator"
    web_scraper_validator: "BasicWebScraperValidator"

    def __init__(self, config: bt.Config):
        self.config = config
        bt.logging.set_config(self.config)

        self.advanced_scraper_validator = AdvancedScraperValidator(neuron=self)
        self.x_scraper_validator = BasicScraperValidator(neuron=self)
        self.web_scraper_validator = BasicWebScraperValidator(neuron=self)

        self.validator_service_client = ValidatorServiceClient()

    async def initialize(self):
        if self.config.neuron.offline:
            from desearch.bittensor.dendrite import Dendrite
            from desearch.bittensor.wallet import Wallet

            wallet = Wallet(config=self.config)

            self.dendrite_list = [
                Dendrite(wallet=wallet),
                Dendrite(wallet=wallet),
                Dendrite(wallet=wallet),
            ]
        else:
            wallet = bt.Wallet(config=self.config)

            self.dendrite_list = [
                bt.Dendrite(wallet=wallet),
                bt.Dendrite(wallet=wallet),
                bt.Dendrite(wallet=wallet),
            ]

        self.dendrites = itertools.cycle(self.dendrite_list)

        await initialize_redis()

    async def get_random_miner(
        self, uid: Optional[int] = None
    ) -> Tuple[int, bt.AxonInfo]:
        if uid is not None:
            return uid, self.metagraph.axons[uid]

        return await self.validator_service_client.get_random_miner()

    async def start(self):
        bt.logging.info("Starting ValidatorAPI")
        await self.initialize()

    async def stop(self):
        bt.logging.info("Stopping ValidatorAPI")

        await close_redis()

        for dendrite in self.dendrite_list:
            dendrite.close_session()
