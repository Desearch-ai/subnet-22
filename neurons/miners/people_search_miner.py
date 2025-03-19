import os
import bittensor as bt
from datura.protocol import PeopleSearchSynapse, PeopleSearchResult
from datura.tools.search.serp_api_wrapper import SerpAPIWrapper


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )


class PeopleSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.serp_api_wrapper = SerpAPIWrapper(SERPAPI_API_KEY)

    async def search(self, synapse: PeopleSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query

        # Log the mock search execution
        bt.logging.info(f"Executing people search with query: {query}")

        synapse.results = []

        return synapse
