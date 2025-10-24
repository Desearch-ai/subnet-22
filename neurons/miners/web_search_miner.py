import os
import bittensor as bt
from desearch.protocol import WebSearchSynapse, WebSearchResult
from desearch.tools.search.serp_api_wrapper import SerpAPIWrapper


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/Desearch-ai/subnet-22/blob/main/docs/env_variables.md"
    )


class WebSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.serp_api_wrapper = SerpAPIWrapper(SERPAPI_API_KEY)

    async def search(self, synapse: WebSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query
        start = synapse.start
        num = synapse.num

        # Log the mock search execution
        bt.logging.info(f"Executing web search with query: {query}")

        res = await self.serp_api_wrapper.arun(query=query, start=start, num=num)

        results = []
        for item in res.get("organic_results", []):
            results.append(WebSearchResult(**item).model_dump())

        synapse.results = results

        return synapse
