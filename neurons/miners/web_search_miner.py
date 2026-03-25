import bittensor as bt
from desearch.protocol import WebSearchSynapse, WebSearchResult
from desearch.tools.search.scrapingdog_google_search import ScrapingDogGoogleSearch


class WebSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def search(self, synapse: WebSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query
        start = synapse.start
        num = synapse.num

        # Log the mock search execution
        bt.logging.info(f"Executing web search with query: {query}")

        page = max((start or 0) // max(num, 1), 0)
        search = ScrapingDogGoogleSearch(results=num)
        res = await search.search(query=query, page=page)

        results = []
        for item in res:
            results.append(WebSearchResult(**item).model_dump())

        synapse.results = results

        return synapse
