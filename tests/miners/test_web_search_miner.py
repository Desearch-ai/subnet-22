import unittest
from neurons.miners.web_search_miner import WebSearchMiner
from datura.protocol import WebSearchSynapse


class TestWebSearchMiner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.miner = WebSearchMiner(None)

    async def test_search(self):
        query = "blockchain"
        synapse = WebSearchSynapse(query=query)

        result = await self.miner.search(synapse)

        self.assertEqual(result, synapse)
        self.assertTrue(
            all(
                query in item.title.lower() or query in item.snippet.lower()
                for item in synapse.results
            )
        )

    async def test_search_with_start_and_num(self):
        query = "blockchain"
        synapse = WebSearchSynapse(query=query, num=7, start=1)

        result = await self.miner.search(synapse)

        self.assertEqual(result, synapse)
        self.assertEqual(len(synapse.results), 7)
        self.assertTrue(
            all(
                query in item.title.lower() or query in item.snippet.lower()
                for item in synapse.results
            )
        )


if __name__ == "__main__":
    unittest.main()
