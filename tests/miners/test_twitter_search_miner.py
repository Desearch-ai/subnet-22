import unittest
from neurons.miners.twitter_search_miner import TwitterSearchMiner
from desearch.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)


class TestTwitterSearchMiner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.miner = TwitterSearchMiner(None)

    async def test_search_with_query(self):
        query = "blockchain"

        synapse = TwitterSearchSynapse(query=query, count=1)

        synapse_with_results = await self.miner.search(synapse)

        self.assertEqual(len(synapse_with_results.results), 1)
        self.assertIn(query, synapse_with_results.results[0]["text"].lower())

    async def test_search_with_user(self):
        synapse = TwitterSearchSynapse(query="bittensor", user="micoolcho", count=1)

        synapse_with_results = await self.miner.search(synapse)

        self.assertEqual(len(synapse_with_results.results), 1)
        self.assertEqual(
            synapse_with_results.results[0]["user"]["username"], "micoolcho"
        )

    async def test_search_with_blue_verified(self):
        synapse = TwitterSearchSynapse(query="blockchain", blue_verified=True, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(
                result["user"]["is_blue_verified"]
                for result in synapse_with_results.results
            )
        )

    async def test_search_with_is_image(self):
        synapse = TwitterSearchSynapse(query="blockchain", is_image=True, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        for result in synapse_with_results.results:
            self.assertTrue(all(media["type"] == "photo" for media in result["media"]))

    async def test_search_with_is_quote(self):
        synapse = TwitterSearchSynapse(query="blockchain", is_quote=True, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(result["is_quote_tweet"] for result in synapse_with_results.results)
        )

    async def test_search_with_is_video(self):
        synapse = TwitterSearchSynapse(query="blockchain", is_video=True, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        for result in synapse_with_results.results:
            self.assertTrue(all(media["type"] == "video" for media in result["media"]))

    async def test_search_with_lang(self):
        synapse = TwitterSearchSynapse(query="blockchain", lang="en", count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(result["lang"] == "en" for result in synapse_with_results.results)
        )

    async def test_search_with_min_likes(self):
        synapse = TwitterSearchSynapse(query="blockchain", min_likes=100, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(result["like_count"] >= 100 for result in synapse_with_results.results)
        )

    async def test_search_with_min_replies(self):
        synapse = TwitterSearchSynapse(query="blockchain", min_replies=10, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(result["reply_count"] >= 10 for result in synapse_with_results.results)
        )

    async def test_search_with_min_retweets(self):
        synapse = TwitterSearchSynapse(query="blockchain", min_retweets=10, count=5)

        synapse_with_results = await self.miner.search(synapse)

        self.assertLessEqual(len(synapse_with_results.results), 5)
        self.assertGreaterEqual(len(synapse_with_results.results), 1)
        self.assertTrue(
            all(
                result["retweet_count"] >= 10 for result in synapse_with_results.results
            )
        )

    # async def test_search_with_verified(self):
    #     synapse = TwitterSearchSynapse(query="blockchain", verified=False, count=5)

    #     synapse_with_results = await self.miner.search(synapse)

    #     self.assertLessEqual(len(synapse_with_results.results), 5)
    #     self.assertGreaterEqual(len(synapse_with_results.results), 1)
    #     self.assertTrue(
    #         all(result["user"]["verified"] for result in synapse_with_results.results)
    #     )

    async def test_search_by_id(self):
        id = "1890190361588560202"
        synapse = TwitterIDSearchSynapse(id=id)

        synapse_with_results = await self.miner.search_by_id(synapse)

        self.assertEqual(len(synapse_with_results.results), 1)
        self.assertEqual(synapse_with_results.results[0]["id"], id)

    async def test_search_by_urls(self):
        urls = [
            "https://x.com/0xMantleIntern/status/1889322183673156047",
            "https://x.com/TheDustyBC/status/1889952582547935280",
        ]
        synapse = TwitterURLsSearchSynapse(urls=urls)

        synapse_with_results = await self.miner.search_by_urls(synapse)

        self.assertEqual(len(synapse_with_results.results), 2)
        self.assertTrue(
            all(result["url"] in urls for result in synapse_with_results.results)
        )


if __name__ == "__main__":
    unittest.main()
