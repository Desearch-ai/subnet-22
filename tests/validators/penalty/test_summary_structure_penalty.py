import unittest

from desearch.protocol import (
    ResultType,
    ScraperStreamingSynapse,
    ScraperTextRole,
    SearchResultItem,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.summary_structure_penalty import (
    SummaryStructurePenaltyModel,
)


def _tweet(tid: str, username: str = "foo") -> dict:
    return {"id": tid, "user": {"username": username}}


def _summary(text: str) -> dict:
    return {ScraperTextRole.FINAL_SUMMARY.value: [text]}


class SummaryStructurePenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SummaryStructurePenaltyModel()

    async def test_all_summary_links_in_data_passes(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("**Header**\n[a](https://x.com/foo/status/123)"),
            miner_tweets=[_tweet("123")],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_one_link_outside_data_full_penalty(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary(
                "**Header**\n[a](https://x.com/foo/status/123)\n[b](https://nope.com/x)"
            ),
            miner_tweets=[_tweet("123")],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_hash_header_full_penalty(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("# Bad header\n[a](https://x.com/foo/status/123)"),
            miner_tweets=[_tweet("123")],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_no_links_full_penalty(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("**Header**\nplain text only"),
            miner_tweets=[_tweet("123")],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_empty_summary_full_penalty(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary(""),
            miner_tweets=[_tweet("123")],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_only_links_result_type_skipped(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("garbage"),
            result_type=ResultType.ONLY_LINKS,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_search_results_match_link(self):
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("**Header**\n[a](https://news.com/article)"),
            search_results=[
                SearchResultItem(
                    title="T", link="https://news.com/article", snippet="s"
                )
            ],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_www_difference_does_not_penalize(self):
        """www.reddit.com in data, reddit.com in summary — should match."""
        response = ScraperStreamingSynapse(
            prompt="x",
            text_chunks=_summary("**Header**\n[a](https://reddit.com/r/foo)"),
            search_results=[
                SearchResultItem(
                    title="T", link="https://www.reddit.com/r/foo", snippet="s"
                )
            ],
            result_type=ResultType.LINKS_WITH_FINAL_SUMMARY,
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_non_twitter_synapse_skipped(self):
        """SummaryStructure only applies to AI search (ScraperStreamingSynapse)."""
        response = TwitterSearchSynapse(query="x", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
