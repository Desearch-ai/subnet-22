import copy
import unittest

from desearch.protocol import (
    ScraperStreamingSynapse,
    SearchResultItem,
    TwitterSearchSynapse,
)
from neurons.validators.penalty.result_schema_penalty import (
    ResultSchemaPenaltyModel,
    _is_valid_tweet,
)
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)


def _valid_tweet(tid: str = "1", text: str = "hello") -> dict:
    return {
        "id": tid,
        "url": f"https://x.com/foo/status/{tid}",
        "text": text,
        "created_at": "Mon May 10 12:00:00 +0000 2026",
        "is_quote_tweet": False,
        "is_retweet": False,
        "conversation_id": tid,
        "in_reply_to_screen_name": None,
        "in_reply_to_status_id": None,
        "in_reply_to_user_id": None,
        "quoted_status_id": None,
        "lang": "en",
        "media": [],
        "reply_count": 0,
        "view_count": 0,
        "retweet_count": 0,
        "like_count": 0,
        "quote_count": 0,
        "bookmark_count": 0,
        "user": {
            "id": "1",
            "url": "u",
            "name": "n",
            "username": "foo",
            "created_at": "c",
            "description": "",
            "profile_image_url": "",
            "profile_banner_url": "",
            "verified": False,
            "can_dm": True,
            "can_media_tag": True,
            "location": "",
            "pinned_tweet_ids": [],
            "is_blue_verified": False,
            "followers_count": 0,
            "media_count": 0,
            "statuses_count": 0,
        },
    }


class ResultSchemaPenaltyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = ResultSchemaPenaltyModel()

    async def test_twitter_all_valid(self):
        response = TwitterSearchSynapse(
            query="x",
            results=[_valid_tweet("1"), _valid_tweet("2")],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])

    async def test_twitter_malformed_tweet(self):
        """A tweet that fails TwitterScraperTweet schema → fractional penalty."""
        response = TwitterSearchSynapse(
            query="x",
            results=[_valid_tweet("1"), {"id": "incomplete"}],
        )
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0.5])

    async def test_twitter_empty_required_field(self):
        """Schema passes but empty `text` → still flagged."""
        bad = _valid_tweet("1")
        bad["text"] = ""
        response = TwitterSearchSynapse(query="x", results=[bad])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [1.0])

    async def test_ai_combined_groups(self):
        """Schema check spans miner_tweets + every *_search_results field."""
        good_tweet = _valid_tweet("1")
        bad_search = SearchResultItem(title="", link="https://x", snippet="s")
        response = ScraperStreamingSynapse(
            prompt="x",
            miner_tweets=[good_tweet],
            search_results=[
                SearchResultItem(title="T1", link="https://a", snippet="s"),
                bad_search,
            ],
        )
        penalties = await self.model.calculate_penalties([response])
        # 1 invalid out of 3 total (1 tweet + 2 search) = 0.333...
        self.assertAlmostEqual(penalties.tolist()[0], 1 / 3, places=5)

    async def test_empty_results_no_penalty(self):
        response = TwitterSearchSynapse(query="x", results=[])
        penalties = await self.model.calculate_penalties([response])
        self.assertEqual(penalties.tolist(), [0])


class PreprocessTweetImmutabilityTestCase(unittest.TestCase):
    """Regression: preprocess_tweet must not mutate its input."""

    def setUp(self):
        self.model = TwitterBasicSearchContentRelevanceModel(
            scoring_type=None, neuron=None
        )

    def _tweet_with_media_and_quote(self) -> dict:
        tweet = _valid_tweet("1")
        tweet["extended_entities"] = {
            "media": [
                {
                    "display_url": "pic.x.com/AAA",
                    "expanded_url": "https://twitter.com/foo/status/1/photo/1",
                    "type": "photo",
                    "url": "https://t.co/AAA",
                    "media_url_https": "https://pbs.twimg.com/media/AAA.jpg",
                }
            ]
        }
        tweet["quote"] = _valid_tweet("2", text="quoted")
        tweet["is_quote_tweet"] = True
        return tweet

    def test_preprocess_does_not_mutate_input(self):
        tweet = self._tweet_with_media_and_quote()
        snapshot = copy.deepcopy(tweet)

        normalized = self.model.preprocess_tweet(tweet)

        self.assertEqual(tweet, snapshot, "preprocess_tweet mutated its input")
        self.assertIsNot(normalized, tweet)
        self.assertIsInstance(normalized["extended_entities"]["media"][0], str)
        self.assertIsNone(normalized["quote"]["reply_count"])

    def test_original_remains_schema_valid_after_preprocess(self):
        tweet = self._tweet_with_media_and_quote()
        self.model.preprocess_tweet(tweet)
        self.assertTrue(_is_valid_tweet(tweet))


if __name__ == "__main__":
    unittest.main()
