from datetime import datetime, timedelta
import pytz
import unittest
from neurons.validators.reward.twitter_basic_search_content_relevance import (
    TwitterBasicSearchContentRelevanceModel,
)
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)
from tests_data.tweets.tweet1 import tweet1
from tests_data.tweets.tweet2 import tweet2


class TwitterBasicSearchContentRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = TwitterBasicSearchContentRelevanceModel(
            self.device, self.scoring_type
        )

    async def test_get_rewards(self):
        rewards, grouped_score = await self.model.get_rewards(
            [
                TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1, tweet2],
                ),
                TwitterSearchSynapse(
                    query="test query",
                    results=[tweet2],
                ),
                TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1],
                ),
            ],
            [1, 2, 3],
        )

        self.assertEqual(rewards[0].reward, 0.5)
        self.assertEqual(rewards[1].reward, 0)
        self.assertEqual(rewards[2].reward, 1)

        self.assertEqual(grouped_score, {1: 0.5, 2: 0, 3: 1})

    def test_check_tweet_content(self):
        for trueCase in [
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"], results=[tweet1], validator_tweets=[tweet1]
                ),
                "description": "should return 1 for TwitterSearchSynapse tweet1",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet2["text"], results=[tweet2], validator_tweets=[tweet2]
                ),
                "description": "should return 1 for TwitterSearchSynapse tweet2",
            },
            {
                "case": TwitterIDSearchSynapse(
                    id=tweet1["id"],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 1 for TwitterIDSearchSynapse",
            },
            {
                "case": TwitterURLsSearchSynapse(
                    urls=[tweet1["url"]],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 1 for TwitterURLsSearchSynapse",
            },
        ]:
            self.assertEqual(
                self.model.check_tweet_content(trueCase["case"]),
                1,
                trueCase["description"],
            )

        for falseCase in [
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"], results=[tweet1], validator_tweets=[]
                ),
                "description": "should return 0 : missing tweet id",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"], results=[tweet1], validator_tweets=[tweet2]
                ),
                "description": "should return 0 : different tweets",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[{**tweet1, "reply_count": "invalid count"}],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : invalid tweet",
            },
            {
                "case": TwitterIDSearchSynapse(
                    id="wrong id",
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong id for TwitterIDSearchSynapse",
            },
            {
                "case": TwitterURLsSearchSynapse(
                    urls=["wrong url"],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong url for TwitterURLsSearchSynapse",
            },
            {
                "case": TwitterSearchSynapse(
                    query="wrong query", results=[tweet1], validator_tweets=[tweet1]
                ),
                "description": "should return 0 : wrong query",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    min_likes=tweet1["like_count"] + 1,
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong min_likes",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    min_retweets=tweet1["retweet_count"] + 1,
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong min_retweets",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    min_replies=tweet1["reply_count"] + 1,
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong min_replies",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    user="wrong user name",
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong user name",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    verified=not tweet1["user"]["verified"],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong verified",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    is_quote=not tweet1["is_quote_tweet"],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong is_quote",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    is_image=True,
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong is_image",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    is_video=False,
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong is_video",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    start_date=(
                        datetime.strptime(
                            tweet1["created_at"], "%a %b %d %H:%M:%S %z %Y"
                        ).replace(tzinfo=pytz.UTC)
                        + timedelta(days=1)
                    ).strftime("%Y-%m-%d"),
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong start_date",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    end_date=(
                        datetime.strptime(
                            tweet1["created_at"], "%a %b %d %H:%M:%S %z %Y"
                        ).replace(tzinfo=pytz.UTC)
                        - timedelta(days=1)
                    ).strftime("%Y-%m-%d"),
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong end_date",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    lang="wrong lang",
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong lang",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    blue_verified=not tweet1["user"]["is_blue_verified"],
                    results=[tweet1],
                    validator_tweets=[tweet1],
                ),
                "description": "should return 0 : wrong blue_verified",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1],
                    validator_tweets=[{**tweet1, "id": "wrong id"}],
                ),
                "description": "should return 0 : wrong tweet exact match field",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1],
                    validator_tweets=[{**tweet1, "like_count": 0}],
                ),
                "description": "should return 0 : wrong tweet numeric field",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1],
                    validator_tweets=[
                        {**tweet1, "user": {**tweet1["user"], "id": "wrong id"}}
                    ],
                ),
                "description": "should return 0 : wrong tweet user exact match field",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet1["text"],
                    results=[tweet1],
                    validator_tweets=[
                        {**tweet1, "user": {**tweet1["user"], "favourites_count": 0}}
                    ],
                ),
                "description": "should return 0 : wrong tweet user numeric value",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet2["text"],
                    results=[
                        {
                            **tweet2,
                            "entities": {**tweet2["entities"], "another_field": "any"},
                        }
                    ],
                    validator_tweets=[tweet2],
                ),
                "description": "should return 0 : wrong tweet nested field value",
            },
            {
                "case": TwitterSearchSynapse(
                    query=tweet2["text"],
                    results=[
                        {
                            **tweet2,
                            "user": {
                                **tweet2["user"],
                                "entities": {
                                    **tweet2["user"]["entities"],
                                    "another_field": "any",
                                },
                            },
                        }
                    ],
                    validator_tweets=[tweet2],
                ),
                "description": "should return 0 : wrong tweet user nested field value",
            },
        ]:
            self.assertEqual(
                self.model.check_tweet_content(falseCase["case"]),
                0,
                falseCase["description"],
            )

    def test_compare_nested_fields(self):
        self.assertTrue(self.model.compare_nested_fields({}, {}))
        self.assertTrue(self.model.compare_nested_fields(None, None))
        self.assertTrue(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )

        self.assertFalse(self.model.compare_nested_fields({}, None))
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                    "g": "other",
                },
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 2, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 1, "c": [1, 3, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "f"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 1, "c": [1, 2]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
            )
        )
        self.assertFalse(
            self.model.compare_nested_fields(
                {
                    "a": {"b": 1, "c": [1, 2, {"d": "e"}]},
                    "f": ([1, 2, "a"], {"g": "h", "i": ["j"]}),
                },
                {
                    "a": {"b": 1, "c": [1, 2]},
                    "f": ([1, 2, "b"], {"g": "h", "i": ["j"]}),
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
