import unittest
from unittest.mock import AsyncMock

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterScraperTweet,
    TwitterScraperUser,
)
from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)
from tests_data.tweets.tweet1 import tweet1
from tests_data.tweets.tweet2 import tweet2


def make_tweet(
    text: str,
    quote_text: str | None = None,
    quote_username: str | None = None,
) -> TwitterScraperTweet:
    quote = None
    if quote_text is not None:
        quote = TwitterScraperTweet(
            user=(
                TwitterScraperUser(id="q-user", username=quote_username)
                if quote_username
                else None
            ),
            id="quoted-id",
            text=quote_text,
            reply_count=0,
            retweet_count=0,
            like_count=0,
            quote_count=0,
            bookmark_count=0,
            url="https://x.com/quoted/status/1",
            created_at="Sun Feb 02 02:39:05 +0000 2025",
            is_quote_tweet=False,
            is_retweet=False,
        )

    return TwitterScraperTweet(
        user=TwitterScraperUser(id="user", username="author"),
        id="main-id",
        text=text,
        reply_count=0,
        retweet_count=0,
        like_count=0,
        quote_count=0,
        bookmark_count=0,
        url="https://x.com/author/status/2",
        created_at="Sun Feb 09 08:40:53 +0000 2025",
        is_quote_tweet=quote is not None,
        is_retweet=False,
        quote=quote,
    )


class BuildRelevanceContentTestCase(unittest.TestCase):
    def test_quote_tweet_includes_quoted_text_and_handle(self):
        tweet = make_tweet(
            "Big news today",
            quote_text="The original announcement about widgets",
            quote_username="OriginalAuthor",
        )

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertIn("Big news today", content)
        self.assertIn("The original announcement about widgets", content)
        self.assertIn("Quoted tweet (@OriginalAuthor):", content)

    def test_quote_without_username_uses_plain_label(self):
        tweet = make_tweet(
            "Commentary",
            quote_text="Quoted body text",
            quote_username=None,
        )

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertIn("Quoted tweet: Quoted body text", content)
        self.assertNotIn("@", content)

    def test_non_quote_tweet_unchanged(self):
        tweet = make_tweet("Standalone tweet text", quote_text=None)

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertEqual(content, "Standalone tweet text")
        self.assertNotIn("Quoted tweet", content)

    def test_empty_quote_text_is_ignored(self):
        tweet = make_tweet("Has empty quote", quote_text="   ", quote_username="x")

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertEqual(content, "Has empty quote")
        self.assertNotIn("Quoted tweet", content)

    def test_none_quote_is_handled(self):
        tweet = make_tweet("No quote field", quote_text=None)
        tweet.quote = None

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertEqual(content, "No quote field")

    def test_real_fixture_non_quote(self):
        tweet = TwitterScraperTweet(**tweet1)

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertEqual(content, tweet1["text"])

    def test_real_fixture_with_quote(self):
        tweet = TwitterScraperTweet(**tweet2)

        content = TwitterContentRelevanceModel.build_relevance_content(tweet)

        self.assertIn(tweet2["text"], content)
        self.assertIn(tweet2["quote"]["text"], content)
        self.assertIn(f"@{tweet2['quote']['user']['username']}", content)


class LlmProcessValidatorTweetsTestCase(unittest.IsolatedAsyncioTestCase):
    def _model_with_mock_llm(self) -> TwitterContentRelevanceModel:
        model = TwitterContentRelevanceModel.__new__(TwitterContentRelevanceModel)
        model.reward_llm = AsyncMock()
        model.reward_llm.llm_processing = AsyncMock(return_value={})
        return model

    async def test_scoring_messages_carry_quoted_context(self):
        model = self._model_with_mock_llm()
        response = ScraperStreamingSynapse(
            prompt="What is happening with widgets?",
            validator_tweets=[
                make_tweet(
                    "See this",
                    quote_text="Detailed widget update from the source",
                    quote_username="SourceAcct",
                )
            ],
            tools=["Twitter Search"],
        )

        await model.llm_process_validator_tweets(response)

        model.reward_llm.llm_processing.assert_awaited_once()
        scoring_messages = model.reward_llm.llm_processing.call_args.args[0]
        user_content = scoring_messages[0]["main-id"][1]["content"]
        self.assertIn("widget update", user_content)

    async def test_scoring_messages_non_quote_have_no_quote_marker(self):
        model = self._model_with_mock_llm()
        response = ScraperStreamingSynapse(
            prompt="Tell me about widgets",
            validator_tweets=[make_tweet("Plain standalone widget tweet")],
            tools=["Twitter Search"],
        )

        await model.llm_process_validator_tweets(response)

        scoring_messages = model.reward_llm.llm_processing.call_args.args[0]
        user_content = scoring_messages[0]["main-id"][1]["content"]
        self.assertNotIn("Quoted tweet", user_content)


if __name__ == "__main__":
    unittest.main()
