import bittensor as bt
from datura.protocol import (
    TwitterSearchSynapse,
    TwitterIDSearchSynapse,
    TwitterURLsSearchSynapse,
)
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.twitter_scraper_actor = TwitterScraperActor()

    async def search(self, synapse: TwitterSearchSynapse):
        # Extract the query parameters from the synapse
        query = synapse.query
        search_params = {
            "sort": synapse.sort,
            "start": synapse.start_date,
            "end": synapse.end_date,
            "tweetLanguage": synapse.lang,
            "onlyVerifiedUsers": synapse.verified,
            "onlyTwitterBlue": synapse.blue_verified,
            "onlyQuote": synapse.is_quote,
            "onlyVideo": synapse.is_video,
            "onlyImage": synapse.is_image,
            "minimumRetweets": synapse.min_retweets,
            "minimumReplies": synapse.min_replies,
            "minimumFavorites": synapse.min_likes,
            "author": synapse.user,
            "maxItems": synapse.count,
        }

        # Log query and search parameters
        bt.logging.info(
            f"Executing apify search with query: {query} and params: {search_params}"
        )

        tweets = await self.twitter_scraper_actor.get_tweets_advanced(
            **search_params, searchTerms=[synapse.query]
        )

        synapse.results = [tweet.model_dump() for tweet in tweets]

        bt.logging.info(f"Here is the final synapse: {synapse}")
        return synapse

    async def search_by_id(self, synapse: TwitterIDSearchSynapse):
        """
        Perform a Twitter search based on a specific tweet ID.
        """
        tweet_id = synapse.id

        # Log the search operation
        bt.logging.info(f"Searching for tweet by ID: {tweet_id}")

        url = [f"https://x.com/twitter/status/{tweet_id}"]

        tweets = await self.twitter_scraper_actor.get_tweets_advanced(
            urls=url,
            includeSearchTerms=False,
            onlyImage=False,
            onlyQuote=False,
            onlyTwitterBlue=False,
            onlyVerifiedUsers=False,
            onlyVideo=False,
        )

        synapse.results = [tweet.model_dump() for tweet in tweets]

        return synapse

    async def search_by_urls(self, synapse: TwitterURLsSearchSynapse):
        """
        Perform a Twitter search based on multiple tweet URLs.

        Parameters:
            synapse (TwitterURLsSearchSynapse): Contains the list of tweet URLs.

        Returns:
            TwitterURLsSearchSynapse: The synapse with fetched tweets in the results field.
        """
        urls = synapse.urls

        # Log the search operation
        bt.logging.info(f"Searching for tweets by URLs: {urls}")

        tweets = await self.twitter_scraper_actor.get_tweets(urls)

        synapse.results = [tweet.model_dump() for tweet in tweets]

        return synapse
