import re
import torch
import json
import time
from typing import List
from urllib.parse import urlparse
from neurons.validators.utils.tasks import TwitterTask
from neurons.validators.penalty import BasePenaltyModel, PenaltyModelType
from datura.protocol import ScraperStreamingSynapse
from datura.services.twitter_api_wrapper import TwitterAPIClient, VALID_DOMAINS


class LinkValidationPenaltyModel(BasePenaltyModel):
    """
    A class to validate the presence and relevance of Twitter links in a text.
    Inherits from BasePenaltyModel.
    """

    def __init__(self, max_penalty: float):
        """
        Initialize the TwitterLinkValidator with a maximum penalty and the user's prompt content.

        Args:
            max_penalty: The maximum penalty that can be applied to a completion.
            prompt_content: The content of the user's prompt to check relevance.
        """
        super().__init__(max_penalty)
        self.client = TwitterAPIClient()

    @property
    def name(self) -> str:
        """
        Returns the name of the penalty model.

        Returns:
            The name of the penalty model as defined in PenaltyModelType.
        """
        return PenaltyModelType.link_validation_penalty.value

    def is_valid_twitter_link(self, url: str) -> bool:
        """
        Check if the given URL is a valid Twitter link.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid Twitter link, False otherwise.
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc.lower() in VALID_DOMAINS

    async def fetch_twitter_data_for_links(self, links: List[str]) -> List[dict]:
        """
        Retrieve Twitter data for the given list of Twitter links.

        Args:
            links: A list of Twitter links to retrieve data for.

        Returns:
            A list of dictionaries containing the retrieved Twitter data.
        """
        tweet_ids = [
            self.client.utils.extract_tweet_id(link)
            for link in links
            if self.is_valid_twitter_link(link)
        ]
        return await self.client.get_tweets_by_ids(tweet_ids)

    async def calculate_penalties(
        self, task: TwitterTask, responses: List[ScraperStreamingSynapse]
    ) -> torch.FloatTensor:
        """
        Calculates the penalties for each completion based on the presence and relevance of Twitter links.

        Args:
            task: The task containing the query criteria.
            responses: A list of strings representing the completed texts.

        Returns:
            A tensor of penalties for each completion.
        """
        self.prompt_content = task.base_text
        penalties = []
        for response in responses:
            # time.sleep(2)
            completion = response.completion
            twitter_links = self.client.utils.find_twitter_links(completion)
            if twitter_links and all(
                self.is_valid_twitter_link(link) for link in twitter_links
            ):
                valid_links = response.completion_links

                # response.tweets = json.dumps(valid_links, indent=4, sort_keys=True)
                penalty = self.max_penalty * len(valid_links) / len(twitter_links)
                penalties.append(penalty)
            else:
                penalties.append(0.0)
        return torch.tensor(penalties, dtype=torch.float32)
