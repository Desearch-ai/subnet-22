import os
import bittensor as bt
from datura.protocol import PeopleSearchSynapse, PeopleSearchResult
from neurons.validators.apify.linkedin_scraper_actor import LinkedinScraperActor
from neurons.validators.utils.prompt.criteria_relevance_profile import (
    SearchCriteriaRelevancePrompt,
)
from datura.synapse import collect_responses
from datura.utils import str_linkedin_profile

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )


class PeopleSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.linkedin_scraper_actor = LinkedinScraperActor()

    async def search(self, synapse: PeopleSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query

        # Log the mock search execution
        bt.logging.info(f"Executing people search with query: {query}")

        links = [
            "https://uk.linkedin.com/in/ethanghoreishi",
            "https://uk.linkedin.com/in/vitali-avagyan-phd-a1566234",
            "https://uk.linkedin.com/in/olly-styles-090437132",
            "https://uk.linkedin.com/in/nvedd",
            "https://www.linkedin.com/in/jean-kaddour-344837267",
        ]

        profiles = await self.linkedin_scraper_actor.get_profiles(links)

        prompt = SearchCriteriaRelevancePrompt()
        async_actions = []
        for profile in profiles:
            for i, criteria in enumerate(synapse.criteria):

                async def generate_criteria_summary(i, criteria, profile):
                    if not profile.get("criteria_summary"):
                        profile["criteria_summary"] = [""] * len(synapse.criteria)

                    response = await prompt.get_response(
                        criteria, str_linkedin_profile(profile)
                    )
                    profile["criteria_summary"][i] = prompt.extract_explanation(
                        response.strip()
                    )

                async_actions.append(generate_criteria_summary(i, criteria, profile))

        await collect_responses(async_actions)

        synapse.results = profiles

        return synapse
