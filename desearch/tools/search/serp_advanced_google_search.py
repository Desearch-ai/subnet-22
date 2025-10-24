from .serp_api_wrapper import SerpAPIWrapper
import os
import bittensor as bt

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )


class SerpAdvancedGoogleSearch:
    def __init__(self, site: str, language: str, region: str, date_filter: str) -> None:
        self.site = site
        self.language = language
        self.region = region
        self.date_filter = date_filter

    async def run(self, query: str):
        search = SerpAPIWrapper(
            serpapi_api_key=SERPAPI_API_KEY,
            params={
                "engine": "google",
                "hl": self.language,
                "gl": self.region,
                "tbs": self.date_filter,
            },
        )

        query = f"{query} site:{self.site}"

        try:
            result = await search.arun(query)
            return result
        except Exception as err:
            if "Invalid API key" in str(err):
                bt.logging.error(f"SERP API Key is invalid: {err}")
                return "SERP API Key is invalid"

            bt.logging.warning(f"Could not perform SERP Search: {err}")
            return "Could not search Google. Please try again later."

    def process_response(self, response):
        if "organic_results" in response:
            results = [
                {
                    "title": result["title"],
                    "url": result["link"],
                    "snippet": result.get("snippet", ""),
                }
                for result in response["organic_results"]
            ]
        else:
            results = []

        return results
