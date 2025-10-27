from wikipedia import set_rate_limiting, page, search, exceptions
import bittensor as bt
from datetime import timedelta
from typing import Optional, List

WIKIPEDIA_MAX_QUERY_LENGTH = 300
MAX_DOC_CONTENT_CHARS = 1000

set_rate_limiting(True, min_wait=timedelta(milliseconds=50))


class WikipediaAPIWrapper:
    """Custom Wikipedia API Wrapper without LangChain dependency."""

    def __init__(self, top_k_results: int = 3):
        self.top_k_results = top_k_results

    def run(self, query: str) -> List[dict]:

        try:
            page_titles = search(query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results)
            summaries = []
            for page_title in page_titles[: self.top_k_results]:
                if wiki_page := self._fetch_page(page_title):
                    summaries.append(
                        {
                            "title": page_title,
                            "summary": wiki_page.summary[:MAX_DOC_CONTENT_CHARS],
                            "url": wiki_page.url,
                        }
                    )
            if not summaries:
                bt.logging.info("No good Wikipedia Search Result was found")

            return summaries
        except Exception as e:
            bt.logging.error(f"Error occurred while fetching Wikipedia results: {e}")
            return []

    @staticmethod
    def _fetch_page(page_title: str) -> Optional[page]:
        try:
            return page(title=page_title, auto_suggest=False)
        except (exceptions.PageError, exceptions.DisambiguationError):
            return None
