import re
from typing import List, Tuple
from urllib.parse import urlparse

VALID_DOMAINS = ["x.com"]


class TwitterUtils:
    def __init__(self):
        self.twitter_link_regex = re.compile(
            r"https?://(?:"
            + "|".join(re.escape(domain) for domain in VALID_DOMAINS)
            + r")/(?![^/]*?(?:Twitter|Admin)[^/]*?/)"
            r"(?P<username>[a-zA-Z0-9_]{1,15})/status/(?P<id>\d+)",
            re.IGNORECASE,
        )

    @staticmethod
    def extract_tweet_id(url: str) -> str:
        """
        Extract the tweet ID from a Twitter URL.

        Args:
            url: The Twitter URL to extract the tweet ID from.

        Returns:
            The extracted tweet ID.
        """
        match = re.search(r"/status(?:es)?/(\d+)", url)
        return match.group(1) if match else None

    @staticmethod
    def is_valid_twitter_link(url: str) -> bool:
        """
        Check if the given URL is a valid Twitter link.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid Twitter link, False otherwise.
        """
        # Use the existing regex pattern to validate the full Twitter link format
        regex = re.compile(
            r"https?://(?:"
            + "|".join(re.escape(domain) for domain in VALID_DOMAINS)
            + r")/(?![^/]*?(?:Twitter|Admin)[^/]*?/)"
            r"(?P<username>[a-zA-Z0-9_]{1,15})/status/(?P<id>\d+)$",
            re.IGNORECASE,
        )

        return bool(regex.match(url))

    def find_twitter_links(self, text: str) -> List[str]:
        """
        Find all Twitter links in the given text.

        Args:
            text: The text to search for Twitter links.

        Returns:
            A list of found Twitter links.
        """
        return [match.group() for match in self.twitter_link_regex.finditer(text)]

    def find_twitter_link_with_descriptions(self, text: str) -> List[Tuple[str, str]]:
        """
        Find all Twitter links in the given text along with their descriptions.

        Args:
        text: The text to search for Twitter links and descriptions.

        Returns:
        A list of tuples, each containing a Twitter link and its description.
        """
        results = []
        lines = text.split("\n")
        for line in lines:
            match = self.twitter_link_regex.search(line)
            if match:
                link = match.group()
                # Extract description by removing the link, any surrounding brackets, and leading "-"
                description = re.sub(r"\[|\]|\(|\)", "", line.replace(link, "")).strip()
                description = re.sub(
                    r"^-\s*", "", description
                )  # Remove leading "-" and any following whitespace
                results.append((link, description))
        return results
