import re

VALID_DOMAINS = ["x.com"]


class TwitterUtils:
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
