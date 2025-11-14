import re
from typing import List


class WebSearchUtils:
    @staticmethod
    def find_links(text: str) -> List[str]:
        link_regex = r"\[.*?\]\((https?://[^\s\)]+)\)"
        links = []
        for match in re.finditer(link_regex, text):
            link = match.group(1)
            links.append(link)
        return links

    @staticmethod
    def find_links_by_domain(text: str, domain: str) -> List[str]:
        link_regex = rf"\[.*?\]\((https?://(?:[a-zA-Z0-9-]+\.)*{re.escape(domain)}(?:/[^\s\)]+)?)\)"
        links = []
        for match in re.finditer(link_regex, text):
            link = match.group(1)
            links.append(link)
        return links

    @staticmethod
    def remove_trailing_slash(url: str) -> str:
        """
        Remove the trailing slash from a URL.

        Args:
        url: The URL to remove the trailing slash from.

        Returns:
        The URL with the trailing slash removed.
        """
        if url.endswith("/"):
            return url[:-1]

        return url
