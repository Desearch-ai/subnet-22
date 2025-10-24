import os
from typing import List
import bittensor as bt
from apify_client import ApifyClientAsync
from desearch.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")


class CheerioScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/apify/cheerio-scraper
        self.actor_id = "YrQuEkowkNCLdk4j2"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def scrape_metadata(self, urls: List[str]) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []

        if not urls:
            return []

        try:
            run_input = {
                "debugLog": False,
                "excludes": [{"glob": "/**/*.{png,jpg,jpeg}"}],
                "forceResponseEncoding": False,
                "ignoreSslErrors": False,
                "keepUrlFragments": False,
                "pageFunction": """function pageFunction(context) {
    const $ = context.$;
    let pageTitle = $('title').first().text().trim();
    let description = $('meta[name="description"]').attr('content') || '';
    const htmlContent = $('html').html();
    const htmlText = $('html').text()

    if (!description) {
        description = $('meta[property="og:description"]').attr('content') || '';
    }

    const isRedditUrl = context.request.url.includes('reddit.com');
    const isWikipediaUrl = context.request.url.includes('wikipedia.org');
    const isHackerNewsUrl = context.request.url.includes('news.ycombinator.com');

    if (isRedditUrl) {
        // Extract content from the first <p> in the first <div> with class 'text-neutral-content'
        description = $('div.text-neutral-content').first().find('p').map(function() {
            return $(this).text().trim();
        }).get().join('');
    } else if (isWikipediaUrl) {
        // Extract text from the first <p> that isn't empty and clean it
        description = $('p').filter(function() {
            return $(this).text().trim().length > 0;
        }).first().text()
            .replace(/\[\d+\]/g, '')     // Remove citation numbers like [33]
            .replace(/\s+/g, ' ')        // Normalize whitespace
            .trim();                     // Initial trim

        // Explicit period removal as a separate step
        if (description.endsWith('.')) {
            description = description.slice(0, -1);
        }

        // Final trim to ensure no extra whitespace
        description = description.trim();

    } else if (isHackerNewsUrl) {
        // Select the first comment div with classes 'commtext' and 'c00'
        description = $('div.commtext.c00').first()
            // Find all <p> tags within this div and replace them with their text content
            .find('p').replaceWith(function() {
                return $(this).text();
            })
            // Return to the original selection (div.commtext.c00)
            .end()
            // Extract the combined text content and trim any whitespace
            .text().trim();
    }

    return {
        url: context.request.url,
        pageTitle,
        description,
        htmlContent,
        htmlText,
    };
}""",
                "postNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept a single argument: the "crawlingContext" object.\n[\n    async (crawlingContext) => {\n        // ...\n    },\n]',
                "preNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept two arguments: the "crawlingContext" object\n// and "requestAsBrowserOptions" which are passed to the `requestAsBrowser()`\n// function the crawler calls to navigate..\n[\n    async (crawlingContext, requestAsBrowserOptions) => {\n        // ...\n    }\n]',
                "proxyConfiguration": self.get_proxy_configuration(urls),
                "proxyRotation": "RECOMMENDED",
                "startUrls": [{"url": url} for url in urls],
            }

            run = await self.client.actor(self.actor_id).call(
                run_input=run_input,
                memory_mbytes=512,
            )

            result = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                url = item.get("url", "")
                title = item.get("pageTitle")
                description = item.get("description")
                html_content = item.get("htmlContent")
                html_text = item.get("htmlText")
                result.append(
                    {
                        "title": title,
                        "snippet": description,
                        "link": url,
                        "html_content": html_content,
                        "html_text": html_text,
                    }
                )

            return result
        except Exception as e:
            bt.logging.warning(
                f"CheerioScraperActor: Failed to scrape web links {urls}: {e}"
            )
            return []

    def get_proxy_configuration(self, urls: List[str]) -> dict:
        """
        Returns the proxy configuration based on the URLs to scrape.
        Used to save scraping costs as some urls don't need residential proxies.
        """

        residential = True

        if all(
            any(
                domain in url
                for domain in ["news.ycombinator.com", "wikipedia.org", "arxiv.org"]
            )
            for url in urls
        ):
            residential = False

        proxy_configuration = {
            "useApifyProxy": True,
            "apifyProxyGroups": [],
        }

        if residential:
            proxy_configuration["apifyProxyGroups"].append("RESIDENTIAL")

        return proxy_configuration
