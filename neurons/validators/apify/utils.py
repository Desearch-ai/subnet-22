import asyncio

import bittensor as bt


async def scrape_links_with_retries(
    urls, scraper_actor_class, group_size, max_attempts
):
    fetched_links_with_metadata = []
    non_fetched_links = urls.copy()
    attempt = 1

    while attempt <= max_attempts and non_fetched_links:
        bt.logging.info(
            f"Attempt {attempt}/{max_attempts} for {scraper_actor_class.__name__}, processing {len(non_fetched_links)} links."
        )

        url_groups = [
            non_fetched_links[i : i + group_size]
            for i in range(0, len(non_fetched_links), group_size)
        ]

        tasks = [
            asyncio.create_task(scraper_actor_class().scrape_metadata(urls=group))
            for group in url_groups
        ]

        # Wait for tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results and handle exceptions
        for result in results:
            if isinstance(result, Exception):
                bt.logging.error(
                    f"Error in {scraper_actor_class.__name__} scraper attempt {attempt}: {str(result)}"
                )
                continue
            fetched_links_with_metadata.extend(result)

        # Update non-fetched links
        fetched_urls = {link.get("link") for link in fetched_links_with_metadata}
        non_fetched_links = [
            url for url in non_fetched_links if url not in fetched_urls
        ]

        attempt += 1

    return fetched_links_with_metadata, non_fetched_links
