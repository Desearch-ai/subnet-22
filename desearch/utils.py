import asyncio
import html
import math
import os
import re
import unicodedata
from typing import List

import aiohttp
import bittensor as bt
import numpy as np
from openai import AsyncOpenAI
from pydantic import ValidationError

from desearch.protocol import (
    SearchMode,
    Model,
    ScoringModel,
    TwitterScraperTweet,
    WebSearchResult,
)
from desearch.redis.utils import save_moving_averaged_scores
from desearch.services.twitter_utils import TwitterUtils
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


class LazyAsyncOpenAIClient:
    def __init__(self, **client_kwargs):
        self._client_kwargs = client_kwargs
        self._client = None
        self._api_key = None

    def _get_client(self):
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        if self._client is None or self._api_key != api_key:
            self._client = AsyncOpenAI(api_key=api_key, **self._client_kwargs)
            self._api_key = api_key

        return self._client

    def __getattr__(self, name):
        return getattr(self._get_client(), name)


client = LazyAsyncOpenAIClient(timeout=90.0)


MODE_BUDGETS: dict[SearchMode, int] = {
    SearchMode.FAST: 5,
    SearchMode.BALANCED: 15,
    SearchMode.DEEP: 30,
}

AI_SEARCH_MODES: List[SearchMode] = list(MODE_BUDGETS.keys())

SERVING_FLOOR = 15


def get_mode_budget(mode: SearchMode) -> int:
    return MODE_BUDGETS[SearchMode(mode)]


def get_mode_serving_budget(mode: SearchMode) -> int:
    return max(get_mode_budget(mode), SERVING_FLOOR)


def get_max_execution_time(model: Model, count: int):
    if count > 10:
        # For every 50 items add additional 5s for execution time
        return 15 + math.ceil((count - 50) / 50) * 5

    if model == Model.NOVA:
        return 15
    elif model == Model.ORBIT:
        return 30
    elif model == Model.HORIZON:
        return 120


def get_chutes_api_key() -> str:
    return os.environ.get("CHUTES_API_TOKEN") or os.environ.get("CHUTES_API_KEY", "")


async def call_chutes(messages, temperature, model, seed=1234, response_format=None):
    api_key = get_chutes_api_key()

    if not api_key:
        bt.logging.warning(
            "Set CHUTES_API_TOKEN (or CHUTES_API_KEY) to score with Chutes."
        )
        return None

    model_name = getattr(model, "value", model)
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format

    for _ in range(2):
        bt.logging.trace(
            f"Calling chutes. Temperature = {temperature}, Model = {model_name}, Seed = {seed}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        if content:
                            return content
                        bt.logging.error("Chutes returned empty content.")
                    else:
                        body = (await response.text())[:200]
                        bt.logging.error(f"Chutes HTTP {response.status}: {body}")
        except Exception as e:
            bt.logging.error(f"Error when calling chutes: {e}")

        await asyncio.sleep(1)

    return None


async def call_openai(messages, model, temperature=1, response_format=None):
    for _ in range(2):
        bt.logging.trace(
            f"Calling Openai. Temperature = {temperature}, Model = {model}, "
            f"Messages = {messages}"
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            content = response.choices[0].message.content
            bt.logging.trace(f"validator response is {content}")
            return content

        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {e}")
            await asyncio.sleep(1)

    return None


async def call_scoring_llm(messages, model, temperature=0.0001, response_format=None):
    if model == ScoringModel.OPENAI_GPT4_1_NANO:
        return await call_openai(
            messages,
            model="gpt-4.1-nano",
            temperature=temperature,
            response_format=response_format,
        )

    result = await call_chutes(
        messages, temperature=temperature, model=model, response_format=response_format
    )

    if result is None:
        bt.logging.warning(
            f"Chutes scoring with {getattr(model, 'value', model)} failed; falling back to gpt-4.1-nano."
        )
        return await call_openai(
            messages,
            model="gpt-4.1-nano",
            temperature=temperature,
            response_format=response_format,
        )

    return result


async def resync_metagraph(self):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")

    # Copies axons before syncing.
    previous_axons = list(self.metagraph.axons)  # Only copy what you need

    self.metagraph = await self.subtensor.metagraph(self.config.netuid)

    # Check if the metagraph axon info has changed.
    if previous_axons == list(self.metagraph.axons):
        return

    bt.logging.info(
        "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
    )

    # Zero out all hotkeys that have been replaced.
    for uid, hotkey in enumerate(self.hotkeys):
        if hotkey != self.metagraph.hotkeys[uid]:
            self.moving_averaged_scores[uid] = 0  # hotkey has been replaced

    # Check to see if the metagraph has changed size.
    # If so, we need to add new hotkeys and moving averages.
    if len(self.hotkeys) < len(self.metagraph.hotkeys):
        new_moving_average = np.zeros(int(self.metagraph.n), dtype=np.float32)
        min_len = min(len(self.hotkeys), len(self.moving_averaged_scores))
        new_moving_average[:min_len] = self.moving_averaged_scores[:min_len]
        self.moving_averaged_scores = new_moving_average

    bt.logging.info("Saving moving averaged scores to Redis after metagraph update")
    await save_moving_averaged_scores(self.moving_averaged_scores)
    bt.logging.info("Saved weights to Redis after metagraph update")

    # Update the hotkeys.
    self.hotkeys = list(self.metagraph.hotkeys)


def clean_text(text):
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)

    # Remove mentions at the beginning of the text
    text = re.sub(r"^(@\w+\s*)+", "", text)

    # Remove emojis and other symbols
    text = re.sub(r"[^\w\s,]", "", text)

    # Normalize whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Remove non-printable characters and other special Unicode characters
    text = "".join(
        char
        for char in text
        if char.isprintable() and not unicodedata.category(char).startswith("C")
    )

    return text


def format_text_for_match(text):
    # Unescape HTML entities first
    text = html.unescape(text)
    # url shorteners can cause problems with tweet verification, so remove urls from the text comparison.
    text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)
    # Some scrapers put the mentions at the front of the text, remove them.
    text = re.sub(r"^(@\w+\s*)+", "", text)
    # And some trim trailing whitespace at the end of newlines, so ignore whitespace.
    text = re.sub(r"\s+", "", text)
    # The validator apify actor uses the tweet.text field and not the note_tweet field (> 280) charts, so only
    # use the first 280 chars for comparison.
    text = text[:280]
    return text


async def scrape_tweets_with_retries(
    urls: List[str], group_size: int, max_attempts: int
):
    fetched_tweets = []
    non_fetched_links = urls.copy()
    attempt = 1

    while attempt <= max_attempts and non_fetched_links:
        bt.logging.info(
            f"Attempt {attempt}/{max_attempts}, processing {len(non_fetched_links)} links."
        )

        url_groups = [
            non_fetched_links[i : i + group_size]
            for i in range(0, len(non_fetched_links), group_size)
        ]

        tasks = [
            asyncio.create_task(TwitterScraperActor().get_tweets(urls=group))
            for group in url_groups
        ]

        # Wait for tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results and handle exceptions
        for result in results:
            if isinstance(result, Exception):
                bt.logging.error(
                    f"Error in TwitterScraperActor attempt {attempt}: {str(result)}"
                )
                continue
            fetched_tweets.extend(result)

        # Update non_fetched_links
        fetched_tweet_ids = {tweet.id for tweet in fetched_tweets}
        non_fetched_links = [
            link
            for link in non_fetched_links
            if TwitterUtils.extract_tweet_id(link) not in fetched_tweet_ids
        ]

        if non_fetched_links:
            bt.logging.info(
                f"Retrying fetching non-fetched {len(non_fetched_links)} tweets. Retries left: {max_attempts - attempt}"
            )
            await asyncio.sleep(3)

        attempt += 1

    return fetched_tweets, non_fetched_links


def is_valid_tweet(tweet):
    try:
        _ = TwitterScraperTweet(**tweet)
    except ValidationError as e:
        bt.logging.error(f"Invalid miner tweet data: {e}")
        return False
    return True


def is_valid_web_search_result(result):
    try:
        WebSearchResult(**result)
    except ValidationError as e:
        bt.logging.error(f"Invalid miner web search result: {e}")
        return False
    return True
