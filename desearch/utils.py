import re
import os
import math
from pydantic import ValidationError
import base64
import asyncio
import copy
import torch
import requests
import bittensor as bt
import aiohttp
from desearch.redis.utils import save_moving_averaged_scores
from . import client
import html
import unicodedata
from desearch.protocol import (
    Model,
    TwitterScraperTweet,
    WebSearchResult,
)
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from typing import List
from desearch.services.twitter_utils import TwitterUtils
from neurons.validators.env import EXPECTED_ACCESS_KEY, PORT


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


async def call_chutes(messages, temperature, model, seed=1234, response_format=None):
    api_key = os.environ.get("CHUTES_API_TOKEN")

    if not api_key:
        bt.logging.warning("Please set the CHUTES_API_TOKEN environment variable.")
        return None

    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "seed": seed,
    }

    for attempt in range(2):
        bt.logging.trace(
            f"Calling chutes. Temperature = {temperature}, Model = {model}, Seed = {seed},  Messages = {messages}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]

        except Exception as e:
            bt.logging.error(f"Error when calling chutes: {e}")
            await asyncio.sleep(0.5)

    return None


async def call_openai(messages, temperature, model, seed=1234, response_format=None):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        bt.logging.warning("Please set the OPENAI_API_KEY environment variable.")
        return None

    for attempt in range(2):
        bt.logging.trace(
            f"Calling Openai. Temperature = {temperature}, Model = {model}, Seed = {seed},  Messages = {messages}"
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                response_format=response_format,
            )
            response = response.choices[0].message.content
            bt.logging.trace(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {e}")
            await asyncio.sleep(0.5)

    return None


# Github unauthorized rate limit of requests per hour is 60. Authorized is 5000.
def get_version(line_number=22):
    url = f"https://api.github.com/repos/datura-ai/desearch/contents/datura/__init__.py"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.json()["content"]
        decoded_content = base64.b64decode(content).decode("utf-8")
        lines = decoded_content.split("\n")
        if line_number <= len(lines):
            version_line = lines[line_number - 1]
            version_match = re.search(r'__version__ = "(.*?)"', version_line)
            if version_match:
                return version_match.group(1)
            else:
                raise Exception("Version information not found in the specified line")
        else:
            raise Exception("Line number exceeds file length")
    else:
        bt.logging.error("github api call failed")
        return None


async def resync_metagraph(self):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")

    # Copies state of metagraph before syncing.
    previous_metagraph = copy.deepcopy(self.metagraph)

    try:
        # Sync the metagraph.
        await self.metagraph.sync(subtensor=self.subtensor)
    except Exception as e:
        bt.logging.error(f"Error in resync_metagraph: {e}")

        await self.subtensor.close()
        self.subtensor = bt.AsyncSubtensor(config=self.config)
        self.metagraph = await self.subtensor.metagraph(self.config.netuid)

    # Check if the metagraph axon info has changed.
    if previous_metagraph.axons == self.metagraph.axons:
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
        # Update the size of the moving average scores.
        new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
        min_len = min(len(self.hotkeys), len(self.moving_averaged_scores))
        new_moving_average[:min_len] = self.moving_averaged_scores[:min_len]
        self.moving_averaged_scores = new_moving_average

    bt.logging.info("Saving moving averaged scores to Redis after metagraph update")
    await save_moving_averaged_scores(self.moving_averaged_scores)

    # Update the hotkeys.
    self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)


async def save_logs(logs, netuid):
    logging_endpoint_url = None

    if netuid == 22:
        logging_endpoint_url = "https://logs.desearch.ai"
    else:
        logging_endpoint_url = "https://logs-dev.desearch.ai"

    try:
        timeout = aiohttp.ClientTimeout(total=600)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await session.post(
                logging_endpoint_url,
                json={
                    "logs": logs,
                },
            )
            bt.logging.debug(f"Executed save_logs, got result: {await result.text()}")
    except aiohttp.ClientError as e:
        bt.logging.error(f"Error in save_logs: {e}")
    except Exception as e:
        bt.logging.error(f"Unexpected error in save_logs: {e}")


async def save_logs_in_chunks_for_deep_research(
    self,
    responses,
    uids,
    rewards,
    content_rewards,
    data_rewards,
    logical_coherence_rewards,
    source_links_rewards,
    system_message_rewards,
    performance_rewards,
    original_content_rewards,
    original_data_rewards,
    original_logical_coherence_rewards,
    original_source_links_rewards,
    original_system_message_rewards,
    original_performance_rewards,
    content_scores,
    data_scores,
    logical_coherence_scores,
    source_links_scores,
    system_message_scores,
    weights,
    neuron,
    netuid,
    organic_penalties,
    query_type,
):
    try:
        logs = [
            {
                "prompt": response.prompt,
                "score": reward,
                "content_score": content_reward,
                "data_score": data_reward,
                "logical_coherence_score": logical_coherence_reward,
                "source_links_score": source_links_reward,
                "system_message_score": system_message_reward,
                "performance_score": performance_reward,
                "original_content_score": original_content_reward,
                "original_data_score": original_data_reward,
                "original_logical_coherence_score": original_logical_coherence_reward,
                "original_source_links_score": original_source_links_reward,
                "original_system_message_score": original_system_message_reward,
                "original_performance_score": original_performance_reward,
                "content_scores": content_score,
                "data_scores": data_score,
                "logical_coherence_scores": logical_coherence_score,
                "source_links_scores": source_links_score,
                "system_messagee_scores": system_message_score,
                "report": response.report,
                "weight": weights.get(str(uid)),
                "miner": {
                    "uid": uid,
                    "hotkey": response.axon.hotkey,
                    "coldkey": next(
                        (
                            axon.coldkey
                            for axon in self.metagraph.axons
                            if axon.hotkey == response.axon.hotkey
                        ),
                        None,  # Provide a default value here, such as None or an appropriate placeholder
                    ),
                },
                "validator": {
                    "uid": neuron.uid,
                    "hotkey": neuron.dendrite.keypair.ss58_address,
                    "coldkey": next(
                        (
                            nr.coldkey
                            for nr in self.metagraph.neurons
                            if nr.hotkey == neuron.dendrite.keypair.ss58_address
                        ),
                        None,
                    ),
                    "ip": neuron.dendrite.external_ip,
                    "port": PORT,
                    "access_key": EXPECTED_ACCESS_KEY,
                },
                "tools": response.tools,
                "date_filter": {
                    "start_date": response.start_date,
                    "end_date": response.end_date,
                    "date_filter_type": response.date_filter_type,
                },
                "time": response.dendrite.process_time,
                "organic_penalty": organic_penalty,
                "max_execution_time": response.max_execution_time,
                "query_type": query_type,
                "language": response.language,
                "region": response.region,
            }
            for response, uid, reward, content_reward, data_reward, logical_coherence_reward, source_links_reward, system_message_reward, performance_reward, original_content_reward, original_data_reward, original_logical_coherence_reward, original_source_links_reward, original_system_message_reward, original_performance_reward, content_score, data_score, logical_coherence_score, source_links_score, system_message_score, organic_penalty in zip(
                responses,
                uids.tolist(),
                rewards.tolist(),
                content_rewards.tolist(),
                data_rewards.tolist(),
                logical_coherence_rewards.tolist(),
                source_links_rewards.tolist(),
                system_message_rewards.tolist(),
                performance_rewards.tolist(),
                original_content_rewards,
                original_data_rewards,
                original_logical_coherence_rewards,
                original_source_links_rewards,
                original_system_message_rewards,
                original_performance_rewards,
                content_scores,
                data_scores,
                logical_coherence_scores,
                source_links_scores,
                system_message_scores,
                organic_penalties,
            )
        ]

        for idx, log in enumerate(logs, start=1):
            bt.logging.debug(
                f"Log Entry {idx} - max_execution_time: {log.get('max_execution_time')}, "
                f"query_type: {log.get('query_type')}, model: {log.get('model')}, "
                f"language: {log.get('language')}, region: {log.get('region')}, "
                f"max_items: {log.get('max_items')}"
            )
        chunk_size = 20

        log_chunks = [logs[i : i + chunk_size] for i in range(0, len(logs), chunk_size)]

        for chunk in log_chunks:
            await save_logs(
                logs=chunk,
                netuid=netuid,
            )
    except Exception as e:
        bt.logging.error(f"Error in save_logs_in_chunks_for_deep_research: {e}")
        raise e


async def save_logs_in_chunks(
    self,
    responses,
    uids,
    rewards,
    summary_rewards,
    twitter_rewards,
    search_rewards,
    performance_rewards,
    original_summary_rewards,
    original_twitter_rewards,
    original_search_rewards,
    original_performance_rewards,
    tweet_scores,
    search_scores,
    summary_link_scores,
    weights,
    neuron,
    netuid,
    organic_penalties,
    query_type,
):
    try:
        logs = [
            {
                "prompt": response.prompt,
                "completion": response.completion,
                # "prompt_analysis": response.prompt_analysis.dict(),
                "data": response.miner_tweets,
                "score": reward,
                "summary_score": summary_reward,
                "twitter_score": twitter_reward,
                "search_score": search_reward,
                "performance_score": performance_reward,
                "original_summary_score": original_summary_reward,
                "original_twitter_score": original_twitter_reward,
                "original_search_score": original_search_reward,
                "original_performance_score": original_performance_reward,
                "tweet_scores": tweet_score,
                "link_scores": search_score,
                "summary_link_scores": summary_link_score,
                "search_results": {
                    "google": [item.model_dump() for item in response.search_results],
                    "wikipedia": [
                        item.model_dump() for item in response.wikipedia_search_results
                    ],
                    "youtube": [
                        item.model_dump() for item in response.youtube_search_results
                    ],
                    "arxiv": [
                        item.model_dump() for item in response.arxiv_search_results
                    ],
                    "reddit": [
                        item.model_dump() for item in response.reddit_search_results
                    ],
                    "hacker_news": [
                        item.model_dump()
                        for item in response.hacker_news_search_results
                    ],
                },
                "texts": response.texts,
                "validator_tweets": [
                    val_tweet.dict() for val_tweet in response.validator_tweets
                ],
                "validator_links": [
                    {
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "link": item.get("link"),
                    }
                    for item in response.validator_links
                ],
                "search_completion_links": response.search_completion_links,
                "twitter_completion_links": response.completion_links,
                "weight": weights.get(str(uid)),
                "miner": {
                    "uid": uid,
                    "hotkey": response.axon.hotkey,
                    "coldkey": next(
                        (
                            axon.coldkey
                            for axon in self.metagraph.axons
                            if axon.hotkey == response.axon.hotkey
                        ),
                        None,  # Provide a default value here, such as None or an appropriate placeholder
                    ),
                },
                "validator": {
                    "uid": neuron.uid,
                    "hotkey": neuron.dendrite.keypair.ss58_address,
                    "coldkey": next(
                        (
                            nr.coldkey
                            for nr in self.metagraph.neurons
                            if nr.hotkey == neuron.dendrite.keypair.ss58_address
                        ),
                        None,
                    ),
                    "ip": neuron.dendrite.external_ip,
                    "port": PORT,
                    "access_key": EXPECTED_ACCESS_KEY,
                },
                "tools": response.tools,
                "date_filter": {
                    "start_date": response.start_date,
                    "end_date": response.end_date,
                    "date_filter_type": response.date_filter_type,
                },
                "time": response.dendrite.process_time,
                "organic_penalty": organic_penalty,
                "max_execution_time": response.max_execution_time,
                "query_type": query_type,
                "model": response.model,
                "language": response.language,
                "region": response.region,
                "max_items": response.max_items,
                "result_type": response.result_type,
            }
            for response, uid, reward, summary_reward, twitter_reward, search_reward, performance_reward, original_summary_reward, original_twitter_reward, original_search_reward, original_performance_reward, tweet_score, search_score, summary_link_score, organic_penalty in zip(
                responses,
                uids.tolist(),
                rewards.tolist(),
                summary_rewards.tolist(),
                twitter_rewards.tolist(),
                search_rewards.tolist(),
                performance_rewards.tolist(),
                original_summary_rewards,
                original_twitter_rewards,
                original_search_rewards,
                original_performance_rewards,
                tweet_scores,
                search_scores,
                summary_link_scores,
                organic_penalties,
            )
        ]

        for idx, log in enumerate(logs, start=1):
            bt.logging.debug(
                f"Log Entry {idx} - max_execution_time: {log.get('max_execution_time')}, "
                f"query_type: {log.get('query_type')}, model: {log.get('model')}, "
                f"language: {log.get('language')}, region: {log.get('region')}, "
                f"max_items: {log.get('max_items')}"
                f"result_type: {log.get('result_type')}"
            )
        chunk_size = 10

        log_chunks = [logs[i : i + chunk_size] for i in range(0, len(logs), chunk_size)]

        for chunk in log_chunks:
            await save_logs(
                logs=chunk,
                netuid=netuid,
            )
    except Exception as e:
        bt.logging.error(f"Error in save_logs_in_chunks: {e}")
        raise e


async def save_logs_in_chunks_for_basic(
    self,
    responses,
    uids,
    rewards,
    twitter_rewards,
    performance_rewards,
    original_twitter_rewards,
    original_performance_rewards,
    tweet_scores,
    weights,
    neuron,
    netuid,
    organic_penalties,
):
    try:
        logs = [
            {
                "prompt": response.query,
                "result": [
                    {
                        "id": tweet.id,
                        "text": tweet.text,
                        "user": tweet.user.dict() if tweet.user else None,
                        "created_at": tweet.created_at,
                        "retweet_count": tweet.retweet_count,
                        "like_count": tweet.like_count,
                        "reply_count": tweet.reply_count,
                        "url": tweet.url,
                    }
                    for tweet in response.results
                ],
                "score": reward,
                "twitter_score": twitter_reward,
                "performance_score": performance_reward,
                "original_twitter_score": original_twitter_reward,
                "original_performance_score": original_performance_reward,
                "tweet_scores": tweet_score,
                "texts": [tweet.text for tweet in response.results if tweet.text],
                "validator_tweets": [
                    val_tweet.dict() for val_tweet in response.validator_tweets
                ],
                "weight": weights.get(str(uid)),
                "miner": {
                    "uid": uid,
                    "hotkey": response.axon.hotkey,
                    "coldkey": next(
                        (
                            axon.coldkey
                            for axon in self.metagraph.axons
                            if axon.hotkey == response.axon.hotkey
                        ),
                        None,  # Provide a default value here, such as None or an appropriate placeholder
                    ),
                },
                "validator": {
                    "uid": neuron.uid,
                    "hotkey": neuron.dendrite.keypair.ss58_address,
                    "coldkey": next(
                        (
                            nr.coldkey
                            for nr in self.metagraph.neurons
                            if nr.hotkey == neuron.dendrite.keypair.ss58_address
                        ),
                        None,
                    ),
                },
                "date_filter": {
                    "start_date": response.start_date,
                    "end_date": response.end_date,
                },
                "time": response.dendrite.process_time,
                "organic_penalty": organic_penalty,
                "max_execution_time": response.max_execution_time,
                "model": response.model,
                "language": response.lang,
            }
            for response, uid, reward, twitter_reward, performance_reward, original_twitter_reward, original_performance_reward, tweet_score, organic_penalty in zip(
                responses,
                uids.tolist(),
                rewards.tolist(),
                twitter_rewards.tolist(),
                performance_rewards.tolist(),
                original_twitter_rewards,
                original_performance_rewards,
                tweet_scores,
                organic_penalties,
            )
        ]

        # Divide logs into chunks for saving
        chunk_size = 20
        log_chunks = [logs[i : i + chunk_size] for i in range(0, len(logs), chunk_size)]

        for chunk in log_chunks:
            await save_logs(
                logs=chunk,
                netuid=netuid,
            )
    except Exception as e:
        bt.logging.error(f"Error in save_logs_in_chunks_for_basic: {e}")
        raise e


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
