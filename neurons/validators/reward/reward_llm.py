from typing import List
import torch
import random
import requests
import os
import asyncio
import bittensor as bt
import re
import time
from desearch.utils import call_openai, call_chutes
from transformers import AutoTokenizer, AutoModelForCausalLM
from desearch.synapse import collect_responses
from desearch.protocol import ScoringModel

from neurons.validators.utils.prompts import ScoringPrompt

from enum import Enum
import torch
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY", "hello")
URL_SUBNET_18 = os.environ.get("URL_SUBNET_18")


class ScoringSource(Enum):
    Subnet18 = 1
    OpenAI = 2
    LocalLLM = 3
    LocalZephyr = 4


class RewardLLM:
    def __init__(self, scoring_model: ScoringModel = ScoringModel.OPENAI_GPT4_MINI):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.pipe = None
        self.scoring_prompt = ScoringPrompt()
        self.scoring_model = scoring_model

    def init_tokenizer(self, device, model_name):
        # https://huggingface.co/VMware/open-llama-7b-open-instruct
        # Fast tokenizer results in incorrect encoding, set the use_fast = False parameter.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Generative default expects most recent token on right-hand side with padding on left.
        # https://github.com/huggingface/transformers/pull/10552
        tokenizer.padding_side = "left"

        # Check if the device is CPU or CUDA and set the precision accordingly
        torch_dtype = torch.float32 if device == "cpu" else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        return tokenizer, model

    def init_pipe_zephyr(self):
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-alpha",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.pipe = pipe
        return pipe

    def clean_text(self, text):
        # Remove newline characters and replace with a space
        text = text.replace("\n", " ")

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Keep hashtags, alphanumeric characters, and spaces
        # Remove other special characters but ensure to keep structured elements like <Question>, <Answer>, etc., intact
        text = re.sub(r"(?<![\w<>#])[^\w\s#<>]+", "", text)

        return text

    def call_to_subnet_18_scoring(self, data):
        start_time = time.time()  # Start timing for execution
        try:
            if not URL_SUBNET_18:
                bt.logging.warning(
                    "Please set the URL_SUBNET_18 environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
                )
                return None

            headers = {
                "access-key": EXPECTED_ACCESS_KEY,
                "Content-Type": "application/json",
            }
            response = requests.post(
                url=f"{URL_SUBNET_18}/text-validator/",
                headers=headers,
                json=data,
                timeout=10 * 60,  # Timeout after 10 minutes
            )  # Using json parameter to automatically set the content-type to application/json

            if response.status_code in [401, 403]:
                bt.logging.error(f"Connection issue with Subnet 18: {response.text}")
                return {}
            if response.status_code != 200:
                bt.logging.error(
                    f"ERROR connect to Subnet 18: Status code: {response.status_code}"
                )
                return None
            execution_time = (
                time.time() - start_time
            ) / 60  # Calculate execution time in minutes
            bt.logging.info(
                f"Subnet 18 scoring call execution time: {execution_time:.2f} minutes"
            )
            return response
        except Exception as e:
            bt.logging.warning(f"Error calling Subnet 18 scoring: {e}")
            return None

    async def get_score_by_openai(self, messages):
        try:
            start_time = time.time()  # Start timing for query execution
            query_tasks = []
            for message_dict in messages:  # Iterate over each dictionary in the list
                ((key, message_list),) = message_dict.items()

                async def query_openai(message):
                    try:
                        if self.scoring_model == ScoringModel.OPENAI_GPT4_MINI:
                            return await call_openai(
                                messages=message,
                                temperature=0.0001,
                                model="gpt-4o-mini",
                            )
                        else:
                            return await call_chutes(
                                messages=message,
                                temperature=0.0001,
                                model=self.scoring_model,
                            )
                    except Exception as e:
                        print(f"Error sending message to OpenAI: {e}")
                        return ""  # Return an empty string to indicate failure

                task = query_openai(message_list)
                query_tasks.append(task)

            query_responses = await collect_responses(query_tasks, group_size=100)
            # query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)

            result = {}
            for response, message_dict in zip(query_responses, messages):
                if isinstance(response, Exception):
                    print(f"Query failed with exception: {response}")
                    response = (
                        ""  # Replace the exception with an empty string in the result
                    )
                ((key, message_list),) = message_dict.items()
                result[key] = response

            execution_time = time.time() - start_time  # Calculate execution time
            # print(f"Execution time for OpenAI queries: {execution_time} seconds")
            return result
        except Exception as e:
            print(f"Error processing OpenAI queries: {e}")
            return None

    async def get_score_by_source(self, messages, source: ScoringSource):
        if source == ScoringSource.Subnet18:
            return self.call_to_subnet_18_scoring(messages)
        else:
            return await self.get_score_by_openai(messages=messages)

    async def llm_processing(self, messages):
        # Initialize score_responses as an empty dictionary to hold the scoring results
        score_responses = {}

        # Define the order of scoring sources to be used
        scoring_sources = [
            ScoringSource.OpenAI,  # Attempt scoring with OpenAI
            # ScoringSource.LocalZephyr,  # Fallback to Local LLM if OpenAI fails
            # ScoringSource.Subnet18,  # First attempt with Subnet 18
        ]

        # Attempt to score messages using the defined sources in order
        for source in scoring_sources:
            # Attempt to score with the current source
            current_score_responses = await self.get_score_by_source(
                messages=messages, source=source
            )
            if current_score_responses:
                # Update the score_responses with the new scores
                score_responses.update(current_score_responses)
            else:
                bt.logging.info(
                    f"Scoring with {source} failed or returned no results. Attempting next source."
                )

        return score_responses
