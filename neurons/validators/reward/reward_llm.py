import os
import time
from enum import Enum

import bittensor as bt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from desearch.protocol import ScoringModel
from desearch.synapse import collect_responses
from desearch.utils import call_chutes, call_openai
from neurons.validators.utils.prompts import ScoringPrompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ScoringSource(Enum):
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
        return await self.get_score_by_openai(messages=messages)

    async def llm_processing(self, messages):
        # Initialize score_responses as an empty dictionary to hold the scoring results
        score_responses = {}

        # Define the order of scoring sources to be used
        scoring_sources = [
            ScoringSource.OpenAI,  # Attempt scoring with OpenAI
            # ScoringSource.LocalZephyr,  # Fallback to Local LLM if OpenAI fails
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
