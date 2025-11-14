import bittensor as bt

from desearch.protocol import ScoringModel
from desearch.synapse import collect_responses
from desearch.utils import call_chutes, call_openai
from neurons.validators.utils.prompts import ScoringPrompt


class RewardLLM:
    def __init__(self, scoring_model: ScoringModel = ScoringModel.OPENAI_GPT4_MINI):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.pipe = None
        self.scoring_prompt = ScoringPrompt()
        self.scoring_model = scoring_model

    async def get_scores(self, messages):
        try:
            query_tasks = []

            for message_dict in messages:
                ((key, message_list),) = message_dict.items()

                async def query_llm(message):
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
                        bt.logging.error(f"Error sending message to OpenAI: {e}")
                        return ""  # Return an empty string to indicate failure

                task = query_llm(message_list)
                query_tasks.append(task)

            query_responses = await collect_responses(query_tasks, group_size=100)

            result = {}

            for response, message_dict in zip(query_responses, messages):
                if isinstance(response, Exception):
                    bt.logging.error(f"Query failed with exception: {response}")
                    response = (
                        ""  # Replace the exception with an empty string in the result
                    )
                ((key, message_list),) = message_dict.items()
                result[key] = response

            return result
        except Exception as e:
            bt.logging.error(f"Error processing OpenAI queries: {e}")
            return None

    async def llm_processing(self, messages):
        # Initialize score_responses as an empty dictionary to hold the scoring results
        score_responses = {}

        current_score_responses = await self.get_scores(messages=messages)

        if current_score_responses:
            # Update the score_responses with the new scores
            score_responses.update(current_score_responses)
        else:
            bt.logging.info("Scoring failed or returned no results.")

        return score_responses
