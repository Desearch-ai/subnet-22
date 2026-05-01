from typing import List

import tiktoken
import torch

from desearch.protocol import ResultType, ScraperStreamingSynapse
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

MAX_TOKENS_PER_CHUNK = 2
PENALTY_PER_EXCEEDING_TOKEN = 0.01

encoding = tiktoken.get_encoding("o200k_base")


class StreamingPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.streaming_penalty.value

    async def calculate_penalties(
        self,
        responses: List[ScraperStreamingSynapse],
        additional_params=None,
    ) -> torch.FloatTensor:
        accumulated_penalties = torch.zeros(len(responses), dtype=torch.float32)

        for index, response in enumerate(responses):
            if response.result_type == ResultType.ONLY_LINKS:
                continue

            streamed_text_chunks = []

            for chunks in response.text_chunks.values():
                streamed_text_chunks.extend(chunks)

            if not streamed_text_chunks:
                accumulated_penalties[index] = 1
                continue

            token_counts = [
                (len(encoding.encode(chunk)) if chunk is not None else 0)
                for chunk in streamed_text_chunks
            ]

            # Apply penalty for exceeding max tokens per chunk
            for token_count in token_counts:
                if token_count > MAX_TOKENS_PER_CHUNK:
                    penalty = (
                        token_count - MAX_TOKENS_PER_CHUNK
                    ) * PENALTY_PER_EXCEEDING_TOKEN

                    accumulated_penalties[index] = min(
                        1, accumulated_penalties[index] + penalty
                    )

        return accumulated_penalties
