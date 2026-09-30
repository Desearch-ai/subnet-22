"""Zero a response whose every source is one the miner could have served itself."""

import asyncio
from typing import List

import numpy as np

from desearch.protocol import ScraperStreamingSynapse
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
from neurons.validators.utils.source_provenance import miner_ip_of, rejected_links

MAX_PENALTY = 1.0


class SourceProvenancePenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.source_provenance_penalty.value

    @staticmethod
    def _links(response: ScraperStreamingSynapse) -> List[str]:
        try:
            links, _ = response.get_links_from_search_results()
            return [link for link in links if link]
        except Exception:
            return []

    async def _penalty_for(self, response) -> float:
        links = self._links(response)
        if not links:
            return 0.0

        rejected = await rejected_links(links, miner_ip_of(response))

        return self.max_penalty if len(rejected) == len(set(links)) else 0.0

    async def calculate_penalties(
        self,
        responses: List[ScraperStreamingSynapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = await asyncio.gather(
            *[self._penalty_for(response) for response in responses]
        )

        return np.array(penalties, dtype=np.float32)
