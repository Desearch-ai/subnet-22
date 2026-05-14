from typing import List

import numpy as np

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import (
    check_markdown_structure,
    extract_markdown_links,
    collect_summary_sources,
    normalize_source_url,
)

MAX_PENALTY = 1.0


class SummaryStructurePenaltyModel(BasePenaltyModel):
    """Penalize summaries with bad markdown, missing links, or links not in the
    miner's own returned sources. Pure code — no LLM."""

    is_deep = False

    def __init__(self, max_penalty: float = MAX_PENALTY, neuron: AbstractNeuron = None):
        super().__init__(max_penalty, neuron)

    @property
    def name(self) -> str:
        return PenaltyModelType.summary_structure_penalty.value

    async def calculate_penalties(
        self,
        responses: List[ScraperStreamingSynapse],
        additional_params=None,
    ) -> np.ndarray:
        penalties = np.zeros(len(responses), dtype=np.float32)

        for i, response in enumerate(responses):
            if not isinstance(response, ScraperStreamingSynapse):
                continue
            if response.result_type == ResultType.ONLY_LINKS:
                continue

            summary = (response.texts or {}).get(
                ScraperTextRole.FINAL_SUMMARY.value, ""
            )

            ok_structure, _ = check_markdown_structure(summary)
            if not ok_structure:
                penalties[i] = self.max_penalty
                continue

            links = [url for _, url in extract_markdown_links(summary)]
            if not links:
                penalties[i] = self.max_penalty
                continue

            sources = collect_summary_sources(response)
            if any(normalize_source_url(link) not in sources for link in links):
                penalties[i] = self.max_penalty

        return penalties
