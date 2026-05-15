from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import (
    check_markdown_structure,
    collect_summary_sources,
    extract_markdown_links,
    normalize_source_url,
)


class SummaryStructurePenaltyModel(CheapPenaltyModel):
    """Penalize summaries with bad markdown, missing links, or links not in the
    miner's own returned sources. Pure code — no LLM."""

    name = PenaltyModelType.summary_structure_penalty.value

    def penalty_for(self, response) -> float:
        if not isinstance(response, ScraperStreamingSynapse):
            return 0.0
        if response.result_type == ResultType.ONLY_LINKS:
            return 0.0

        summary = (response.texts or {}).get(ScraperTextRole.FINAL_SUMMARY.value, "")
        ok_structure, _ = check_markdown_structure(summary)
        if not ok_structure:
            return self.max_penalty

        links = [url for _, url in extract_markdown_links(summary)]
        if not links:
            return self.max_penalty

        sources = collect_summary_sources(response)
        if any(normalize_source_url(link) not in sources for link in links):
            return self.max_penalty
        return 0.0
