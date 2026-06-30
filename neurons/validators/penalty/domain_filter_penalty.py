from desearch.protocol import ScraperStreamingSynapse
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.web_query_operators import (
    host_in_domains,
    normalize_domains,
)


class DomainFilterPenaltyModel(CheapPenaltyModel):
    name = PenaltyModelType.domain_filter_penalty.value

    def penalty_for(self, response) -> float:
        if not isinstance(response, ScraperStreamingSynapse):
            return 0.0

        include = normalize_domains(response.include_domains)
        exclude = normalize_domains(response.exclude_domains)
        if not include and not exclude:
            return 0.0

        links = self._search_result_links(response)
        if not links:
            return 0.0

        violations = sum(1 for link in links if self._violates(link, include, exclude))
        return min(violations / len(links), self.max_penalty)

    @staticmethod
    def _violates(link: str, include, exclude) -> bool:
        if include and not host_in_domains(link, include):
            return True
        if exclude and host_in_domains(link, exclude):
            return True
        return False

    @staticmethod
    def _search_result_links(response: ScraperStreamingSynapse):
        links = []
        for result in response.search_results or []:
            link = (
                result.get("link")
                if isinstance(result, dict)
                else getattr(result, "link", None)
            )
            if link:
                links.append(link)
        return links
