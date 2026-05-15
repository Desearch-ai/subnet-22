from desearch.protocol import TwitterSearchSynapse
from neurons.validators.penalty.penalty import CheapPenaltyModel, PenaltyModelType
from neurons.validators.utils.response_checks import is_descending_by_created_at


class SortOrderPenaltyModel(CheapPenaltyModel):
    """Penalize Twitter responses with sort=Latest that aren't sorted by
    created_at descending."""

    name = PenaltyModelType.sort_order_penalty.value

    def penalty_for(self, response) -> float:
        if not isinstance(response, TwitterSearchSynapse):
            return 0.0
        if getattr(response, "sort", None) != "Latest":
            return 0.0
        if not is_descending_by_created_at(response.results or []):
            return self.max_penalty
        return 0.0
