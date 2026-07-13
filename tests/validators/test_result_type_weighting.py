from types import SimpleNamespace

import numpy as np
import pytest

from desearch.protocol import ResultType
from neurons.validators.scrapers.advanced_scraper_validator import (
    AdvancedScraperValidator,
)


def _validator():
    v = object.__new__(AdvancedScraperValidator)
    v.content_weight = 0.60
    v.summary_relevance_weight = 0.40
    v.reward_weights = np.array([0.60, 0.40], dtype=np.float32)
    return v


def _resp(result_type):
    return SimpleNamespace(result_type=result_type)


def test_only_links_puts_full_weight_on_content():
    m = _validator().compute_reward_weights_matrix([_resp(ResultType.ONLY_LINKS)])
    assert m[0].tolist() == [1.0, 0.0]


def test_only_links_matches_string_result_type():
    m = _validator().compute_reward_weights_matrix([_resp("ONLY_LINKS")])
    assert m[0].tolist() == [1.0, 0.0]


def test_summary_type_keeps_split_weights():
    m = _validator().compute_reward_weights_matrix(
        [_resp(ResultType.LINKS_WITH_FINAL_SUMMARY)]
    )
    assert m[0] == pytest.approx([0.60, 0.40])


def test_mixed_batch_weights_row_aligned():
    v = _validator()
    m = v.compute_reward_weights_matrix(
        [
            _resp(ResultType.LINKS_WITH_FINAL_SUMMARY),
            _resp(ResultType.ONLY_LINKS),
            _resp(None),
        ]
    )
    assert m[0] == pytest.approx([0.60, 0.40])
    assert m[1].tolist() == [1.0, 0.0]
    assert m[2] == pytest.approx([0.60, 0.40])


def test_free_summary_no_longer_lifts_only_links_over_the_gate():
    v = _validator()
    content, free_summary = 0.5, 1.0
    m = v.compute_reward_weights_matrix([_resp(ResultType.ONLY_LINKS)])[0]
    assert m[0] * content + m[1] * free_summary == pytest.approx(0.5)


if __name__ == "__main__":
    test_only_links_puts_full_weight_on_content()
    test_only_links_matches_string_result_type()
    test_summary_type_keeps_split_weights()
    test_mixed_batch_weights_row_aligned()
    test_free_summary_no_longer_lifts_only_links_over_the_gate()
    print("ok")
