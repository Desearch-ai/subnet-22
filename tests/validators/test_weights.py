import asyncio
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from neurons.validators.ledger import Ledger
from neurons.validators.validator import Validator
from neurons.validators.weights import (
    EMISSION_CONTROL_HOTKEY,
    POOLS,
    set_weights,
    weights_from_shares,
)
from tests.local_http import serving

BURN = EMISSION_CONTROL_HOTKEY
CRAWL = POOLS["crawl"]


def test_the_crawl_pool_is_split_by_share_and_the_rest_is_burned():
    weights = weights_from_shares(
        ["m1", BURN, "m2", "m3"], {"crawl": {"m1": 0.75, "m2": 0.25}}
    )

    assert weights[1] == pytest.approx(1 - CRAWL)
    assert weights[0] == pytest.approx(CRAWL * 0.75)
    assert weights[2] == pytest.approx(CRAWL * 0.25)
    assert weights[3] == 0
    assert weights.sum() == pytest.approx(1.0)


def test_a_burn_hotkey_at_uid_zero_still_burns():
    weights = weights_from_shares([BURN, "m1"], {"crawl": {"m1": 1.0}})

    assert list(weights) == pytest.approx([1 - CRAWL, CRAWL])


def test_hotkeys_off_the_metagraph_and_empty_shares_are_dropped():
    weights = weights_from_shares(
        [BURN, "m1", "m2"], {"crawl": {"m1": 0.5, "gone": 0.5, "m2": 0.0}}
    )

    assert list(weights) == pytest.approx([1 - CRAWL, CRAWL, 0.0])


def test_the_burn_hotkey_earns_no_share():
    weights = weights_from_shares([BURN, "m1"], {"crawl": {BURN: 0.5, "m1": 0.5}})

    assert list(weights) == pytest.approx([1 - CRAWL, CRAWL])


def test_a_pool_nobody_earned_goes_to_the_burn_hotkey():
    assert list(weights_from_shares(["m1", BURN], {})) == pytest.approx([0.0, 1.0])
    unknown_pool = weights_from_shares(["m1", BURN], {"translate": {"m1": 1.0}})
    assert list(unknown_pool) == pytest.approx([0.0, 1.0])


def test_crawl_pays_half_and_embedding_nothing_until_it_opens():
    weights = weights_from_shares(
        ["crawler", "embedder", BURN],
        {"crawl": {"crawler": 1.0}, "embed": {"embedder": 1.0}},
    )

    assert list(weights) == pytest.approx([0.5, 0.0, 0.5])


def test_every_pool_pays_its_part(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.weights.POOLS", {"crawl": 0.3, "embed": 0.2}
    )

    weights = weights_from_shares(
        ["crawler", "embedder", "both", BURN],
        {
            "crawl": {"crawler": 0.5, "both": 0.5},
            "embed": {"embedder": 0.5, "both": 0.5},
        },
    )

    assert list(weights) == pytest.approx([0.15, 0.1, 0.25, 0.5])


def test_without_the_burn_hotkey_the_pools_keep_their_proportions():
    weights = weights_from_shares(["m1", "m2"], {"crawl": {"m1": 0.2, "m2": 0.6}})

    assert list(weights / weights.sum()) == pytest.approx([0.25, 0.75])
    assert not weights_from_shares(["m1"], {}).any()


def weights_from_api(hotkeys, shares=None, fail=False, ledger=None):
    async def respond(request: web.Request) -> web.Response:
        if fail:
            return web.Response(status=503)
        if request.path == "/v1/health":
            return web.json_response({"coverage": {"lazy": {"eligible": False}}})
        assert request.path == "/v1/shares"
        return web.json_response({"window_hours": 24, "pools": {"crawl": shares or {}}})

    async def run():
        async with serving(respond) as api, aiohttp.ClientSession() as http:
            made = Validator.__new__(Validator)
            made.config = SimpleNamespace(neuron=SimpleNamespace(task_api_url=api))
            made.metagraph = SimpleNamespace(hotkeys=hotkeys)
            made.http = http
            made.ledger = ledger or Ledger(":memory:")
            return await made.weights()

    return asyncio.run(run())


def test_the_validator_weights_the_metagraph_by_share():
    weights = weights_from_api(["m1", BURN], {"m1": 1.0})

    assert isinstance(weights, np.ndarray)
    assert list(weights) == pytest.approx([CRAWL, 1 - CRAWL])


def test_an_unreachable_task_api_keeps_the_last_weights():
    assert weights_from_api(["m1", BURN], fail=True) is None


def test_set_weights_submits_the_processed_weights(monkeypatch):
    monkeypatch.setattr("neurons.validators.weights.SET_WEIGHTS_RETRY_S", 0)
    sent = []

    class FakeSubtensor:
        async def min_allowed_weights(self, netuid):
            return 1

        async def max_weight_limit(self, netuid):
            return 1.0

        async def set_weights(self, **call):
            sent.append(call)
            return len(sent) > 1, "ok" if len(sent) > 1 else "busy"

    weights = weights_from_shares(["m1", BURN, "m2"], {"crawl": {"m1": 0.5, "m2": 0.5}})
    neuron = SimpleNamespace(
        config=SimpleNamespace(netuid=22),
        metagraph=SimpleNamespace(uids=np.array([0, 1, 2]), n=3),
        subtensor=FakeSubtensor(),
        wallet="wallet",
    )

    assert asyncio.run(set_weights(neuron, weights))
    assert len(sent) == 2, "a failed attempt is retried"
    submitted = dict(
        zip(sent[-1]["uids"].tolist(), sent[-1]["weights"].tolist(), strict=True)
    )
    assert submitted[1] == pytest.approx((1 - CRAWL) / (CRAWL / 2) * submitted[0])
    assert submitted[0] == pytest.approx(submitted[2])


def test_non_finite_shares_keep_the_last_weights():
    assert weights_from_api(["m1", BURN], {"m1": float("inf")}) is None
    assert weights_from_api(["m1", BURN], {"m1": float("nan")}) is None


def test_a_validator_that_cannot_check_tasks_sets_no_weights():
    made = Validator.__new__(Validator)
    made.config = SimpleNamespace(neuron=SimpleNamespace(disable_set_weights=False))
    made.scrapingdog_key = ""
    assert not made.should_set_weights()

    made.scrapingdog_key = "key"
    made.crawl_checker = SimpleNamespace(
        trouble="the provider failed on 3 tasks in a row"
    )
    assert not made.should_set_weights()

    made.crawl_checker.trouble = None
    assert made.should_set_weights()


def test_weights_come_from_the_validators_own_verdicts_once_it_has_any():
    ledger = Ledger(":memory:")
    ledger.record("t1", "crawl", "m1", "pass", 90, 100, 100)
    ledger.record("t2", "crawl", "m2", "pass", 10, 100, 100)
    ledger.record("t3", "crawl", "lazy", "pass", 50, 100, 100)
    ledger.record("t4", "crawl", "short", "pass", 40, 100, 80)

    weights = weights_from_api(
        ["m1", "m2", "lazy", "short", BURN], {"m1": 1.0}, ledger=ledger
    )

    assert list(weights) == pytest.approx(
        [CRAWL * 0.9, CRAWL * 0.1, 0.0, 0.0, 1 - CRAWL]
    )


def test_a_verdict_ledger_windows_and_gates_its_shares():
    ledger = Ledger(":memory:")
    ledger.record("old", "crawl", "m1", "pass", 100, 100, 100, at=1.0)
    ledger.record("new", "crawl", "m2", "pass", 60, 100, 100)
    ledger.record("gated", "crawl", "m3", "pass", 60, 100, 84)
    ledger.record("failed", "crawl", "m4", "fail", 0, 100, 100)

    assert ledger.shares() == {"crawl": {"m2": 1.0}}
    assert ledger.count() == 3
