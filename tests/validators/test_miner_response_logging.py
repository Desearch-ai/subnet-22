from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch

from neurons.validators.clients.miner_response_logger import build_log_entry, submit_logs
from neurons.validators.scrapers.advanced_scraper_validator import AdvancedScraperValidator
from neurons.validators.scrapers.web_scraper_validator import WebScraperValidator
from neurons.validators.scrapers.x_scraper_validator import XScraperValidator


def _fake_owner():
    return SimpleNamespace(
        config=SimpleNamespace(netuid=22),
        validator_identity={
            "uid": 7,
            "hotkey": "validator-hotkey",
            "coldkey": "validator-coldkey",
            "netuid": 22,
        },
        utility_api=SimpleNamespace(save_logs=AsyncMock()),
    )


def _fake_response():
    return SimpleNamespace(
        axon=SimpleNamespace(
            hotkey="miner-hotkey",
            coldkey="miner-coldkey",
        ),
        dendrite=SimpleNamespace(status_code=200, process_time="1.5"),
    )


@pytest.mark.asyncio
async def test_x_search_logs_selected_uid():
    validator = object.__new__(XScraperValidator)
    validator.neuron = _fake_owner()
    response = _fake_response()
    validator.call_miner = AsyncMock(
        return_value=(
            response,
            42,
            SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        )
    )

    with (
        patch(
            "neurons.validators.x_scraper_validator.build_log_entry",
            return_value={"ok": True},
        ) as build_log_entry,
        patch(
            "neurons.validators.x_scraper_validator.submit_logs_best_effort"
        ) as submit_logs_best_effort,
    ):
        items = [item async for item in validator.x_search({"query": "bittensor"})]

    assert items == [response]
    build_log_entry.assert_called_once()
    assert build_log_entry.call_args.kwargs["miner_uid"] == 42
    submit_logs_best_effort.assert_called_once_with(validator.neuron, [{"ok": True}])


@pytest.mark.asyncio
async def test_web_organic_logs_selected_uid():
    validator = object.__new__(WebScraperValidator)
    validator.neuron = _fake_owner()
    response = _fake_response()
    validator.call_miner = AsyncMock(
        return_value=(
            response,
            84,
            SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        )
    )

    with (
        patch(
            "neurons.validators.web_scraper_validator.build_log_entry",
            return_value={"ok": True},
        ) as build_log_entry,
        patch(
            "neurons.validators.web_scraper_validator.submit_logs_best_effort"
        ) as submit_logs_best_effort,
    ):
        items = [item async for item in validator.organic({"query": "tao"})]

    assert items == [response]
    build_log_entry.assert_called_once()
    assert build_log_entry.call_args.kwargs["miner_uid"] == 84
    submit_logs_best_effort.assert_called_once_with(validator.neuron, [{"ok": True}])


@pytest.mark.asyncio
async def test_ai_organic_logs_after_stream_finishes():
    validator = object.__new__(AdvancedScraperValidator)
    validator.neuron = _fake_owner()
    validator.language = "en"
    validator.region = "us"
    validator.date_filter = "qdr:w"
    final_synapse = _fake_response()

    async def fake_stream():
        yield "chunk-1"
        yield final_synapse

    validator.call_miner = AsyncMock(
        return_value=(
            fake_stream(),
            torch.tensor([55]),
            0.0,
            SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        )
    )

    with (
        patch(
            "neurons.validators.advanced_scraper_validator.build_log_entry",
            return_value={"ok": True},
        ) as build_log_entry,
        patch(
            "neurons.validators.advanced_scraper_validator.submit_logs_best_effort"
        ) as submit_logs_best_effort,
    ):
        chunks = [
            item
            async for item in validator.organic(
                {"content": "hello", "tools": ["Web Search"]},
                uid=55,
            )
        ]

    assert chunks == ["chunk-1"]
    build_log_entry.assert_called_once()
    assert build_log_entry.call_args.kwargs["miner_uid"] == 55
    submit_logs_best_effort.assert_called_once_with(validator.neuron, [{"ok": True}])


@pytest.mark.asyncio
async def test_submit_logs_swallows_utility_api_failures():
    owner = _fake_owner()
    owner.utility_api.save_logs.side_effect = RuntimeError("boom")

    await submit_logs(owner, [{"log": 1}])


def test_build_log_entry_excludes_html_fields_from_response_payload():
    owner = _fake_owner()
    response = SimpleNamespace(
        prompt="what is bittensor",
        validator_links=[
            {
                "link": "https://example.com",
                "title": "Example",
                "snippet": "Summary",
                "html_content": "<html>big payload</html>",
                "html_text": "big payload",
            }
        ],
        axon=SimpleNamespace(
            hotkey="miner-hotkey",
            coldkey="miner-coldkey",
        ),
        dendrite=SimpleNamespace(status_code=200, process_time="1.5"),
    )

    log_entry = build_log_entry(
        owner=owner,
        search_type="web_search",
        query_kind="scoring",
        response=response,
    )

    assert "html_content" not in log_entry["response_payload"]["validator_links"][0]
    assert "html_text" not in log_entry["response_payload"]["validator_links"][0]
    assert response.validator_links[0]["html_content"] == "<html>big payload</html>"
    assert response.validator_links[0]["html_text"] == "big payload"


@pytest.mark.asyncio
async def test_x_post_by_id_logs_organic():
    validator = object.__new__(XScraperValidator)
    validator.neuron = _fake_owner()
    validator.max_execution_time = 10
    synapse = SimpleNamespace(
        results=[{"id": "123"}],
        axon=SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        dendrite=SimpleNamespace(status_code=200, process_time="1.5"),
    )
    call = AsyncMock(return_value=synapse)
    validator.neuron.get_random_miner = AsyncMock(
        return_value=(
            42,
            SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        )
    )
    validator.neuron.dendrites = iter([SimpleNamespace(call=call)])

    with (
        patch(
            "neurons.validators.x_scraper_validator.build_log_entry",
            return_value={"ok": True},
        ) as build_log_entry,
        patch(
            "neurons.validators.x_scraper_validator.submit_logs_best_effort"
        ) as submit_logs_best_effort,
    ):
        results = await validator.x_post_by_id("123")

    assert results == [{"id": "123"}]
    assert build_log_entry.call_args.kwargs["miner_uid"] == 42
    submit_logs_best_effort.assert_called_once_with(validator.neuron, [{"ok": True}])


@pytest.mark.asyncio
async def test_x_posts_by_urls_logs_organic():
    validator = object.__new__(XScraperValidator)
    validator.neuron = _fake_owner()
    validator.calc_max_execution_time = lambda count: 10
    synapse = SimpleNamespace(
        results=[{"id": "abc"}],
        axon=SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        dendrite=SimpleNamespace(status_code=200, process_time="1.5"),
    )
    call = AsyncMock(return_value=synapse)
    validator.neuron.get_random_miner = AsyncMock(
        return_value=(
            99,
            SimpleNamespace(hotkey="miner-hotkey", coldkey="miner-coldkey"),
        )
    )
    validator.neuron.dendrites = iter([SimpleNamespace(call=call)])

    with (
        patch(
            "neurons.validators.x_scraper_validator.build_log_entry",
            return_value={"ok": True},
        ) as build_log_entry,
        patch(
            "neurons.validators.x_scraper_validator.submit_logs_best_effort"
        ) as submit_logs_best_effort,
    ):
        results = await validator.x_posts_by_urls(
            ["https://x.com/a/status/1", "https://x.com/b/status/2"]
        )

    assert results == [{"id": "abc"}]
    assert build_log_entry.call_args.kwargs["miner_uid"] == 99
    submit_logs_best_effort.assert_called_once_with(validator.neuron, [{"ok": True}])
