import asyncio
import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

import bittensor as bt
import numpy as np
from pydantic import BaseModel

REWARD_COMPONENT_NAMES = {
    "ai_search": ["twitter", "search", "summary"],
    "x_search": ["twitter"],
    "web_search": ["search"],
}
_RESPONSE_PAYLOAD_EXCLUDED_KEYS = {"html_content", "html_text"}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            key: to_jsonable(item)
            for key, item in value.model_dump(mode="python").items()
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="python"))
        except TypeError:
            pass
    if hasattr(value, "dict"):
        try:
            return to_jsonable(value.dict())
        except TypeError:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return to_jsonable(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_hotkey_uid(owner, hotkey: Optional[str]) -> Optional[int]:
    if not hotkey:
        return None

    metagraph = getattr(owner, "metagraph", None)
    hotkeys = getattr(metagraph, "hotkeys", None)

    if not hotkeys:
        return None

    try:
        return hotkeys.index(hotkey)
    except ValueError:
        return None


def _find_miner_coldkey(owner, hotkey: Optional[str]) -> Optional[str]:
    if not hotkey:
        return None

    metagraph = getattr(owner, "metagraph", None)

    return next(
        (
            axon.coldkey
            for axon in getattr(metagraph, "axons", []) or []
            if getattr(axon, "hotkey", None) == hotkey
        ),
        None,
    )


def get_validator_identity(owner) -> dict[str, Any]:
    return to_jsonable(owner.validator_identity)


def _extract_request_query(response) -> str:
    if hasattr(response, "prompt") and getattr(response, "prompt"):
        return str(response.prompt)
    if hasattr(response, "query") and getattr(response, "query"):
        return str(response.query)
    if hasattr(response, "id") and getattr(response, "id"):
        return str(response.id)
    if hasattr(response, "urls") and getattr(response, "urls"):
        return ", ".join(to_jsonable(response.urls))
    return ""


def _slice_event_value(value: Any, index: int, response_count: int) -> Any:
    serialized = to_jsonable(value)
    if isinstance(serialized, list) and len(serialized) == response_count:
        return serialized[index]
    return serialized


def _sanitize_response_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_response_payload(item)
            for key, item in value.items()
            if key not in _RESPONSE_PAYLOAD_EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_response_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_response_payload(item) for item in value)
    return value


def build_reward_payload(
    search_type: str,
    response_count: int,
    index: int,
    uid: int,
    total_reward: float,
    all_rewards,
    all_original_rewards,
    validator_scores,
    event: dict[str, Any],
) -> dict[str, Any]:
    component_names = REWARD_COMPONENT_NAMES[search_type]
    components = {}
    original_components = {}

    for component_name, rewards, original_rewards in zip(
        component_names, all_rewards, all_original_rewards
    ):
        reward_values = to_jsonable(rewards)
        original_values = to_jsonable(original_rewards)

        if isinstance(reward_values, list) and len(reward_values) > index:
            components[component_name] = reward_values[index]
        else:
            components[component_name] = reward_values

        if isinstance(original_values, list) and len(original_values) > index:
            original_components[component_name] = original_values[index]
        else:
            original_components[component_name] = original_values

    validator_score_payload = {}
    for component_name, scores in zip(component_names, validator_scores):
        serialized_scores = to_jsonable(scores)

        if isinstance(serialized_scores, list) and len(serialized_scores) > index:
            validator_score_payload[component_name] = serialized_scores[index]
        elif isinstance(serialized_scores, dict):
            validator_score_payload[component_name] = serialized_scores.get(
                str(uid), serialized_scores.get(uid, serialized_scores)
            )
        else:
            validator_score_payload[component_name] = serialized_scores

    event_slice = {}
    penalties = {}
    for key, value in (event or {}).items():
        if key in {"step_length", "prompts", "uids", "rewards"}:
            continue

        sliced_value = _slice_event_value(value, index, response_count)
        event_slice[key] = sliced_value

        for suffix in ("_raw", "_adjusted", "_applied"):
            if key.endswith(suffix):
                penalties.setdefault(key[: -len(suffix)], {})[suffix[1:]] = sliced_value
                break

    return {
        "total_reward": total_reward,
        "components": components,
        "original_components": original_components,
        "validator_scores": validator_score_payload,
        "penalties": penalties,
        "event_slice": event_slice,
    }


def build_log_entry(
    owner,
    search_type: str,
    query_kind: str,
    response,
    miner_uid: Optional[int] = None,
    miner_hotkey: Optional[str] = None,
    miner_coldkey: Optional[str] = None,
    total_reward: Optional[float] = None,
    reward_payload: Optional[dict[str, Any]] = None,
    scoring_epoch_start: Optional[datetime] = None,
) -> dict[str, Any]:
    validator_identity = get_validator_identity(owner)
    response_payload = _sanitize_response_payload(to_jsonable(response))
    response_axon = getattr(response, "axon", None)

    miner_hotkey = miner_hotkey or getattr(response_axon, "hotkey", None)
    miner_coldkey = miner_coldkey or _find_miner_coldkey(owner, miner_hotkey)

    if miner_uid is None:
        miner_uid = _find_hotkey_uid(owner, miner_hotkey)

    return {
        "query_kind": query_kind,
        "search_type": search_type,
        "netuid": validator_identity.get("netuid")
        or getattr(owner.config, "netuid", 0),
        "scoring_epoch_start": to_jsonable(scoring_epoch_start),
        "miner_uid": miner_uid,
        "miner_hotkey": miner_hotkey or "",
        "miner_coldkey": miner_coldkey,
        "validator_uid": validator_identity.get("uid"),
        "validator_hotkey": validator_identity.get("hotkey") or "",
        "validator_coldkey": validator_identity.get("coldkey"),
        "request_query": _extract_request_query(response),
        "status_code": _safe_int(
            getattr(getattr(response, "dendrite", None), "status_code", None)
        ),
        "process_time": _safe_float(
            getattr(getattr(response, "dendrite", None), "process_time", None)
        ),
        "total_reward": _safe_float(total_reward),
        "response_payload": response_payload,
        "reward_payload": to_jsonable(reward_payload),
    }


async def submit_logs(owner, logs: list[dict[str, Any]]) -> None:
    if not logs:
        return

    utility_api = getattr(owner, "utility_api", None)
    if utility_api is None:
        bt.logging.warning("Utility API client is not configured; skipping logs save.")
        return

    try:
        bt.logging.debug(f"Saving miner response logs count={len(logs)}")
        await utility_api.save_logs(logs)
        bt.logging.debug(f"Saved miner response logs count={len(logs)}")
    except Exception as exc:
        bt.logging.error(f"Failed to save miner response logs count={len(logs)}: {exc}")


def submit_logs_best_effort(owner, logs: list[dict[str, Any]]) -> None:
    if not logs:
        return

    asyncio.create_task(submit_logs(owner, logs))
