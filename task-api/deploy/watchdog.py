"""Posts to a Discord webhook when the crawl pipeline stalls, once when it starts and again when it clears."""

import json
import os
import time
import urllib.request

API = os.environ.get("WATCHDOG_API", "https://api-22.desearch.ai")
WEBHOOK = os.environ.get("WATCHDOG_WEBHOOK", "")
STATE = os.environ.get("WATCHDOG_STATE", "/var/lib/desearch-watchdog/state.json")
AGENT = "desearch-watchdog/1"

NO_ENQUEUE_S = 300
NO_ROOM_S = 900
NO_PUBLISH_S = 300
OLDEST_PUBLISH_S = 1800
OLDEST_VALIDATION_S = 1800
VALIDATORS = 2
FAILS_PER_15_MIN = 10
DISK_USED = 0.8
API_DOWN_RUNS = 3
REMIND_S = 3600


def get(path):
    request = urllib.request.Request(f"{API}{path}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def post(text):
    body = json.dumps({"content": text[:1900]}).encode()
    request = urllib.request.Request(
        WEBHOOK,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": AGENT},
    )
    urllib.request.urlopen(request, timeout=20).close()


def problems(state, health, room, newest, now):
    """What is wrong now by name, and what got worse since the last run; updates the counters kept between runs."""
    found, worse = {}, set()
    opened = newest.get("opened_at") or 0
    if room["room_tasks"] > 0 and not room["refusing"] and now - opened > NO_ENQUEUE_S:
        found["no_enqueue"] = (
            f"nothing enqueued for {(now - opened) / 60:.0f} min while the API has room for {room['room_tasks']} tasks"
        )

    if room["refusing"] or room["room_tasks"] <= 0:
        state.setdefault("no_room_since", now)
    else:
        state.pop("no_room_since", None)
    if now - state.get("no_room_since", now) > NO_ROOM_S:
        found["no_room"] = (
            f"the API has had no room for {(now - state['no_room_since']) / 60:.0f} min "
            f"(published {room['published_per_min']}/min, oldest publish {health['oldest_publish_s'] / 60:.0f} min, "
            f"oldest validation {health['oldest_validation_s'] / 60:.0f} min)"
        )

    if room["published_per_min"] <= 0 and health["publishing"] > 0:
        state.setdefault("no_publish_since", now)
    else:
        state.pop("no_publish_since", None)
    if now - state.get("no_publish_since", now) > NO_PUBLISH_S:
        found["publisher_stalled"] = (
            f"the publisher finished nothing for {(now - state['no_publish_since']) / 60:.0f} min with {health['publishing']} jobs waiting"
        )
    if health["oldest_publish_s"] > OLDEST_PUBLISH_S:
        found["publish_slow"] = (
            f"the oldest publish job has waited {health['oldest_publish_s'] / 60:.0f} min"
        )

    set_aside = health["publish_set_aside"]
    if set_aside > 0:
        found["set_aside"] = (
            f"{set_aside} publish jobs set aside after failing repeatedly; they need requeueing"
        )
        if set_aside > state.get("set_aside", 0):
            worse.add("set_aside")
    state["set_aside"] = set_aside

    active = health["active_validators"]
    if len(active) < VALIDATORS:
        found["validators"] = (
            f"only {len(active)} validators active: {', '.join(v[:8] for v in active) or 'none'}"
        )
    if health["oldest_validation_s"] > OLDEST_VALIDATION_S:
        found["validation_slow"] = (
            f"the oldest upload has waited {health['oldest_validation_s'] / 60:.0f} min for its verdict"
        )

    fails = [entry for entry in state.get("fails", []) if now - entry[0] <= 900]
    fails.append([now, health["verdicts"].get("fail", 0)])
    state["fails"] = fails
    failed = fails[-1][1] - fails[0][1]
    if failed >= FAILS_PER_15_MIN:
        found["failed_checks"] = (
            f"{failed} uploads failed their checks in the last 15 min"
        )

    used = health.get("data_disk_used")
    if used is not None and used >= DISK_USED:
        found["disk"] = f"the API's data volume is {used:.0%} full"
    return found, worse


def notices(state, found, worse, now):
    """Messages to send: new problems, ones that got worse, reminders for old ones, and what cleared."""
    active = state.setdefault("active", {})
    out = []
    for name, text in found.items():
        seen = active.get(name)
        if seen is None or name in worse:
            out.append(f"🔴 {text}")
            active[name] = {
                "since": seen["since"] if seen else now,
                "said": now,
                "text": text,
            }
        elif now - seen["said"] >= REMIND_S:
            out.append(f"🔴 still, for {(now - seen['since']) / 60:.0f} min: {text}")
            seen.update(said=now, text=text)
        else:
            seen["text"] = text
    for name in [name for name in active if name not in found]:
        out.append(
            f"✅ cleared after {(now - active[name]['since']) / 60:.0f} min: {active.pop(name)['text']}"
        )
    return out


def load():
    try:
        with open(STATE) as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def save(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(f"{STATE}.tmp", "w") as file:
        json.dump(state, file)
    os.replace(f"{STATE}.tmp", STATE)


def main():
    state, now = load(), time.time()
    try:
        health, room = get("/v1/health"), get("/v1/room")
        newest = (get("/v1/rounds?limit=1").get("rounds") or [{}])[0]
        state["api_down"] = 0
        found, worse = problems(state, health, room, newest, now)
    except Exception as error:
        state["api_down"] = state.get("api_down", 0) + 1
        found = {name: seen["text"] for name, seen in state.get("active", {}).items()}
        worse = set()
        if state["api_down"] >= API_DOWN_RUNS:
            found["api_down"] = (
                f"the task API has not answered for {state['api_down']} min: {error}"
            )
    for text in notices(state, found, worse, now):
        if WEBHOOK:
            post(text)
        else:
            print(text)
    save(state)


if __name__ == "__main__":
    main()
