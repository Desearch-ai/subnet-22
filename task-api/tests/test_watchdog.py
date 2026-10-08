import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "watchdog", Path(__file__).parent.parent / "deploy" / "watchdog.py"
)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)

NOW = 1_800_000_000.0


def health(**changes):
    healthy = {
        "active_validators": ["5Aaa", "5Bbb"],
        "oldest_validation_s": 120.0,
        "publishing": 3,
        "oldest_publish_s": 60.0,
        "publish_set_aside": 0,
        "verdicts": {"pass": 100, "fail": 2},
        "data_disk_used": 0.6,
    }
    return {**healthy, **changes}


def room(**changes):
    return {"room_tasks": 500, "refusing": False, "published_per_min": 350.0, **changes}


def run(state, at, newest_at=None, **kwargs):
    found, worse = watchdog.problems(
        state,
        kwargs.get("health", health()),
        kwargs.get("room", room()),
        {"opened_at": at - 20 if newest_at is None else newest_at},
        at,
    )
    return watchdog.notices(state, found, worse, at)


def test_a_healthy_pipeline_says_nothing():
    state = {}
    assert run(state, NOW) == [] and run(state, NOW + 60) == []


def test_nothing_enqueued_while_there_is_room_is_said_once_then_cleared():
    state = {}
    said = run(state, NOW, newest_at=NOW - 400)
    assert len(said) == 1 and "nothing enqueued for 7 min" in said[0]
    assert run(state, NOW + 60, newest_at=NOW - 400) == []
    assert run(state, NOW + 120)[0].startswith("✅ cleared after 2 min")


def test_no_room_is_only_said_after_it_lasts():
    state = {}
    full = room(room_tasks=0)
    assert run(state, NOW, room=full) == []
    assert run(state, NOW + 600, room=full) == []
    said = run(state, NOW + 960, room=full)
    assert len(said) == 1 and "no room for 16 min" in said[0]


def test_set_aside_jobs_stay_flagged_and_more_of_them_are_said_at_once():
    state = {}
    assert (
        "3 publish jobs set aside"
        in run(state, NOW, health=health(publish_set_aside=3))[0]
    )
    assert run(state, NOW + 60, health=health(publish_set_aside=3)) == []
    assert (
        "5 publish jobs set aside"
        in run(state, NOW + 120, health=health(publish_set_aside=5))[0]
    )
    assert run(state, NOW + 180)[0].startswith("✅")


def test_failed_checks_count_over_fifteen_minutes():
    state = {}
    run(state, NOW, health=health(verdicts={"fail": 2}))
    assert run(state, NOW + 300, health=health(verdicts={"fail": 8})) == []
    said = run(state, NOW + 600, health=health(verdicts={"fail": 12}))
    assert len(said) == 1 and "10 uploads failed" in said[0]
    assert run(state, NOW + 1600, health=health(verdicts={"fail": 12}))[0].startswith(
        "✅"
    )


def test_a_problem_is_repeated_hourly_while_it_lasts():
    state = {}
    low = health(active_validators=["5Aaa"])
    assert len(run(state, NOW, health=low)) == 1
    assert run(state, NOW + 1800, health=low) == []
    assert run(state, NOW + 3600, health=low)[0].startswith("🔴 still, for 60 min")
