from app import logs

from tests.test_trust import VALIDATOR, judged, run, three_active


async def read(h, path: str, **params) -> dict:
    response = await h.public.get(path, params=params)
    assert response.status == 200, response.text
    return response.json()


def test_every_validators_vote_on_a_task_is_kept_and_marked(api_env, memory):
    async def scenario(h):
        task = await three_active(h)
        await judged(h, h.validator, "fail", task_id=task["task_id"])
        await judged(h, h.other_validator, task_id=task["task_id"])
        task_id = task["task_id"]
        return (
            (await h.view(task_id))["votes"],
            (await read(h, "/v1/votes", task_id=task_id))["votes"],
            (await read(h, "/v1/votes", agreed="false"))["votes"],
            await read(h, f"/v1/validators/{VALIDATOR}"),
            (await read(h, "/v1/validators"))["validators"],
        )

    on_task, listed, disputed, validator, validators = run(memory, scenario, 2)

    assert [vote["verdict"] for vote in on_task] == ["pass", "fail", "pass"]
    assert [vote["agreed"] for vote in on_task] == [True, False, True]
    assert sum(vote["decided"] for vote in on_task) == 1
    assert {vote["final_verdict"] for vote in on_task} == {"pass"}
    assert listed == on_task[::-1], "the list is newest first"
    assert [vote["validator"] for vote in disputed] == [VALIDATOR]
    assert (validator["votes"], validator["disagreed"]) == (3, 1)
    assert validator["agreement"] == round(2 / 3, 4) and validator["active"]
    assert validator["audits"] == 2 and not validator["excluded"]
    assert len(validators) == 3 and sum(v["decided"] for v in validators) == 3


def test_votes_on_a_task_without_a_majority_count_for_nobody(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator)
        task = await h.mine(h.rival)
        await judged(h, h.other_validator, "fail", task_id=task["task_id"])
        await judged(h, h.validator, task_id=task["task_id"])
        return (await read(h, "/v1/votes", task_id=task["task_id"]))["votes"]

    votes = run(memory, scenario)

    assert sorted(vote["verdict"] for vote in votes) == ["fail", "pass"]
    assert {vote["final_verdict"] for vote in votes} == {"void"}
    assert all(vote["agreed"] is None and not vote["decided"] for vote in votes)


def test_the_overview_and_the_miner_list_follow_the_work(api_env, memory):
    async def scenario(h):
        task = await h.mine()
        await judged(h, h.validator)
        held = (await h.rival.post("/v1/tasks/claim"))["task"]
        waiting = await h.mine()
        return (
            {h.miner.hotkey, h.rival.hotkey},
            task,
            held,
            waiting,
            await read(h, "/v1/overview"),
            (await read(h, "/v1/miners"))["miners"],
            await read(h, f"/v1/miners/{h.miner.hotkey}"),
            await read(h, "/v1/live"),
            (await read(h, "/v1/stats/series", bucket_minutes=5, buckets=3))["points"],
            (await read(h, "/v1/events", miner=h.miner.hotkey))["events"],
            (await read(h, "/v1/tasks", verdict="fail"))["tasks"],
        )

    (
        hotkeys,
        task,
        held,
        waiting,
        overview,
        miners,
        miner,
        live,
        points,
        events,
        failed,
    ) = run(memory, scenario, 2)

    assert (overview["claimed"], overview["validating"]) == (1, 1)
    assert (
        overview["window"] | {"tasks": 1, "pass": 1, "credited": 2, "votes": 1}
        == (overview["window"])
    )
    assert overview["validators"] == {"active": 1, "known": 1}
    assert overview["miners"] == 1 and overview["total"]["pass"] == 1

    first = miners[0]
    assert (first["share"], first["credited"], first["budget"]) == (1.0, 2, 2)
    assert (first["coverage"], first["eligible"]) == (1.0, True)
    assert (first["in_flight"], first["locked_until"]) == (1, None)
    assert {m["hotkey"] for m in miners} == hotkeys
    assert (miner["share"], miner["window"]["tasks"]) == (1.0, 1)

    assert [claim["task_id"] for claim in live["claims"]] == [held["task_id"]]
    assert [upload["task_id"] for upload in live["uploads"]] == [waiting["task_id"]]
    assert live["uploads"][0]["voters"] == [] and live["claims"][0]["urls"] == 2

    assert len(points) == 3 and points[-1] | {"tasks": 1, "pass": 1} == points[-1]
    assert [point["tasks"] for point in points[:-1]] == [0, 0]
    assert [event["outcome"] for event in events] == [
        "completed",
        "issued",
        "completed",
        "issued",
    ]
    assert events[-1]["task_id"] == task["task_id"] and failed == []


def test_log_reads_are_bounded(api_env, memory, monkeypatch):
    async def scenario(h):
        wide = await h.public.get("/v1/stats/series", params={"bucket_minutes": 7})
        both = await h.public.get(
            "/v1/stats/series", params={"miner": "a", "validator": "b"}
        )
        long = await h.public.get("/v1/votes", params={"miner": "m" * 65})
        monkeypatch.setattr(logs, "MAX_WAITING", 0)
        busy = await h.public.get("/v1/tasks")
        return wide.status, both.status, long.status, busy

    wide, both, long, busy = run(memory, scenario)

    assert (wide, both, long) == (422, 422, 422)
    assert busy.status == 503 and busy.headers["Retry-After"] == "1"


def test_a_finalized_task_says_when_it_was_claimed_and_uploaded(api_env, memory):
    async def scenario(h):
        await h.mine()
        await judged(h, h.validator)
        return (await read(h, "/v1/tasks"))["tasks"][0]

    task = run(memory, scenario)

    assert task["claimed_at"] <= task["completed_at"] <= task["scored_at"]
