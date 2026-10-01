import asyncio

from app import limits

from tests.http_client import HttpClient
from tests.test_api_flow import Harness

JUNK = {
    "X-Hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "X-Timestamp": "1",
    "X-Nonce": "0" * 32,
    "X-Signature": "00" * 64,
}


def run(memory, scenario):
    async def main():
        async with Harness(memory) as h:
            await h.enqueue()
            return await scenario(h)

    return asyncio.run(main())


def client(h, ip: str) -> HttpClient:
    return HttpClient(h.url, headers={"X-Forwarded-For": ip, **JUNK})


def test_a_body_over_its_limit_is_refused_without_being_read(api_env, memory):
    async def scenario(h):
        async with client(h, "1.1.1.1") as sender:
            big = await sender.post(
                "/v1/tasks/claim", data=b"x" * (limits.BODY_BYTES + 1)
            )

            async def chunks():
                for _ in range(limits.BODY_BYTES // 1000 + 2):
                    yield b"x" * 1000

            streamed = await sender.post("/v1/tasks/claim", data=chunks())
            small = await sender.post("/v1/tasks/claim", data=b"{}")
        return (
            big,
            streamed.status,
            small.status,
            (await h.miner.post("/v1/tasks/claim"))["tasks"][0],
        )

    big, streamed, small, task = run(memory, scenario)

    assert big.status == 413 and str(limits.BODY_BYTES) in big.json()["detail"]
    assert streamed == 413, "with no declared length the body is counted as it arrives"
    assert small == 401, "a small body reaches the signature check"
    assert task is not None


def test_a_verdict_may_be_larger_than_a_claim():
    assert limits.body_limit("/v1/validation/t1/score") == limits.LARGE_BODY_BYTES
    assert limits.body_limit("/v1/admin/enqueue") == limits.LARGE_BODY_BYTES
    assert limits.body_limit("/v1/tasks/claim") == limits.BODY_BYTES
    assert limits.body_limit("/v1/tasks/t1/complete") == limits.BODY_BYTES


def test_an_address_that_keeps_failing_to_sign_in_is_refused(api_env, memory):
    api_env.setenv("TASK_API_FAILED_WRITES_PER_MINUTE", "3")

    async def scenario(h):
        async with (
            client(h, "1.1.1.1") as flooder,
            HttpClient(h.url, headers={"X-Forwarded-For": "1.1.1.1"}) as reader,
        ):
            answers = [
                (await flooder.post("/v1/tasks/claim", data=b"{}")).status
                for _ in range(5)
            ]
            refused = await flooder.post("/v1/tasks/claim", data=b"{}")
            read = await reader.get("/v1/health")
        return (
            answers,
            refused,
            read.status,
            (await h.miner.post("/v1/tasks/claim"))["tasks"][0],
        )

    answers, refused, read, task = run(memory, scenario)

    assert answers == [401, 401, 401, 429, 429]
    assert 0 < int(refused.headers["Retry-After"]) <= 60
    assert read == 200, "reads are counted apart from writes"
    assert task is not None, "a miner at another address is not affected"
