import asyncio
import time

from app.auth import Keypair

from tests.http_client import HttpClient
from tests.test_api_flow import Harness


def run(memory, scenario):
    async def main():
        async with Harness(memory) as h:
            return await scenario(h)

    return asyncio.run(main())


def test_reads_are_counted_per_cloudflare_caller_not_per_forwarded_header(
    api_env, memory
):
    api_env.setenv("TASK_API_READS_PER_MINUTE", "2")

    def behind_cloudflare(h, caller: str) -> HttpClient:
        headers = {"CF-Connecting-IP": caller, "X-Forwarded-For": "9.9.9.9"}
        return HttpClient(h.url, headers=headers)

    async def scenario(h):
        async with (
            behind_cloudflare(h, "1.1.1.1") as one,
            behind_cloudflare(h, "2.2.2.2") as other,
        ):
            first = [(await one.get("/v1/shares")).status for _ in range(3)]
            return first, (await other.get("/v1/shares")).status

    first, other = run(memory, scenario)

    assert first == [200, 200, 429]
    assert other == 200, "a shared forwarded header does not share the limit"


def test_an_unregistered_hotkey_is_turned_away_before_its_signature_is_checked(
    api_env, memory
):
    stranger = Keypair.create_from_uri("//not-on-the-subnet").ss58_address
    headers = {
        "X-Hotkey": stranger,
        "X-Timestamp": str(int(time.time())),
        "X-Nonce": "0" * 32,
        "X-Signature": "00" * 64,
    }

    class Registry:
        async def lookup(self, hotkey):
            return None

    async def scenario(h):
        h.core.registry = Registry()
        async with HttpClient(h.url, headers=headers) as client:
            return await client.post("/v1/tasks/claim")

    response = run(memory, scenario)

    assert response.status == 403
    assert "not registered" in response.text


def test_without_cloudflare_in_front_the_connection_address_is_used(api_env, memory):
    api_env.setenv("TASK_API_READS_PER_MINUTE", "2")

    def direct(h, address: str) -> HttpClient:
        return HttpClient(h.url, headers={"X-Forwarded-For": address})

    async def scenario(h):
        async with direct(h, "3.3.3.3") as one, direct(h, "4.4.4.4") as other:
            first = [(await one.get("/v1/shares")).status for _ in range(3)]
            return first, (await other.get("/v1/shares")).status

    first, other = run(memory, scenario)

    assert first == [200, 200, 429]
    assert other == 200
