import json
import asyncio
from starlette.types import Send
from desearch.protocol import ScraperTextRole
import bittensor as bt


class ResponseStreamer:
    def __init__(self, send: Send) -> None:
        self.texts = {}
        self.role_order = []
        self.more_body = True
        self.send = send

    async def send_text_event(self, text: str, role: ScraperTextRole):
        text_data_json = json.dumps(
            {"type": "text", "role": role.value, "content": text}
        )
        await self.send(
            {
                "type": "http.response.body",
                "body": text_data_json.encode("utf-8"),
                "more_body": True,
            }
        )

    async def stream_response(self, response, role: ScraperTextRole, wait_time=None):
        if role not in self.role_order:
            self.role_order.append(role)

        if role not in self.texts:
            await self.send_text_event(text="\n\n", role=role)
            self.texts[role] = ["\n\n"]

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            self.texts[role].append(token)

            await self.send_text_event(text=token, role=role)

            if wait_time is not None:
                await asyncio.sleep(wait_time)

            bt.logging.trace(f"Streamed tokens: {token}")

    async def send_completion_event(self):
        completion_response_body = {
            "type": "completion",
            "content": self.get_full_text(),
        }

        await self.send(
            {
                "type": "http.response.body",
                "body": json.dumps(completion_response_body).encode("utf-8"),
            }
        )

    def get_full_text(self):
        full_text = []

        for role in self.role_order:
            if role in self.texts:
                full_text.append("".join(self.texts[role]))

        return "".join(full_text)
