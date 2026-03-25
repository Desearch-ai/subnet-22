import asyncio
import traceback
from typing import Dict

import bittensor as bt

from desearch.protocol import ResultType, ScraperStreamingSynapse, ScraperTextRole
from desearch.tools.final_summary import generate_summary, prepare_data_for_summary
from desearch.tools.get_tools import (
    find_toolkit_by_tool_name,
    get_all_tools,
)
from desearch.tools.response_streamer import ResponseStreamer


class ToolManager:
    def __init__(
        self,
        synapse: ScraperStreamingSynapse,
        date_filter,
        send,
        start_time: float,
    ):
        self.synapse = synapse
        self.prompt = synapse.prompt
        self.tools = synapse.tools
        self.system_message = synapse.system_message
        self.date_filter = date_filter
        self.start_time = start_time
        self.max_execution_time = synapse.max_execution_time

        self.response_streamer = ResponseStreamer(send=send)
        self.send = send

        self.all_tools = get_all_tools()
        self.tool_name_to_instance = {tool.name: tool for tool in self.all_tools}

    async def run(self):
        actions = await self.detect_tools_to_use()
        tasks = [asyncio.create_task(self.run_tool(action)) for action in actions]
        tool_results = {}

        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if not result:
                continue

            tool_result, _, tool_name = result
            if tool_result is not None:
                tool_results[tool_name] = tool_result

        if self.synapse.miner_link_scores:
            await self.response_streamer.send_event(
                "miner_link_scores", self.synapse.miner_link_scores
            )

        if self.synapse.result_type == ResultType.LINKS_WITH_FINAL_SUMMARY:
            await self.finalize_summary_and_stream(tool_results)

        await self.response_streamer.send_completion_event()

    async def detect_tools_to_use(self):
        # If user provided tools manually, use them
        if not self.tools:
            raise ValueError(
                "No manual tool names provided. Please specify tools to use."
            )

        return [
            {"action": tool_name, "args": self.prompt}
            for tool_name in self.tools
            if tool_name in self.tool_name_to_instance
        ]

    async def run_tool(self, action: Dict[str, str]):
        tool_name = action.get("action")
        tool_args = action.get("args")
        tool_instance = self.tool_name_to_instance.get(tool_name)

        if not tool_instance:
            return

        bt.logging.info(f"Running tool: {tool_name} with args: {tool_args}")

        tool_instance.tool_manager = self
        result = None

        try:
            if isinstance(tool_args, dict):
                result = await tool_instance._arun(**tool_args)
            elif isinstance(tool_args, str):
                result = await tool_instance._arun(tool_args)
        except Exception as e:
            bt.logging.error(f"Error running tool {tool_name}: {e}")

        if tool_instance.send_event and result is not None:
            bt.logging.info(f"Sending event with data from {tool_name} tool")

            await tool_instance.send_event(
                send=self.send,
                response_streamer=self.response_streamer,
                data=result,
            )

        return result, find_toolkit_by_tool_name(tool_name).name, tool_name

    async def finalize_summary_and_stream(self, tool_results):
        standardized_results = prepare_data_for_summary(tool_results)

        formatted_data = []

        for standardized_result in standardized_results:
            if standardized_result.get("id"):
                formatted_data.append(
                    {
                        "tweet": standardized_result.get("text"),
                        "link": standardized_result.get("url"),
                        "date": standardized_result.get("created_at"),
                    }
                )
            else:
                formatted_data.append(
                    {
                        "title": standardized_result.get("title"),
                        "snippet": standardized_result.get("snippet"),
                        "link": standardized_result.get("link"),
                    }
                )

        try:
            response = generate_summary(
                prompt=self.prompt,
                formatted_data=formatted_data,
                date_filter=self.date_filter
                if "Twitter Search" in self.tools
                else None,
            )

            await self.response_streamer.stream_response(
                response=response, role=ScraperTextRole.FINAL_SUMMARY
            )
        except Exception as err:
            bt.logging.error(
                f"Error generating summary for prompt {self.prompt}: {err}\n"
                f"{traceback.format_exc()}"
            )
