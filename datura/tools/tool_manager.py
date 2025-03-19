from typing import List, Dict, Optional, Any
from openai import OpenAI
import asyncio
import os
import json
import bittensor as bt
from datura.tools.base import BaseTool
from datura.tools.get_tools import (
    TOOLKITS,
    get_all_tools,
    find_toolkit_by_tool_name,
    find_toolkit_by_name,
)
from datura.tools.twitter.twitter_toolkit import TwitterToolkit
from datura.protocol import ScraperTextRole
from openai import AsyncOpenAI
from datura.tools.response_streamer import ResponseStreamer
from datura.protocol import TwitterPromptAnalysisResult

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


TEMPLATE = """Answer the following question as best you can.
User Question: {input}

You have access to the following tools:
{tools}

You can use multiple tools to answer the question. Order of tools does not matter.

Here is example of JSON array format to return. Keep in mind that this is example:
[
  {{
    "action": "Twitter Search",
    "args": {{
      "query": "AI trends"
    }}
  }},
  {{
    "action": "Web Search",
    "args": {{
      "query": "What are AI trends?"
    }}
  }}
]
"""

# prompt_template = PromptTemplate.from_template(TEMPLATE)

client = AsyncOpenAI(timeout=60.0)


class ToolManager:
    openai_summary_model: str = "gpt-3.5-turbo-0125"
    all_tools: List[BaseTool]
    manual_tool_names: List[str]
    tool_name_to_instance: Dict[str, BaseTool]

    # is_intro_text: bool
    miner: any
    language: str
    region: str
    date_filter: str

    twitter_prompt_analysis: Optional[TwitterPromptAnalysisResult]
    twitter_data: Optional[Dict[str, Any]]

    def __init__(
        self,
        prompt,
        manual_tool_names,
        send,
        # is_intro_text,
        miner,
        language,
        region,
        date_filter,
        google_date_filter,
        system_message,
    ):
        self.prompt = prompt
        self.manual_tool_names = manual_tool_names
        self.miner = miner
        # self.is_intro_text = is_intro_text
        self.language = language
        self.region = region
        self.date_filter = date_filter
        self.google_date_filter = google_date_filter
        self.system_message = system_message

        self.response_streamer = ResponseStreamer(send=send)
        self.send = send
        self.openai_summary_model = self.miner.config.miner.openai_summary_model

        self.all_tools = get_all_tools()
        self.tool_name_to_instance = {tool.name: tool for tool in self.all_tools}
        self.toolkit_name_to_instance = {toolkit.name: toolkit for toolkit in TOOLKITS}
        self.twitter_prompt_analysis = None
        self.twitter_data = None

    async def run(self):
        actions = await self.detect_tools_to_use()

        toolkit_actions = {}

        independent_tools = []

        for action in actions:
            tool_name = action["action"]

            toolkit = find_toolkit_by_tool_name(tool_name)
            if toolkit:
                toolkit_name = toolkit.name
                if toolkit_name not in toolkit_actions:
                    toolkit_actions[toolkit_name] = []
                toolkit_actions[toolkit_name].append(action)
            else:
                bt.logging.info(
                    f"Tool {tool_name} does not belong to any toolkit and is not an independent tool."
                )

        toolkit_tasks = []

        for toolkit_name, actions in toolkit_actions.items():
            toolkit_task = asyncio.create_task(self.run_toolkit(toolkit_name, actions))
            toolkit_tasks.append(toolkit_task)

        tool_tasks = []

        for action in independent_tools:
            tool_task = asyncio.create_task(self.run_tool(action))
            tool_tasks.append(tool_task)

        streaming_tasks = []

        for completed_task in asyncio.as_completed(toolkit_tasks):
            toolkit_name, results = await completed_task

            if results:
                response, role = await find_toolkit_by_name(toolkit_name).summarize(
                    prompt=self.prompt,
                    model=self.openai_summary_model,
                    data=results,
                    system_message=self.system_message,
                )

                streaming_task = asyncio.create_task(
                    self.response_streamer.stream_response(response=response, role=role)
                )

                streaming_tasks.append(streaming_task)

        await asyncio.gather(*streaming_tasks)

        await self.finalize_summary_and_stream(
            self.response_streamer.get_full_text(),
        )

        await asyncio.gather(*tool_tasks)

        await self.response_streamer.send_completion_event()

        if self.response_streamer.more_body:
            await self.send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

    async def detect_tools_to_use(self):
        # If user provided tools manually, use them
        if not self.manual_tool_names:
            raise ValueError(
                "No manual tool names provided. Please specify tools to use."
            )

        return [
            {"action": tool_name, "args": self.prompt}
            for tool_name in self.manual_tool_names
        ]

    async def run_toolkit(self, toolkit_name, actions):
        tasks = [asyncio.create_task(self.run_tool(action)) for action in actions]
        toolkit_instance = self.toolkit_name_to_instance[toolkit_name]

        if not toolkit_instance:
            return

        toolkit_instance.tool_manager = self
        toolkit_results = {}

        for completed_task in asyncio.as_completed(tasks):
            result, _, tool_name = await completed_task

            if result is not None:
                toolkit_results[tool_name] = result

        return toolkit_name, toolkit_results

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

    async def intro_text(self, model, tool_names):
        bt.logging.trace("miner.intro_text => ", self.miner.config.miner.intro_text)
        # bt.logging.trace("Synapse.is_intro_text => ", self.is_intro_text)
        if not self.miner.config.miner.intro_text:
            return

        # if not self.is_intro_text:
        #     return

        bt.logging.trace("Run intro text")

        tool_names = ", ".join(tool_names)

        content = f"""
        Generate introduction for that prompt: "{self.prompt}".
        You are going to use {tool_names} to fetch information.

        Something like it: "To effectively address your query, my approach involves a comprehensive analysis and integration of relevant Twitter and web search data. Here's how it works:

        Question or Topic Analysis: I start by thoroughly examining your question or topic to understand the core of your inquiry or the specific area you're interested in.

        Twitter Data Search: Next, I delve into Twitter, seeking out information, discussions, and insights that directly relate to your prompt.
        Web search: Next, I search web, seeking out information, discussions, and insights that directly relate to your prompt.

        Synthesis and Response: After gathering and analyzing this data, I compile my findings and craft a detailed response, which will be presented below"

        Output: Just return only introduction text without your comment
        """
        messages = [{"role": "user", "content": content}]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            stream=True,
        )

        response_streamer = ResponseStreamer(send=self.send)
        await response_streamer.stream_response(
            response=response, role=ScraperTextRole.INTRO, wait_time=0.1
        )

        return response_streamer.get_full_text()

    async def finalize_summary_and_stream(self, information):
        content = f"""
            In <UserPrompt> provided User's prompt (Question).
            In <Information>, provided highlighted key information and relevant links from Twitter and Web Search.

            <UserPrompt>
            {self.prompt}
            </UserPrompt>

                Output Guidelines (Tasks):
                1. Final Summary: Conduct a thorough analysis of <TwitterData> in relation to <UserPrompt> and generate a comprehensive summary.
                Synthesize insights from both the <UserPrompt> and the <TwitterData> to formulate a well-rounded response. But don't provide any twitter link, which is not related to <UserPrompt>.
                2. Highlight Key Information: Identify and emphasize any crucial information that will be beneficial to the user.
            <Information>
            {information}
            </Information>
        """

        system_message = """As a summary analyst, your task is to provide users with a clear and concise summary derived from the given information and the user's query.

        Output Guidelines (Tasks):
        1. Summary: Conduct a thorough analysis of <Information> in relation to <UserPrompt> and generate a comprehensive summary.

        Operational Rules:
        1. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
        2. Seamless Integration: Avoid explicitly stating "Based on the provided <Information>" in responses. Assume user awareness of the data integration process.
        3. Not return text like <UserPrompt> to your response, make response easy to understand to any user.
        4. Start text with bold text "Summary:".
        """

        if self.system_message:
            system_message += f"5. Follow the next rules for summarizing the content: {self.system_message}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]

        response = await client.chat.completions.create(
            model=self.openai_summary_model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )

        await self.response_streamer.stream_response(
            response=response, role=ScraperTextRole.FINAL_SUMMARY
        )

        bt.logging.info(
            "================================== Completion Response ==================================="
        )
        bt.logging.info(
            f"{self.response_streamer.get_full_text()}"
        )  # Print the full text at the end
        bt.logging.info(
            "================================== Completion Response ==================================="
        )
