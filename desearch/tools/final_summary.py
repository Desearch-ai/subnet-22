import json
from datetime import datetime, timezone

import bittensor as bt

from desearch import client
from desearch.dataset.date_filters import DateFilter

FINAL_SUMMARY_MODEL = "gpt-4.1-nano"

SYSTEM_MESSAGE = """
Answer the question in markdown with accurate and relevant information backed by sources.

<Context>
Current date: {current_date}.
{date_filter_context}
<Context>

1. Support each section, paragraph or sentence by source using markdown links where the label is the source index and the destination is the exact source link from <Data>.
3. Example: if source 1 has link https://example.com, cite it as [1](https://example.com).
4. You can put multiple sources in one sentence, like this: [1](https://example.com), [2](https://example.org).
5. Split sections by bold headers using **section**. Do not use headers starting with #.
6. Write 2-3 sections besides **Conclusion**.
7. Last section must be **Conclusion** with a concise summary of all sections.
8. Max 400 words.
9. Do NOT include a separate **Sources** section.
"""

USER_MESSAGE = """You are provided with <Question> asked by user.
In <Data> is provided different sources with their index numbers.

<Question>
{prompt}
</Question>

<Data>
{formatted_data}
</Data>
"""


def prepare_data_for_summary(data):
    standardized_results = []

    for tool_name, tool_result in data.items():
        if tool_name == "Twitter Search":
            standardized_results.extend(tool_result)
            continue

        for result in tool_result:
            standardized_results.append(
                {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                }
            )

    return standardized_results


async def generate_summary(
    prompt: str,
    formatted_data,
    date_filter: DateFilter | None = None,
    timeout: int = 10,
):
    date_filter_context = ""

    if date_filter:
        date_filter_context = (
            "Date filter applied to tweets: "
            f"{date_filter.date_filter_type.value}, "
            f"start date: {date_filter.start_date.strftime('%Y-%m-%d')}, "
            f"end date: {date_filter.end_date.strftime('%Y-%m-%d')}."
        )

    system_prompt = SYSTEM_MESSAGE.format(
        current_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        date_filter_context=date_filter_context,
    )

    formatted_data_with_index = []

    for i, item in enumerate(formatted_data[:20], 1):
        item_copy = item.copy()
        item_copy["index"] = i
        formatted_data_with_index.append(item_copy)

    user_prompt = USER_MESSAGE.format(
        prompt=prompt,
        formatted_data=json.dumps(formatted_data_with_index, indent=2),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = await client.chat.completions.create(
        model=FINAL_SUMMARY_MODEL,
        messages=messages,
        temperature=0.5,
        stream=True,
        timeout=timeout,
    )

    async for chunk in response:
        yield chunk
