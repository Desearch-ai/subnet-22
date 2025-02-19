# Smart Scrape Search API Documentation

The Smart Scrape Search API provides a unified interface to search across multiple platforms including Google, Twitter, YouTube, and more. It returns streaming responses that can include both links and summaries of search results.

## Base URL

```
https://api.smartscrape.ai/search
```

## Authentication

Authentication is required via an `Access-Key` header.

```python
headers = {"Access-Key": "access_key_here"}
```

## Available Search Tools

The API supports the following search platforms:

-   Twitter Search (past week tweets)
-   Web Search (web results)
-   ArXiv Search (academic papers)
-   Wikipedia Search (articles)
-   Youtube Search (videos)
-   Hacker News Search (posts)
-   Reddit Search (posts)

## Request Structure

### Endpoint

```
POST /search
```

### Request Body Parameters

| Parameter      | Type   | Required | Description                                         | Example                           |
| -------------- | ------ | -------- | --------------------------------------------------- | --------------------------------- |
| prompt         | string | Yes      | Search query text                                   | "What are the recent sport news?" |
| tools          | array  | Yes      | List of search tools to use                         | ["Web Search"]                    |
| model          | string | No       | Model to use for scraping (NOVA, ORBIT, HORIZON)    | NOVA                              |
| date_filter    | string | No       | Time range filter for results                       | "PAST_WEEK"                       |

### Date Filter Options

-   PAST_DAY
-   PAST_WEEK (default)
-   PAST_2_WEEKS
-   PAST_MONTH
-   PAST_YEAR

## Response Format

The API returns a streaming response where each chunk is prefixed with "data: " and contains a JSON object with the following structure:

```json
{
    "type": "completion",
    "content": "search result content here"
}
```

## Example Usage

```python
import asyncio
import aiohttp
import json

async def search_smart_scrape():
    url = "https://api.smartscrape.ai/search"

    body = {
        "prompt": "What are the recent sport news?",
        "tools": ["Web Search"],
        "model": "NOVA",
        "date_filter": "PAST_WEEK"
    }

    headers = {"Access-Key": "your_access_key_here"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers) as response:
            async for chunk in response.content.iter_any():
                decoded_data = chunk.decode("utf-8")

                if decoded_data.startswith("data: "):
                    json_data = decoded_data[6:]  # Remove 'data: ' prefix
                    try:
                        parsed_data = json.loads(json_data)

                        content_type = parsed_data.get("type")
                        content = parsed_data.get("content")

                        if content_type == "completion":
                            print("-" * 50)
                            print("Completion:\n")
                            print(content.strip())
                        else:
                            # Process the other type of chunks
                            print(parsed_data)
                    except json.JSONDecodeError:
                        print("Failed to decode JSON:", json_data)

# Run the async function
asyncio.run(search_smart_scrape())
```

## Limitations

-   Maximum execution time must be one of: 10, 30, or 120 seconds
-   Date filter is applied to all search tools where applicable
