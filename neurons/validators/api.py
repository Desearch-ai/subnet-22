import os

os.environ["USE_TORCH"] = "1"

import asyncio
from typing import Optional, Annotated, List, Optional
from pydantic import BaseModel, Field, conint
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Header, Query, Path
from neurons.validators.env import PORT, EXPECTED_ACCESS_KEY
from datura import __version__
from datura.dataset.date_filters import DateFilterType
from datura.protocol import (
    ChatHistoryItem,
    Model,
    TwitterScraperTweet,
    WebSearchResultList,
    ResultType,
    PeopleSearchResultList,
)
import uvicorn
import aiohttp
import bittensor as bt
import traceback
from neurons.validators.validator import Neuron
from neurons.validators.validator_service_client import ValidatorServiceClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import json

neu: Neuron = None


async def get_validator_config():
    async with ValidatorServiceClient() as client:
        while True:
            print("Waiting for validator service to start...")

            try:
                config = await client.get_config()
                print("Validator config fetched successfully.")
                return config
            except aiohttp.ClientError:
                print("Waiting for validator service to start...")
            finally:
                await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app):
    # Start the neuron when the app starts
    global neu

    config = await get_validator_config()
    neu = Neuron(lite=True, config=config)
    await neu.run()

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


available_tools = [
    "Twitter Search",
    "Web Search",
    "ArXiv Search",
    "Wikipedia Search",
    "Youtube Search",
    "Hacker News Search",
    "Reddit Search",
]

twitter_tool = ["Twitter Search"]


def format_enum_values(enum):
    values = [value.value for value in enum]
    values = ", ".join(values)

    return f"Options: {values}"


class SearchRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Search query prompt",
        example="What are the recent sport events?",
    )
    tools: List[str] = Field(
        ..., description="List of tools to search with", example=available_tools
    )
    date_filter: Optional[DateFilterType] = Field(
        default=DateFilterType.PAST_WEEK,
        description=f"Date filter for the search results.{format_enum_values(DateFilterType)}",
        example=DateFilterType.PAST_WEEK.value,
    )

    model: Optional[Model] = Field(
        default=Model.NOVA,
        description=f"Model to use for scraping. {format_enum_values(Model)}",
        example=Model.NOVA.value,
    )

    count: Optional[int] = Field(
        10,
        title="Count",
        description="The number of results to return per source. Min 10. Max 200.",
        ge=10,
        le=200,
    )

    result_type: Optional[ResultType] = Field(
        default=ResultType.LINKS_WITH_FINAL_SUMMARY,
        description=f"Type of result. {format_enum_values(ResultType)}",
        example=ResultType.LINKS_WITH_FINAL_SUMMARY.value,
    )

    system_message: Optional[str] = Field(
        default=None,
        description="Rules influencing how summaries are generated",
        example="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
    )

    scoring_system_message: Optional[str] = Field(
        default=None,
        description="System message for scoring the response",
        example='Business Relevance Scoring Guide:\n\n                    Task: As an evaluator, determine how well a tweet represents a business opportunity for the agent based on contextual relevance to the specific agent use case provided.\n\n                    IMPORTANT: Only score highly for needs that directly relate to the agent\'s specific use case. Similar but different problems should receive lower scores unless they specifically mention the agent\'s domain.\n\n                    Agent use case: Find people who needs SERP API service and is looking for cheaper options for scraping\n\n                    Scoring Criteria:\n\n                    Score 2 - No Business Opportunity:\n                    - Criteria: Tweet is completely unrelated to the agent\'s use case or shows no indication of need for the agent\'s specific offering\n                    - Context: Author is not expressing any pain point, question, or situation relevant to the agent\'s domain\n                    - Examples:\n                    - Agent Use Case: "Find people needing SERP API services"\n                    - Tweet: "Just had amazing pizza for lunch!" → Score 2 (completely unrelated)\n                    - Tweet: "Our streaming API is having problems" → Score 2 (different API domain)\n\n                    Score 5 - Potential Interest:\n                    - Criteria: Tweet shows indirect relevance to the agent\'s domain but lacks clear intent, urgency, or specific need\n                    - Context: Author mentions related topics but doesn\'t express explicit need for the agent\'s specific solution\n                    - Examples:\n                    - Agent Use Case: "Find people needing SERP API services"\n                    - Tweet: "Working on a new web scraping project" → Score 5 (related activity, no explicit SERP need)\n                    - Tweet: "APIs are getting expensive these days" → Score 5 (general API concern, not SERP-specific)\n\n                    Score 9 - Strong Business Opportunity:\n                    - Criteria: Tweet indicates a clear need, problem, or interest that directly aligns with the agent\'s specific use case\n                    - Context: Author is seeking solutions, expressing frustration, asking for recommendations, or describing challenges specifically in the agent\'s domain\n                    - Examples:\n                    - Agent Use Case: "Find people needing SERP API services"\n                    - Tweet: "Anyone know a reliable API for Google search results? Current one keeps failing" → Score 9 (direct SERP API need)\n                    - Tweet: "SERP API costs are killing our budget, need alternatives" → Score 9 (specific SERP API problem)\n                    \n                    Output Format:\n                    Score: [2, 5, or 9], Explanation: [Brief explanation focusing on how specifically this relates to the agent\'s use case and the level of expressed need]',
    )

    chat_history: Optional[List[ChatHistoryItem]] = Field(
        default_factory=list,
        title="Chat History",
        description="A list of chat history items.",
    )

    uid: Optional[int] = Query(default=None)


class DeepResearchRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Search query prompt",
        example="What are the recent sport events?",
    )

    tools: List[str] = Field(
        ..., description="List of tools to search with", example=available_tools
    )

    date_filter: Optional[DateFilterType] = Field(
        default=DateFilterType.PAST_WEEK,
        description=f"Date filter for the search results.{format_enum_values(DateFilterType)}",
        example=DateFilterType.PAST_WEEK.value,
    )

    system_message: Optional[str] = Field(
        default=None,
        description="Rules influencing how summaries are generated",
        example="Summarize the content by categorizing key points into 'Pros' and 'Cons' sections.",
    )

    uid: Optional[int] = Field(
        default=None,
    )


class LinksSearchRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Search query prompt",
        example="What are the recent sport events?",
    )
    tools: List[str] = Field(
        ..., description="List of tools to search with", example=available_tools
    )

    model: Optional[Model] = Field(
        default=Model.NOVA,
        description=f"Model to use for scraping. {format_enum_values(Model)}",
        example=Model.NOVA.value,
    )

    count: Optional[int] = Field(
        10,
        title="Count",
        description="The number of results to return per source. Min 10. Max 200.",
        ge=10,
        le=200,
    )

    uid: Optional[int] = Field(default=None)


fields = "\n".join(
    f"- {key}: {item.get('description')}"
    for key, item in SearchRequest.schema().get("properties", {}).items()
)

SEARCH_DESCRIPTION = f"""Performs a search across multiple platforms. Available tools are:
- Twitter Search: Uses Twitter API to search for tweets in past week date range.
- Web Search: Searches the web.
- ArXiv Search: Searches academic papers on ArXiv.
- Wikipedia Search: Searches articles on Wikipedia.
- Youtube Search: Searches videos on Youtube.
- Hacker News Search: Searches posts on Hacker News, under the hood it uses web search.
- Reddit Search: Searches posts on Reddit, under the hood it uses web search.

Request Body Fields:
{fields}
"""


async def response_stream_event(data: SearchRequest):
    try:
        query = {
            "content": data.prompt,
            "tools": data.tools,
            "count": data.count,
            "date_filter": data.date_filter.value,
            "system_message": data.system_message,
            "scoring_system_message": data.scoring_system_message,
            "chat_history": data.chat_history,
        }

        merged_chunks = ""

        async for response in neu.advanced_scraper_validator.organic(
            query,
            data.model,
            result_type=data.result_type,
            uid=data.uid,
        ):
            # Decode the chunk if necessary and merge
            chunk = str(response)  # Assuming response is already a string
            merged_chunks += chunk
            lines = chunk.split("\n")
            sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
            yield f"{sse_data}\n\n"
    except Exception as e:
        bt.logging.error(f"error in response_stream {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def aggregate_search_results(responses: List[bt.Synapse], tools: List[str]):
    """
    Aggregates search results from multiple Synapse responses into a dictionary
    with tool names as keys and their corresponding results.
    """

    # Define the mapping of tool names to response fields in Synapse
    field_mapping = {
        "Twitter Search": "miner_tweets",
        "Web Search": "search_results",
        "ArXiv Search": "arxiv_search_results",
        "Wikipedia Search": "wikipedia_search_results",
        "Youtube Search": "youtube_search_results",
        "Hacker News Search": "hacker_news_search_results",
        "Reddit Search": "reddit_search_results",
    }

    aggregated = {}

    # Loop through each Synapse response
    for synapse_index, synapse in enumerate(responses):
        for tool in tools:
            # Get the corresponding field name for the tool
            field_name = field_mapping.get(tool)

            try:
                result = getattr(synapse, field_name)

                if result:

                    # If result is a list, extend the existing aggregated list
                    if isinstance(result, list):
                        if field_name not in aggregated:
                            aggregated[field_name] = []
                        aggregated[field_name].extend(result)

                    # If result is a dict, just assign it
                    elif isinstance(result, dict):
                        aggregated[field_name] = result

                    else:
                        # Handle unexpected result types if necessary
                        bt.logging.warning(
                            f"Unexpected result type for tool '{tool}': {type(result)}"
                        )
                        aggregated[field_name] = result
                else:
                    # If result is None or empty, just log it
                    bt.logging.debug(
                        f"No data found for '{tool}' on Synapse {synapse_index}."
                    )
            except AttributeError:
                pass

    # Replace None values with empty dictionaries for tools with no results
    for tool in tools:
        field_name = field_mapping.get(tool)

        if field_name not in aggregated:
            aggregated[field_name] = []

    return aggregated


async def handle_search_links(
    body: LinksSearchRequest,
    access_key: str | None,
    expected_access_key: str,
    tools: List[str],
):
    if access_key != expected_access_key:
        raise HTTPException(status_code=401, detail="Invalid access key")

    query = {"content": body.prompt, "tools": tools, "count": body.count}
    synapses = []

    bt.logging.info(f"Handle search links, query: {query}")

    try:
        async for item in neu.advanced_scraper_validator.organic(
            query,
            body.model,
            is_collect_final_synapses=True,
            result_type=ResultType.ONLY_LINKS,
            uid=body.uid,
        ):
            synapses.append(item)

        # Aggregate the results
        aggregated_results = await aggregate_search_results(synapses, tools)

        return aggregated_results

    except Exception as e:
        bt.logging.error(f"Error in handle_search_links: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(
    "/search",
    summary="Search across multiple platforms",
    description=SEARCH_DESCRIPTION,
    response_description="A stream of search results from the specified tools.",
)
async def search(
    body: SearchRequest, access_key: Annotated[str | None, Header()] = None
):
    """
    Search endpoint that accepts a JSON body with search parameters.
    """

    bt.logging.info(f"/search request: {body}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return StreamingResponse(response_stream_event(body))


async def stream_deep_research(data: DeepResearchRequest):
    try:
        query = {
            "content": data.prompt,
            "tools": data.tools,
            "date_filter": data.date_filter.value,
            "system_message": data.system_message,
        }

        merged_chunks = ""

        async for response in neu.deep_research_validator.organic(query, uid=data.uid):
            # Decode the chunk if necessary and merge
            chunk = str(response)  # Assuming response is already a string
            merged_chunks += chunk
            lines = chunk.split("\n")
            sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
            yield f"{sse_data}\n\n"
    except Exception as e:
        bt.logging.error(f"error in stream_deep_research: {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post(
    "/deep-research",
    summary="Deep research",
    response_description="A stream of search results",
)
async def deep_search(
    body: DeepResearchRequest, access_key: Annotated[str | None, Header()] = None
):
    """
    Search endpoint that accepts a JSON body with search parameters.
    """

    bt.logging.info(f"/deep-research request: {body}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return StreamingResponse(stream_deep_research(body))


@app.post(
    "/search/links/web",
    summary="Search links across web platforms",
    description="Search links using all tools except Twitter Search.",
    response_description="A JSON object mapping tool names to their search results.",
)
async def search_links_web(
    body: LinksSearchRequest, access_key: Annotated[str | None, Header()] = None
):
    bt.logging.info(f"/search/links/web request: {body}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return await handle_search_links(body, access_key, EXPECTED_ACCESS_KEY, body.tools)


@app.post(
    "/search/links/twitter",
    summary="Search links on Twitter",
    description="Search links using only Twitter Search.",
    response_description="A JSON object mapping Twitter Search to its search results.",
)
async def search_links_twitter(
    body: LinksSearchRequest, access_key: Annotated[str | None, Header()] = None
):
    bt.logging.info(f"/search/links/twitter request: {body}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return await handle_search_links(body, access_key, EXPECTED_ACCESS_KEY, body.tools)


@app.post(
    "/search/links",
    summary="Search links for all tools",
    description="Search links using all tools.",
    response_description="A JSON object mapping all tools to their search results.",
)
async def search_links(
    body: LinksSearchRequest, access_key: Annotated[str | None, Header()] = None
):
    bt.logging.info(f"/search/links request: {body}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return await handle_search_links(
        body, access_key, EXPECTED_ACCESS_KEY, available_tools
    )


class TwitterSearchRequest(BaseModel):
    query: Optional[str] = ""
    sort: Optional[str] = "Top"
    user: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lang: Optional[str] = None
    verified: Optional[bool] = None
    blue_verified: Optional[bool] = None
    is_quote: Optional[bool] = None
    is_video: Optional[bool] = None
    is_image: Optional[bool] = None
    min_retweets: Optional[int] = None
    min_replies: Optional[int] = None
    min_likes: Optional[int] = None
    count: Optional[conint(le=100)] = 20

    uid: Optional[int] = None


@app.post(
    "/twitter/search",
    summary="Twitter basic filter Search",
    description="Using filters to search for precise results from Twitter.",
    response_model=List[TwitterScraperTweet],
)
async def advanced_twitter_search(
    request: TwitterSearchRequest, access_key: Annotated[str | None, Header()] = None
):
    """
    Perform an advanced Twitter search using multiple filtering parameters.

    Returns:
        List[TwitterScraperTweet]: A list of fetched tweets.
    """

    bt.logging.info(f"/twitter/search request: {request}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    try:
        bt.logging.info("Advanced Twitter search initiated with organic approach.")

        query_dict = request.model_dump()

        # Collect all yielded synapses from organic
        final_synapses = []

        async for synapse in neu.basic_scraper_validator.organic(
            query=query_dict, uid=request.uid
        ):
            final_synapses.append(synapse)

        # Transform final synapses into a flattened list of tweets
        all_tweets = []

        for syn in final_synapses:
            # Each synapse (if successful) should have a 'results' field of TwitterScraperTweet
            if hasattr(syn, "results") and isinstance(syn.results, list):
                all_tweets.extend(syn.results)

        return all_tweets
    except Exception as e:
        bt.logging.error(f"Error in advanced_twitter_search: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class TwitterURLSearchRequest(BaseModel):
    urls: List[str]

    uid: Optional[int] = Field(
        default=None,
    )


@app.post(
    "/twitter/urls",
    summary="Fetch Tweets by URLs",
    description="Fetch details of multiple tweets using their URLs.",
    response_model=List[TwitterScraperTweet],
)
async def get_tweets_by_urls(
    request: TwitterURLSearchRequest, access_key: Annotated[str | None, Header()] = None
):
    """
    Fetch the details of multiple tweets using their URLs.

    Parameters:
        urls (List[str]): A list of tweet URLs.

    Returns:
        List[TwitterScraperTweet]: A list of fetched tweets.
    """

    bt.logging.info(f"/twitter/urls request: {request}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    results = []

    try:
        urls = list(set(request.urls))

        bt.logging.info(f"Fetching tweets for URLs: {urls}")

        results = await neu.basic_scraper_validator.twitter_urls_search(
            urls, uid=request.uid
        )
    except Exception as e:
        bt.logging.error(f"Error fetching tweets by URLs: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    if results:
        return results
    else:
        raise HTTPException(status_code=404, detail="Tweets not found")


@app.get(
    "/twitter/{id}",
    summary="Fetch Tweet by ID",
    description="Fetch details of a tweet using its unique tweet ID.",
    response_model=TwitterScraperTweet,
)
async def get_tweet_by_id(
    id: str = Path(..., description="The unique ID of the tweet to fetch"),
    uid: Optional[int] = Query(default=None),
    access_key: Annotated[str | None, Header()] = None,
):
    """
    Fetch the details of a tweet by its ID.

    Returns:
        List[TwitterScraperTweet]: A list containing the tweet details.
    """

    bt.logging.info(f"/twitter/id request: id={id}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    results = []

    try:
        bt.logging.info(f"Fetching tweet with ID: {id}")

        results = await neu.basic_scraper_validator.twitter_id_search(id, uid=uid)
    except Exception as e:
        bt.logging.error(f"Error fetching tweet by ID: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    if results:
        return results[0]
    else:
        raise HTTPException(status_code=404, detail="Tweet not found")


@app.get(
    "/web/search",
    summary="Web Search",
    description="Search the web using a query with options for result count and pagination.",
    response_model=WebSearchResultList,
)
async def web_search_endpoint(
    query: str = Query(
        ..., description="The search query string, e.g., 'latest news on AI'."
    ),
    num: int = Query(10, le=100, description="The maximum number of results to fetch."),
    start: int = Query(
        0, description="The number of results to skip (used for pagination)."
    ),
    uid: Optional[int] = Query(default=None),
    access_key: Annotated[str | None, Header()] = None,
):
    """
    Perform a web search using the given query, number of results, and start index.

    Parameters:
        query (str): The search query string.
        num (int): The maximum number of results to fetch.
        start (int): The number of results to skip (for pagination).
        uid (Optional[int]): The unique identifier of the target axon. Defaults to None.

    Returns:
        List[WebSearchResult]: A list of web search results.
    """

    bt.logging.info(f"/web/search request: query={query}, num={num}, start={start}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    try:
        bt.logging.info(
            f"Performing web search with query: '{query}', num: {num}, start: {start}"
        )

        # Collect all yielded synapses from organic
        final_synapses = []

        async for synapse in neu.basic_web_scraper_validator.organic(
            query={"query": query, "num": num, "start": start}, uid=uid
        ):
            final_synapses.append(synapse)

        # Transform final synapses into a flattened list of links
        results = []

        for syn in final_synapses:
            # Each synapse (if successful) should have a 'results' field of WebSearchResult
            if hasattr(syn, "results") and isinstance(syn.results, list):
                results.extend(syn.results)

        return {"data": results}
    except Exception as e:
        bt.logging.error(f"Error in web search: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class PeopleSearchRequest(BaseModel):
    query: str = Field(
        ...,
        title="Query",
        description="The query string to fetch results for. Example: 'Former investment bankers who transitioned into startup CFO roles'. Immutable.",
    )

    num: int = Field(
        10,
        title="Number of Results",
        description="The maximum number of results to fetch. Immutable.",
    )

    criteria: Optional[List[str]] = Field(
        ...,
        title="Search criteria",
        description="Search criteria based on query.",
    )

    uid: Optional[int] = Field(
        default=None,
    )


async def stream_people_search(data: PeopleSearchRequest):
    try:
        query = {
            "query": data.query,
            "num": data.num,
            "criteria": data.criteria,
        }

        bt.logging.info(f"People search query: {query}")

        merged_chunks = ""

        async for response in neu.people_search_validator.organic(query, uid=data.uid):
            # Decode the chunk if necessary and merge
            chunk = str(response)  # Assuming response is already a string
            merged_chunks += chunk
            lines = chunk.split("\n")
            sse_data = "\n".join(f"data: {line if line else ' '}" for line in lines)
            yield f"{sse_data}\n\n"
    except Exception as e:
        bt.logging.error(f"error in stream_people_search: {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post(
    "/people/search",
    summary="People Search",
    description="Search the people using a query",
    response_model=PeopleSearchResultList,
)
async def people_search_endpoint(
    request: PeopleSearchRequest,
    access_key: Annotated[str | None, Header()] = None,
):
    """
    Perform a people search using the given query.

    Parameters:
        query (str): The search query string.

    Returns:
        List[PeopleSearchResult]: A list of people search results.
    """

    bt.logging.info(f"/people/search request: {request}")

    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return StreamingResponse(stream_people_search(request))


@app.get("/")
async def health_check(access_key: Annotated[str | None, Header()] = None):
    if access_key != EXPECTED_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")

    return {"status": "healthy", "version": __version__}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Datura API",
        version="1.0.0",
        summary="API for searching across multiple platforms",
        routes=app.routes,
        servers=[
            {"url": "http://localhost:8005", "description": "Datura API"},
        ],
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
