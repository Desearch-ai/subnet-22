import pydantic
import bittensor as bt
import typing
import json
import asyncio
from typing import List, Dict, Optional, Any
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum
from aiohttp import ClientResponse
import traceback
from datura.services.twitter_utils import TwitterUtils
from datura.services.web_search_utils import WebSearchUtils
import traceback
from datura.synapse import Synapse, StreamingSynapse


class IsAlive(Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )


class TwitterPromptAnalysisResult(BaseModel):
    api_params: Dict[str, Any] = {}
    keywords: List[str] = []
    hashtags: List[str] = []
    user_mentions: List[str] = []

    def fill(self, response: Dict[str, Any]):
        if "api_params" in response:
            self.api_params = response["api_params"]
        if "keywords" in response:
            self.keywords = response["keywords"]
        if "hashtags" in response:
            self.hashtags = response["hashtags"]
        if "user_mentions" in response:
            self.user_mentions = response["user_mentions"]

    def __str__(self):
        return f"Query String: {self.api_params}, Keywords: {self.keywords}, Hashtags: {self.hashtags}, User Mentions: {self.user_mentions}"


class TwitterScraperMedia(BaseModel):
    media_url: str = ""
    type: str = ""


class TwitterScraperEntitiesUserMention(BaseModel):
    id_str: str
    name: str
    screen_name: str
    indices: List[int]


class TwitterScraperEntitiesSymbol(BaseModel):
    indices: List[int]
    text: str


class TwitterScraperEntitiesMediaAdditionalInfo(BaseModel):
    monetizable: Optional[bool] = None
    source_user: Optional[Dict[str, Any]] = None


class TwitterScraperEntitiesMediaExtAvailability(BaseModel):
    status: Optional[str] = None


class MediaSize(BaseModel):
    w: int
    h: int
    resize: str


class Rect(BaseModel):
    x: int
    y: int
    w: int
    h: int


class TwitterScraperEntitiesMediaSizes(BaseModel):
    large: Optional[MediaSize] = None
    medium: Optional[MediaSize] = None
    small: Optional[MediaSize] = None
    thumb: Optional[MediaSize] = None


class TwitterScraperEntitiesMediaOriginalInfo(BaseModel):
    height: int
    width: int
    focus_rects: Optional[List[Rect]] = []


class TwitterScraperEntitiesMediaAllowDownloadStatus(BaseModel):
    allow_download: Optional[bool] = None


class TwitterScraperEntitiesMediaVideoInfoVariant(BaseModel):
    content_type: str
    url: str
    bitrate: Optional[int] = None


class TwitterScraperEntitiesMediaVideoInfo(BaseModel):
    duration_millis: Optional[int] = None
    aspect_ratio: Optional[List[int]] = []
    variants: Optional[List[TwitterScraperEntitiesMediaVideoInfoVariant]] = []


class TwitterScraperEntitiesMediaResult(BaseModel):
    media_key: str


class TwitterScraperEntitiesMediaResults(BaseModel):
    result: Optional[TwitterScraperEntitiesMediaResult] = None


class TwitterScraperEntitiesMediaFeature(BaseModel):
    faces: Optional[List[Rect]] = []


class TwitterScraperEntitiesMediaFeatures(BaseModel):
    large: Optional[TwitterScraperEntitiesMediaFeature] = None
    medium: Optional[TwitterScraperEntitiesMediaFeature] = None
    small: Optional[TwitterScraperEntitiesMediaFeature] = None
    orig: Optional[TwitterScraperEntitiesMediaFeature] = None


class TwitterScraperEntitiesMedia(BaseModel):
    display_url: Optional[str] = None
    expanded_url: Optional[str] = None
    id_str: Optional[str] = None
    indices: Optional[List[int]] = None
    media_key: Optional[str] = None
    media_url_https: Optional[str] = None
    type: Optional[str] = None
    url: Optional[str] = None
    additional_media_info: Optional[TwitterScraperEntitiesMediaAdditionalInfo] = None
    ext_media_availability: Optional[TwitterScraperEntitiesMediaExtAvailability] = None
    features: Optional[TwitterScraperEntitiesMediaFeatures] = None
    sizes: Optional[TwitterScraperEntitiesMediaSizes] = None
    original_info: Optional[TwitterScraperEntitiesMediaOriginalInfo] = None
    allow_download_status: Optional[TwitterScraperEntitiesMediaAllowDownloadStatus] = (
        None
    )
    video_info: Optional[TwitterScraperEntitiesMediaVideoInfo] = None
    media_results: Optional[TwitterScraperEntitiesMediaResults] = None


class TwitterScraperEntityUrl(BaseModel):
    display_url: str
    expanded_url: str
    url: str
    indices: List[int]


class TwitterScraperEntities(BaseModel):
    hashtags: Optional[List[TwitterScraperEntitiesSymbol]] = []
    media: Optional[List[TwitterScraperEntitiesMedia]] = []
    symbols: Optional[List[TwitterScraperEntitiesSymbol]] = []
    timestamps: Optional[List[Any]] = []
    urls: Optional[List[TwitterScraperEntityUrl]] = []
    user_mentions: Optional[List[TwitterScraperEntitiesUserMention]] = []


class TwitterScraperExtendedEntities(BaseModel):
    media: Optional[List[TwitterScraperEntitiesMedia]] = []


class TwitterScraperUserEntitiesDescription(BaseModel):
    urls: Optional[List[TwitterScraperEntityUrl]] = []


class TwitterScraperUserEntities(BaseModel):
    description: Optional[TwitterScraperUserEntitiesDescription] = None
    url: Optional[TwitterScraperUserEntitiesDescription] = None


class TwitterScraperUser(BaseModel):
    id: str
    url: Optional[str] = None
    name: Optional[str] = None
    username: str
    created_at: Optional[str] = None
    description: Optional[str] = None
    favourites_count: Optional[int] = None
    followers_count: Optional[int] = None
    listed_count: Optional[int] = None
    media_count: Optional[int] = None
    profile_image_url: Optional[str] = None
    profile_banner_url: Optional[str] = None
    statuses_count: Optional[int] = None
    verified: Optional[bool] = None
    is_blue_verified: Optional[bool] = None
    entities: Optional[TwitterScraperUserEntities] = None
    can_dm: Optional[bool] = None
    can_media_tag: Optional[bool] = None
    location: Optional[str] = None
    pinned_tweet_ids: Optional[List[str]] = None


class TwitterScraperTweet(BaseModel):
    user: TwitterScraperUser
    id: str
    text: str
    reply_count: int
    retweet_count: int
    like_count: int
    quote_count: int
    bookmark_count: int
    url: Optional[str]
    created_at: str
    media: Optional[List[TwitterScraperMedia]] = []
    is_quote_tweet: Optional[bool]
    is_retweet: Optional[bool]
    lang: Optional[str] = None
    conversation_id: Optional[str] = None
    in_reply_to_screen_name: Optional[str] = None
    in_reply_to_status_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    quoted_status_id: Optional[str] = None
    quote: Optional["TwitterScraperTweet"] = None
    display_text_range: Optional[List[int]] = None
    entities: Optional[TwitterScraperEntities] = None
    extended_entities: Optional[TwitterScraperExtendedEntities] = None


class ScraperTextRole(str, Enum):
    INTRO = "intro"
    TWITTER_SUMMARY = "twitter_summary"
    SEARCH_SUMMARY = "search_summary"
    REDDIT_SUMMARY = "reddit_summary"
    HACKER_NEWS_SUMMARY = "hacker_news_summary"
    FINAL_SUMMARY = "summary"


class ResultType(str, Enum):
    ONLY_LINKS = "ONLY_LINKS"
    LINKS_WITH_SUMMARIES = "LINKS_WITH_SUMMARIES"
    LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"


class Model(str, Enum):
    NOVA = "NOVA"
    ORBIT = "ORBIT"
    HORIZON = "HORIZON"


class ContextualRelevance(Enum):
    HIGH = "HIGH"  # Exact match, deep context understanding
    MEDIUM = "MEDIUM"  # Partially relevant, missing some context
    LOW = "LOW"  # Weak or loose connection to query


class ScoringModel(str, Enum):
    OPENAI_GPT4_MINI = "openai/gpt-4-mini"
    QWEN_QWEN2_5_CODER_32B_INSTRUCT = "Qwen/Qwen2.5-Coder-32B-Instruct"
    UNSLOTH_MISTRAL_SMALL_24B_INSTRUCT_2501 = "unsloth/Mistral-Small-24B-Instruct-2501"
    DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_QWEN_32B = (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )


class ScraperStreamingSynapse(StreamingSynapse):
    scoring_model: ScoringModel = pydantic.Field(
        ScoringModel.OPENAI_GPT4_MINI,
        title="scoring model",
        description="The llm model to score synapse result.",
    )

    prompt: str = pydantic.Field(
        ...,
        title="Prompt",
        description="The initial input or question provided by the user to guide the scraping and data collection process.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. This attribute is mutable and can be updated.",
    )

    model: Model = pydantic.Field(
        Model.NOVA,
        title="model",
        description="The model to define the max execution time.",
    )

    system_message: Optional[str] = pydantic.Field(
        "",
        title="Sysmtem Message",
        description="System message for formatting the response.",
    )

    tools: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Tools",
        description="A list of tools specified by user to use to answer question.",
    )

    start_date: Optional[str] = pydantic.Field(
        None,
        title="Start Date",
        description="The start date for the search query.",
    )

    end_date: Optional[str] = pydantic.Field(
        None,
        title="End Date",
        description="The end date for the search query.",
    )

    date_filter_type: Optional[str] = pydantic.Field(
        None,
        title="Date filter enum",
        description="The date filter enum.",
    )

    language: Optional[str] = pydantic.Field(
        "en",
        title="Language",
        description="Language specified by user.",
    )

    region: Optional[str] = pydantic.Field(
        "us",
        title="Region",
        description="Region specified by user.",
    )

    google_date_filter: Optional[str] = pydantic.Field(
        "qdr:w",
        title="Date Filter",
        description="Date filter specified by user.",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    validator_links: Optional[List[Dict]] = pydantic.Field(
        default_factory=list, title="Links", description="Fetched Links Data."
    )

    miner_link_scores: Optional[Dict[str, ContextualRelevance]] = pydantic.Field(
        default_factory=dict,
        title="Miner link scores",
    )

    miner_tweets: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="Miner Tweets",
        description="Optional JSON object containing tweets data from the miner.",
    )

    search_completion_links: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of links extracted from search summary text.",
    )

    completion_links: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Links Content",
        description="A list of JSON objects representing the extracted links content from the tweets.",
    )

    search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Search Results",
        description="Optional JSON object containing search results from SERP",
    )

    wikipedia_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Wikipedia Search Results",
        description="Optional JSON object containing search results from Wikipedia",
    )

    youtube_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="YouTube Search Results",
        description="Optional JSON object containing search results from YouTube",
    )

    arxiv_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Arxiv Search Results",
        description="Optional JSON object containing search results from Arxiv",
    )

    reddit_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Reddit Search Results",
        description="Optional JSON object containing search results from Reddit",
    )

    hacker_news_search_results: Optional[Any] = pydantic.Field(
        default_factory=dict,
        title="Hacker News Search Results",
        description="Optional JSON object containing search results from Hacker News",
    )

    text_chunks: Optional[Dict[str, List[str]]] = pydantic.Field(
        default_factory=dict,
        title="Text Chunks",
    )

    is_synthetic: Optional[bool] = pydantic.Field(
        False,
        title="Is Synthetic",
        description="A boolean flag to indicate if the prompt is synthetic.",
    )

    @property
    def texts(self) -> Dict[str, str]:
        """Returns a dictionary of texts, containing a role (twitter summary, search summary, reddit summary, hacker news summary, final summary) and content."""
        texts = {}

        for key in self.text_chunks:
            texts[key] = "".join(self.text_chunks[key])

        return texts

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    max_items: Optional[int] = pydantic.Field(
        None,
        title="Max Results",
        description="The maximum number of results to be returned per query",
    )

    result_type: ResultType = pydantic.Field(
        None,
        title="Result Type",
        description="The result type for miners",
    )

    def set_tweets(self, data: any):
        self.tweets = data

    def get_twitter_completion(self) -> Optional[str]:
        return self.texts.get(ScraperTextRole.TWITTER_SUMMARY.value, "")

    def get_search_completion(self) -> Dict[str, str]:
        """Gets the search completion text from the texts dictionary based on tools used."""

        completions = {}

        if any(
            tool in self.tools
            for tool in [
                "Web Search",
                "Wikipedia Search",
                "Youtube Search",
                "ArXiv Search",
            ]
        ):
            search_summary = self.texts.get(
                ScraperTextRole.SEARCH_SUMMARY.value, ""
            ).strip()
            completions[ScraperTextRole.SEARCH_SUMMARY.value] = search_summary

        if "Reddit Search" in self.tools:
            reddit_summary = self.texts.get(
                ScraperTextRole.REDDIT_SUMMARY.value, ""
            ).strip()
            completions[ScraperTextRole.REDDIT_SUMMARY.value] = reddit_summary

        if "Hacker News Search" in self.tools:
            hacker_news_summary = self.texts.get(
                ScraperTextRole.HACKER_NEWS_SUMMARY.value, ""
            ).strip()
            completions[ScraperTextRole.HACKER_NEWS_SUMMARY.value] = hacker_news_summary

        links_per_completion = 10
        links_expected = len(completions) * links_per_completion

        return completions, links_expected

    def get_all_completions(self) -> Dict[str, str]:
        completions, _ = self.get_search_completion()

        if "Twitter Search" in self.tools:
            completions[ScraperTextRole.TWITTER_SUMMARY.value] = (
                self.get_twitter_completion()
            )

        return completions

    def get_search_links(self) -> List[str]:
        """Extracts web links from each summary making sure to filter by domain for each tool used.
        In Reddit and Hacker News Search, the links are filtered by domains.
        In search summary part, if Web Search is used, the links are allowed from any domain,
        Otherwise search summary will only look for Wikipedia, ArXiv, Youtube links.
        Returns list of all links and links per each summary role.
        """

        completions, _ = self.get_search_completion()
        all_links = []
        links_per_summary = {}

        for key, value in completions.items():
            links = []

            if key == ScraperTextRole.REDDIT_SUMMARY.value:
                links.extend(WebSearchUtils.find_links_by_domain(value, "reddit.com"))
            elif key == ScraperTextRole.HACKER_NEWS_SUMMARY.value:
                links.extend(
                    WebSearchUtils.find_links_by_domain(value, "news.ycombinator.com")
                )
            elif key == ScraperTextRole.SEARCH_SUMMARY.value:
                if any(tool in self.tools for tool in ["Web Search"]):
                    links.extend(WebSearchUtils.find_links(value))
                else:
                    if "Wikipedia Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "wikipedia.org")
                        )
                    if "ArXiv Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "arxiv.org")
                        )
                    if "Youtube Search" in self.tools:
                        links.extend(
                            WebSearchUtils.find_links_by_domain(value, "youtube.com")
                        )

            all_links.extend(links)
            links_per_summary[key] = links

        return all_links, links_per_summary

    async def process_streaming_response(self, response: StreamingResponse):
        if self.completion is None:
            self.completion = ""

        buffer = ""  # Initialize an empty buffer to accumulate data across chunks

        try:
            async for chunk in response.content.iter_any():
                chunk_str = chunk.decode("utf-8", errors="ignore")

                # Attempt to parse the chunk as JSON, updating the buffer with remaining incomplete JSON data
                json_objects, buffer = extract_json_chunk(
                    chunk_str, response, self.axon.hotkey, buffer
                )
                for json_data in json_objects:
                    content_type = json_data.get("type")

                    if content_type == "text":
                        text_content = json_data.get("content", "")
                        role = json_data.get("role")

                        if role not in self.text_chunks:
                            self.text_chunks[role] = []

                        self.text_chunks[role].append(text_content)

                        yield json.dumps(
                            {"type": "text", "role": role, "content": text_content}
                        )

                    elif content_type == "completion":
                        completion = json_data.get("content", "")
                        self.completion = completion

                        yield json.dumps({"type": "completion", "content": completion})

                    elif content_type == "tweets":
                        tweets_json = json_data.get("content", "[]")
                        self.miner_tweets = tweets_json
                        yield json.dumps({"type": "tweets", "content": tweets_json})

                    elif content_type == "search":
                        search_json = json_data.get("content", "{}")
                        self.search_results = search_json
                        yield json.dumps({"type": "search", "content": search_json})

                    elif content_type == "wikipedia_search":
                        search_json = json_data.get("content", "{}")
                        self.wikipedia_search_results = search_json
                        yield json.dumps(
                            {"type": "wikipedia_search", "content": search_json}
                        )

                    elif content_type == "youtube_search":
                        search_json = json_data.get("content", "{}")
                        self.youtube_search_results = search_json
                        yield json.dumps(
                            {"type": "youtube_search", "content": search_json}
                        )

                    elif content_type == "arxiv_search":
                        search_json = json_data.get("content", "{}")
                        self.arxiv_search_results = search_json
                        yield json.dumps(
                            {"type": "arxiv_search", "content": search_json}
                        )

                    elif content_type == "reddit_search":
                        search_json = json_data.get("content", "{}")
                        self.reddit_search_results = search_json
                        yield json.dumps(
                            {"type": "reddit_search", "content": search_json}
                        )

                    elif content_type == "hacker_news_search":
                        search_json = json_data.get("content", "{}")
                        self.hacker_news_search_results = search_json
                        yield json.dumps(
                            {"type": "hacker_news_search", "content": search_json}
                        )

        except json.JSONDecodeError as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            bt.logging.debug(
                f"process_streaming_response: Host: {host}:{port}, hotkey: {hotkey}, ERROR: json.JSONDecodeError: {e}, "
            )
        except (TimeoutError, asyncio.exceptions.TimeoutError) as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            print(
                f"process_streaming_response TimeoutError: Host: {host}:{port}, hotkey: {hotkey}, Error: {e}"
            )
        except Exception as e:
            port = response.real_url.port
            host = response.real_url.host
            hotkey = self.axon.hotkey
            error_details = traceback.format_exc()
            bt.logging.debug(
                f"process_streaming_response: Host: {host}:{port}, hotkey: {hotkey}, ERROR: {e}, DETAILS: {error_details}, chunk: {chunk}"
            )

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: ClientResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        completion_links = TwitterUtils().find_twitter_links(self.completion)
        search_completion_links, _ = self.get_search_links()

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "prompt": self.prompt,
            # "model": self.model,
            "completion": self.completion,
            "miner_tweets": self.miner_tweets,
            "search_results": self.search_results,
            "wikipedia_search_results": self.wikipedia_search_results,
            "youtube_search_results": self.youtube_search_results,
            "arxiv_search_results": self.arxiv_search_results,
            "hacker_news_search_results": self.hacker_news_search_results,
            "reddit_search_results": self.reddit_search_results,
            # "prompt_analysis": self.prompt_analysis.dict(),
            "completion_links": completion_links,
            "search_completion_links": search_completion_links,
            "texts": self.texts,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_filter_type": self.date_filter_type,
            "tools": self.tools,
            "max_execution_time": self.max_execution_time,
            "text_chunks": self.text_chunks,
            "result_type": self.result_type,
            "model": self.model,
            "max_items": self.max_items,
            "language": self.language,
            "region": self.region,
            "system_message": self.system_message,
        }

    class Config:
        arbitrary_types_allowed = True


def extract_json_chunk(chunk, response, hotkey, buffer=""):
    """
    Extracts JSON objects from a chunk of data, handling cases where JSON objects are split across multiple chunks.

    :param chunk: The current chunk of data to process.
    :param response: The response object, used for logging.
    :param buffer: A buffer holding incomplete JSON data from previous chunks.
    :return: A tuple containing a list of extracted JSON objects and the updated buffer.
    """
    buffer += chunk  # Add the current chunk to the buffer
    json_objects = []

    while True:
        try:
            json_obj, end = json.JSONDecoder(strict=False).raw_decode(buffer)
            json_objects.append(json_obj)
            buffer = buffer[end:]
        except json.JSONDecodeError as e:
            if e.pos == len(buffer):
                # Reached the end of the buffer without finding a complete JSON object
                break
            elif e.msg.startswith("Unterminated string"):
                # Incomplete JSON object at the end of the chunk
                break
            else:
                # Invalid JSON data encountered
                port = response.real_url.port
                host = response.real_url.host
                bt.logging.debug(
                    f"Host: {host}:{port}; hotkey: {hotkey}; Failed to decode JSON object: {e} from {buffer}"
                )
                break

    return json_objects, buffer


class WebSearchResult(BaseModel):
    title: str
    snippet: str
    link: str
    date: Optional[str] = None


class WebSearchValidatorResult(WebSearchResult):
    html_content: Optional[str] = None
    html_text: Optional[str] = None


class WebSearchResultList(BaseModel):
    data: List[WebSearchResult]


class WebSearchSynapse(Synapse):
    """A class to represent web search synapse"""

    query: str = pydantic.Field(
        "",
        title="Query",
        description="The query string to fetch results for. Example: 'latest news on AI'. Immutable.",
        allow_mutation=False,
    )

    num: int = pydantic.Field(
        10,
        title="Number of Results",
        description="The maximum number of results to fetch. Immutable.",
        allow_mutation=False,
    )

    start: int = pydantic.Field(
        0,
        title="Start Index",
        description="The number of results to skip (used for pagination). Immutable.",
        allow_mutation=False,
    )

    is_synthetic: Optional[bool] = pydantic.Field(
        False,
        title="Is Synthetic",
        description="A boolean flag to indicate if the prompt is synthetic.",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    results: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="Web",
        description="Fetched Web Data.",
    )

    validator_links: Optional[List[WebSearchValidatorResult]] = pydantic.Field(
        default_factory=list,
        title="Validator Web",
        description="Fetched validator Web Data.",
    )

    def deserialize(self) -> str:
        return self


class LinkedinExperienceItem(BaseModel):
    company_id: Optional[str] = None
    company_link: Optional[str] = None
    title: str
    subtitle: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[str] = None


class LinkedinEducationItem(BaseModel):
    company_id: Optional[str] = None
    company_link: Optional[str] = None
    title: str
    subtitle: Optional[str] = None
    caption: Optional[str] = None


class LinkedinLanguageItem(BaseModel):
    title: str
    caption: str


class PeopleSearchResult(BaseModel):
    link: str
    first_name: str
    last_name: str
    full_name: str
    title: str
    summary: str
    avatar: str
    experiences: List[LinkedinExperienceItem]
    educations: List[LinkedinEducationItem]
    languages: List[LinkedinLanguageItem]

    relevance_summary: Optional[str] = None
    criteria_summary: Optional[List[str]] = []
    extra_information: Optional[Dict[str, Any]] = None


class PeopleSearchResultList(BaseModel):
    data: List[PeopleSearchResult]


class PeopleSearchSynapse(Synapse):
    """A class to represent web search synapse"""

    query: str = pydantic.Field(
        "",
        title="Query",
        description="The query string to fetch results for. Example: 'Former investment bankers who transitioned into startup CFO roles'. Immutable.",
        allow_mutation=False,
    )

    is_synthetic: Optional[bool] = pydantic.Field(
        False,
        title="Is Synthetic",
        description="A boolean flag to indicate if the prompt is synthetic.",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    criteria: Optional[List[str]] = pydantic.Field(
        default_factory=list,
        title="Search criteria",
        description="Search criteria based on query.",
    )

    results: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="Web",
        description="Fetched Web Data.",
    )

    validator_results: Optional[List[PeopleSearchResult]] = pydantic.Field(
        default_factory=list,
        title="Validator Web",
        description="Fetched validator Web Data.",
    )

    def deserialize(self) -> str:
        return self


class TwitterSearchSynapse(Synapse):
    """A class to represent Twitter Advanced Search Synapse"""

    query: str = pydantic.Field(
        ...,
        title="Query",
        description="Search query string, e.g., 'from:user bitcoin'.",
        allow_mutation=False,
    )

    sort: Optional[str] = pydantic.Field(
        None,
        title="Sort",
        description="Sort by 'Top' or 'Latest'.",
        allow_mutation=False,
    )

    user: Optional[str] = pydantic.Field(
        None,
        title="User",
        description="Search for tweets from a specific user.",
        allow_mutation=False,
    )

    count: int = pydantic.Field(
        20,
        title="Count",
        description="Count of tweets to fetch.",
        allow_mutation=False,
    )

    start_date: Optional[str] = pydantic.Field(
        None,
        title="Start Date",
        description="Start date in UTC (e.g., '2021-12-31').",
        allow_mutation=False,
    )

    end_date: Optional[str] = pydantic.Field(
        None,
        title="End Date",
        description="End date in UTC (e.g., '2021-12-31').",
        allow_mutation=False,
    )

    lang: Optional[str] = pydantic.Field(
        None,
        title="Language",
        description="Language filter (e.g., 'en').",
        allow_mutation=False,
    )

    verified: Optional[bool] = pydantic.Field(
        None,
        title="Verified",
        description="Filter for verified accounts.",
        allow_mutation=False,
    )

    blue_verified: Optional[bool] = pydantic.Field(
        None,
        title="Blue Verified",
        description="Filter for blue verified accounts.",
        allow_mutation=False,
    )

    is_quote: Optional[bool] = pydantic.Field(
        None,
        title="Quote",
        description="Filter for quote tweets.",
        allow_mutation=False,
    )

    is_video: Optional[bool] = pydantic.Field(
        None,
        title="Video",
        description="Filter for tweets with videos.",
        allow_mutation=False,
    )

    is_image: Optional[bool] = pydantic.Field(
        None,
        title="Image",
        description="Filter for tweets with images.",
        allow_mutation=False,
    )

    min_retweets: Optional[int] = pydantic.Field(
        None,
        title="Minimum Retweets",
        description="Minimum number of retweets.",
        allow_mutation=False,
    )

    min_replies: Optional[int] = pydantic.Field(
        None,
        title="Minimum Replies",
        description="Minimum number of replies.",
        allow_mutation=False,
    )

    min_likes: Optional[int] = pydantic.Field(
        None,
        title="Minimum Likes",
        description="Minimum number of likes.",
        allow_mutation=False,
    )

    is_synthetic: Optional[bool] = pydantic.Field(
        False,
        title="Is Synthetic",
        description="A boolean flag to indicate if the prompt is synthetic.",
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="validator tweets",
        description="Fetched validator Tweets Data.",
    )

    results: Optional[List[Dict[str, Any]]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    def deserialize(self) -> str:
        return self


class TwitterIDSearchSynapse(Synapse):
    """A class to represent Twitter ID Advanced Search Synapse"""

    id: str = pydantic.Field(
        ...,
        title="id",
        description="Search id string, tweet ID to fetch",
        allow_mutation=False,
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="validator tweets",
        description="Fetched validator Tweets Data.",
    )

    results: Optional[List[Dict]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    def deserialize(self) -> str:
        return self


class TwitterURLsSearchSynapse(Synapse):
    """A class to represent Twitter URLs Advanced Search Synapse"""

    urls: List[str] = pydantic.Field(
        ...,
        title="URLs",
        description="A list of tweet URLs to fetch.",
        allow_mutation=False,
    )

    max_execution_time: Optional[int] = pydantic.Field(
        None,
        title="Max Execution Time (timeout)",
        description="Maximum time to execute concrete request",
    )

    validator_tweets: Optional[List[TwitterScraperTweet]] = pydantic.Field(
        default_factory=list,
        title="validator tweets",
        description="Fetched validator Tweets Data.",
    )

    results: Optional[List[Dict]] = pydantic.Field(
        default_factory=list,
        title="tweets",
        description="Fetched Tweets Data.",
    )

    def deserialize(self) -> str:
        return self
