"""Pure-code response checks shared between cheap penalties and deep reward models."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytz

from desearch.protocol import ScraperStreamingSynapse, ScraperTextRole
from desearch.services.web_search_utils import WebSearchUtils

MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADER_HASH_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)
HEADER_BOLD_PATTERN = re.compile(r"\*\*[^*]+\*\*")

AI_SEARCH_RESULT_FIELDS = ("search_results",)
AI_ALL_RESULT_FIELDS = ("miner_tweets",) + AI_SEARCH_RESULT_FIELDS


def extract_markdown_links(text: str) -> List[Tuple[str, str]]:
    """Returns list of (link_text, url) tuples from markdown."""
    return MARKDOWN_LINK_PATTERN.findall(text or "")


def check_markdown_structure(text: str) -> Tuple[bool, List[str]]:
    """Reject `#` headers and empty responses; bold headers no longer required."""
    issues = []
    if HEADER_HASH_PATTERN.search(text or ""):
        issues.append("Uses # headers instead of **")
    if not (text or "").strip():
        issues.append("Empty response")
    return len(issues) == 0, issues


def normalize_source_url(url: str) -> str:
    """Normalize for comparison: lowercase, no www., no trailing slash. Scheme
    and query string are kept — miner is accountable for matching those."""
    url = (url or "").strip().lower()
    if url.startswith("https://www."):
        url = "https://" + url[len("https://www.") :]
    elif url.startswith("http://www."):
        url = "http://" + url[len("http://www.") :]
    return WebSearchUtils.remove_trailing_slash(url)


_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "yclid",
        "twclid",
        "igshid",
        "igsh",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
        "mkt_tok",
        "si",
        "feature",
        "spm",
        "scm",
    }
)


def _is_tracking_param(key: str) -> bool:
    key = key.lower()
    return key.startswith("utm_") or key in _TRACKING_PARAMS


def source_key(url: str) -> str:
    """Identity key for a URL: base + content query params, tracking params dropped."""
    parts = urlsplit(url or "")
    base = normalize_source_url(
        urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    )
    kept = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(k)
    )
    return f"{base}?{urlencode(kept)}" if kept else base


def collect_summary_sources(response: ScraperStreamingSynapse) -> set:
    """Collect every URL the miner returned that a summary link could legitimately reference."""
    sources: set = set()

    if response.miner_tweets:
        for tweet in response.miner_tweets:
            if not isinstance(tweet, dict):
                continue
            username = tweet.get("user", {}).get("username", "")
            tweet_id = tweet.get("id", "")
            if username and tweet_id:
                sources.add(
                    normalize_source_url(f"https://x.com/{username}/status/{tweet_id}")
                )

    for field in AI_SEARCH_RESULT_FIELDS:
        for result in getattr(response, field, []) or []:
            link = (
                result.get("link")
                if isinstance(result, dict)
                else getattr(result, "link", None)
            )
            if link:
                sources.add(normalize_source_url(link))

    return sources


def verify_summary_links(response: ScraperStreamingSynapse) -> Tuple[int, int]:
    """Returns (verified_count, total_count) for markdown links in the final summary."""
    summary = (
        response.texts.get(ScraperTextRole.FINAL_SUMMARY.value, "")
        if response.texts
        else ""
    )
    links = [url for _, url in extract_markdown_links(summary)]
    if not links:
        return 0, 0
    sources = collect_summary_sources(response)
    verified = sum(1 for link in links if normalize_source_url(link) in sources)
    return verified, len(links)


def parse_tweet_date(value: str) -> Optional[datetime]:
    """Parse Twitter's '%a %b %d %H:%M:%S %z %Y' or ISO 8601."""
    if not value:
        return None
    for fmt in (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%d_%H:%M:%S_%Z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=pytz.UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=pytz.UTC)
    except ValueError:
        return None


def parse_synapse_date(value: str) -> Optional[datetime]:
    """Parse a synapse start_date/end_date string."""
    return parse_tweet_date(value)


def tweet_date_in_range(
    tweet_created_at: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> bool:
    """True if tweet date falls within [start_date, end_date], inclusive. Missing bounds skip that side."""
    dt = parse_tweet_date(tweet_created_at)
    if dt is None:
        return False
    if start_date:
        start = parse_synapse_date(start_date)
        if start and dt < start:
            return False
    if end_date:
        end = parse_synapse_date(end_date)
        if end and dt > end:
            return False
    return True


def first_duplicate_id(
    items: List[Dict[str, Any]], key: str = "id", normalize=None
) -> Optional[Any]:
    """Returns the first duplicated key value, or None if all unique."""
    seen: set = set()
    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if value is None:
            continue
        if normalize is not None:
            value = normalize(value)
        if value in seen:
            return value
        seen.add(value)
    return None


def is_descending_by_created_at(items: List[Dict[str, Any]]) -> bool:
    """For Twitter sort=Latest — strictly non-increasing created_at."""
    previous: Optional[datetime] = None
    for item in items or []:
        created = item.get("created_at") if isinstance(item, dict) else None
        if not created:
            return False
        dt = parse_tweet_date(created)
        if dt is None:
            return False
        if previous is not None and dt > previous:
            return False
        previous = dt
    return True
