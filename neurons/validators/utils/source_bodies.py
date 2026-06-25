"""Build/normalize the cited source bodies the groundedness judge reads."""

import html
import random
import re

from neurons.validators.utils.response_checks import (
    extract_markdown_links,
    normalize_source_url,
    source_key,
)

_CITATION_MARKER = re.compile(r"\[\d+\]\((https?://[^)]+)\)")
_TWEET_ID = re.compile(r"/status/(\d+)")
_NON_WORD = re.compile(r"\W+")


def _normalize_for_match(text: str) -> str:
    """Casefold, unescape entities, drop non-word chars (Unicode-aware) for fuzzy containment."""
    return _NON_WORD.sub("", html.unescape(text or "").casefold())


def highlight_subset_of_body(highlights, body):
    """Return the subset of highlights actually present in body (normalized fuzzy containment)."""
    normalized_body = _normalize_for_match(body)
    if not normalized_body:
        return []
    verified = []
    for highlight in highlights or []:
        normalized = _normalize_for_match(highlight)
        if normalized and normalized in normalized_body:
            verified.append(highlight)
    return verified


def cited_urls_normalized(summary):
    return {normalize_source_url(u) for _, u in extract_markdown_links(summary)}


def sample_cited_and_uncited(urls, cited_norm, max_cited, max_total):
    cited = [u for u in urls if normalize_source_url(u) in cited_norm]
    other = [u for u in urls if normalize_source_url(u) not in cited_norm]
    picks = random.sample(cited, min(max_cited, len(cited)))
    if other:
        picks.append(random.choice(other))
    while len(picks) < max_total:
        pool = [u for u in urls if u not in picks]
        if not pool:
            break
        picks.append(random.choice(pool))
    return picks


def dedup_richest(bodies):
    best = {}
    for b in bodies:
        key = source_key(b.get("url", ""))
        if key not in best or len(b.get("text") or "") > len(
            best[key].get("text") or ""
        ):
            best[key] = b
    return list(best.values())


def align_citation_markers(summary, bodies):
    index_by_key = {source_key(b.get("url", "")): i for i, b in enumerate(bodies, 1)}

    def renumber(match):
        i = index_by_key.get(source_key(match.group(1)))
        return f"[{i}]({match.group(1)})" if i else match.group(0)

    return _CITATION_MARKER.sub(renumber, summary)


def tweet_relevance_text(tweet):
    text = getattr(tweet, "text", "") or ""
    quote = getattr(tweet, "quote", None)
    quoted = (getattr(quote, "text", "") or "").strip() if quote else ""
    if not quoted:
        return text
    handle = getattr(getattr(quote, "user", None), "username", None)
    header = f"Quoted tweet (@{handle}):" if handle else "Quoted tweet:"
    return f"{text}\n\n{header} {quoted}"


def collect_cited_bodies(response, cited_urls):
    """Resolve each cited URL to a body already fetched by the relevance models —
    web pages from validator_links, tweets from validator_tweets (matched by id)."""
    link_map = {}
    for link in getattr(response, "validator_links", None) or []:
        url = link.get("link") or link.get("url") or ""
        text = link.get("body") or ""
        if url and text:
            link_map[source_key(url)] = {
                "url": url,
                "title": link.get("title", ""),
                "text": text,
            }

    tweet_map = {}
    for tweet in getattr(response, "validator_tweets", None) or []:
        text = tweet_relevance_text(tweet)
        tid = getattr(tweet, "id", None)
        if tid and text:
            author = getattr(getattr(tweet, "user", None), "username", None)
            tweet_map[str(tid)] = {
                "url": getattr(tweet, "url", "") or "",
                "title": f"Tweet by @{author}" if author else "Tweet",
                "text": text,
            }

    bodies = []
    for u in cited_urls:
        hit = link_map.get(source_key(u))
        if not hit:
            m = _TWEET_ID.search(u)
            if m:
                hit = tweet_map.get(m.group(1))
        if hit and hit.get("text"):
            bodies.append({**hit, "url": u})
    return dedup_richest(bodies)
