import random
from typing import Any, Iterable, List, TypeVar

T = TypeVar("T")


def seeded_sample(population: List[T], k: int, seed: int | None = None) -> List[T]:
    """Return a random sample using a deterministic seed when available.

    When validators all receive the same scoring_seed for a given
    (hour, uid, search_type), they will select identical items,
    ensuring consistent scoring and improving vTrust.

    Falls back to standard random.sample when no seed is provided
    (backwards compatible with validators running an older API).
    """

    if seed is not None:
        rng = random.Random(seed)
        return rng.sample(population, k)

    return random.sample(population, k)


def sort_links_for_sampling(links: Iterable[str]) -> List[str]:
    """Normalize link order before seeded sampling."""
    return sorted(links)


def sort_tweets_for_sampling(tweets: Iterable[Any]) -> List[Any]:
    """Normalize tweet order before seeded sampling using tweet id."""

    def sort_key(tweet: Any):
        if isinstance(tweet, dict):
            return str(tweet.get("id", ""))
        return str(getattr(tweet, "id", ""))

    return sorted(tweets, key=sort_key)
