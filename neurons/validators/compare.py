import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

MATCH_THRESHOLD = 0.9
MIN_PRECISION = 0.9
SHINGLE_SIZE = 3

CHURN_TYPES = frozenset({"home", "listing"})
CHURN_MIN_PRECISION = 0.7
CHURN_MIN_RECALL = 0.7
CHURN_MIN_GROWTH = 0.8
CHURN_MAX_GROWTH = 1.1

NUMBER_FLOOR = 0.5
COUNTED_NUMBERS = 5

NON_WORD = re.compile(r"[^\w\s]|_")
NUMBER = re.compile(r"\d+")


@dataclass
class Similarity:
    containment: float
    jaccard: float
    length_ratio: float
    title_match: bool
    score: float
    precision: float = 0.0
    recall: float = 0.0
    growth: float = 0.0
    numbers: float = 1.0


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(NON_WORD.sub(" ", folded).split())


def shingle_list(tokens: list[str], size: int) -> list[int]:
    if len(tokens) < size:
        return [hash(tuple(tokens))] if tokens else []
    return [hash(tuple(tokens[i : i + size])) for i in range(len(tokens) - size + 1)]


def shingles(tokens: list[str], size: int) -> set[int]:
    return set(shingle_list(tokens, size))


def numbers_kept(text: str, live: str) -> float:
    mine = Counter(NUMBER.findall(text))
    total = sum(mine.values())
    if total < COUNTED_NUMBERS:
        return 1.0
    return round(sum((mine & Counter(NUMBER.findall(live))).values()) / total, 4)


def supported(items: list[int], known: set[int]) -> float:
    return sum(item in known for item in items) / len(items)


def similarity(a: str, b: str, title_a: str = "", title_b: str = "") -> Similarity:
    tokens_a = normalize(a).split()
    tokens_b = normalize(b).split()
    title_match = bool(normalize(title_a)) and normalize(title_a) == normalize(title_b)

    if not tokens_a and not tokens_b:
        return Similarity(1.0, 1.0, 1.0, title_match, 1.0, 1.0, 1.0, 1.0)
    if not tokens_a or not tokens_b:
        return Similarity(0.0, 0.0, 0.0, title_match, 0.0)

    size = SHINGLE_SIZE if min(len(tokens_a), len(tokens_b)) >= SHINGLE_SIZE else 1
    list_a = shingle_list(tokens_a, size)
    list_b = shingle_list(tokens_b, size)
    shingles_a = set(list_a)
    shingles_b = set(list_b)
    shared = len(shingles_a & shingles_b)
    containment = shared / min(len(shingles_a), len(shingles_b))
    jaccard = shared / len(shingles_a | shingles_b)
    length_ratio = min(len(tokens_a), len(tokens_b)) / max(len(tokens_a), len(tokens_b))
    score = 0.5 * containment + 0.25 * jaccard + 0.25 * containment * length_ratio

    return Similarity(
        containment=round(containment, 4),
        jaccard=round(jaccard, 4),
        length_ratio=round(length_ratio, 4),
        title_match=title_match,
        score=round(score, 4),
        precision=round(supported(list_a, shingles_b), 4),
        recall=round(supported(list_b, shingles_a), 4),
        growth=round(len(tokens_a) / len(tokens_b), 4),
        numbers=numbers_kept(a, b),
    )


def is_match(
    sim: Similarity, threshold: float = MATCH_THRESHOLD, page_type: str = ""
) -> bool:
    if sim.numbers < NUMBER_FLOOR:
        return False
    if sim.score >= threshold and sim.precision >= MIN_PRECISION:
        return True
    # Listings swap teasers between fetches: allow a balanced swap.
    return (
        page_type in CHURN_TYPES
        and sim.precision >= CHURN_MIN_PRECISION
        and sim.recall >= CHURN_MIN_RECALL
        and CHURN_MIN_GROWTH <= sim.growth <= CHURN_MAX_GROWTH
    )
