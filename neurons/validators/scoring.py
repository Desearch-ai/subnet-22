from __future__ import annotations

import bisect
import hashlib
import io
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from desearch.extraction import Page, extract, looks_blocked
from desearch.extraction.schema import PAGE_SCHEMA, sha256_hex
from neurons.validators.compare import (
    CHURN_TYPES,
    MATCH_THRESHOLD,
    MIN_PRECISION,
    NUMBER_FLOOR,
    Similarity,
    is_match,
    normalize,
    similarity,
)

MATCHED = "matched"
MISMATCHED = "mismatched"
UNVERIFIABLE = "unverifiable"
ERRORS_CONFIRMED = "errors_confirmed"
ERRORS_UNCONFIRMED = "errors_unconfirmed"
NOT_FETCHED = "not_fetched"
OUTCOMES = (
    MATCHED,
    MISMATCHED,
    UNVERIFIABLE,
    ERRORS_CONFIRMED,
    ERRORS_UNCONFIRMED,
    NOT_FETCHED,
)
COVERAGE = 0.85
MIN_SAMPLES = 10
SAMPLE_RATE = 0.10
# Share of compared samples that must match; set from calibration runs.
MATCH_RATIO = 0.8
REEXTRACT_TOLERANCE = 0.2
INTEGRITY_CAP = 50
REEXTRACT_THRESHOLD = 0.95
SHORT_TEXT_CHARS = 50
UNCONFIRMED_SHARE = 0.5
MAX_ROW_BYTES = 6_000_000
MAX_ROW_GROUP_BYTES = 512_000_000
TOO_LARGE_BYTES = 5_000_000
# Provider failures say nothing about the page.
PROVIDER_STATUSES = frozenset({400, 401, 402, 407, 429})
UNREACHED = frozenset({"timeout", "connect", "dns", "tls", "other", "not_fetched"})
OWN_IP = "own_ip"
SCRAPINGDOG = "scrapingdog"
ERROR_SLOTS = 2
ERROR_SHARE = 0.2
SNIPPET_CHARS = 500
WINDOW_CHARS = 250
STRUCTURED = (
    "page_type",
    "title",
    "description",
    "lang",
    "canonical",
    "published",
    "author",
    "json_ld_types",
    "headings",
)
# A forged one of these would pass on the text alone; the live page must agree.
CHECKED_FIELDS = ("title", "published", "canonical", "author")
# These never change between two fetches of the same page, so a difference is a forgery.
HARD_FIELDS = frozenset({"published", "canonical"})


@dataclass
class FetchedPage:
    status: int
    html: str = ""
    error: str = ""
    via: str = ""


def read_pages(data: bytes | None, assigned: int | None = None) -> list[dict] | None:
    if not data:
        return None
    try:
        parquet = pq.ParquetFile(io.BytesIO(data))
        if not parquet.schema_arrow.equals(PAGE_SCHEMA):
            return None
        if assigned is not None and upload_too_big(parquet.metadata, assigned):
            return None
        table = parquet.read(columns=[n for n in PAGE_SCHEMA.names if n != "html"])
        # The footer is the miner's word; the decoded columns are not.
        if assigned is not None and table.nbytes > max(assigned, 1) * MAX_ROW_BYTES:
            return None
        return table.to_pylist()
    except Exception:
        return None


class UploadTooLarge(Exception):
    pass


def upload_too_big(metadata, assigned: int) -> bool:
    """Checked before decompressing anything."""
    groups = [
        sum(
            metadata.row_group(g).column(c).total_uncompressed_size
            for c in range(metadata.num_columns)
        )
        for g in range(metadata.num_row_groups)
    ]
    rows = max(assigned, 1)
    return (
        metadata.num_rows > 2 * rows
        or sum(groups) > rows * MAX_ROW_BYTES
        or max(groups, default=0) > MAX_ROW_GROUP_BYTES
    )


def attach_html(data: bytes, rows: list[dict], wanted: list[dict]) -> None:
    """Reads HTML only for the wanted rows, one row group at a time."""
    parquet = pq.ParquetFile(io.BytesIO(data))
    sizes = [
        parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)
    ]
    starts = list(itertools.accumulate(sizes, initial=0))
    position = {id(row): index for index, row in enumerate(rows)}
    by_group: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row in wanted:
        index = position[id(row)]
        group = bisect.bisect_right(starts, index) - 1
        by_group[group].append((index - starts[group], row))
    for group, entries in sorted(by_group.items()):
        html = parquet.read_row_group(group, columns=["html"]).column("html")
        # The footer is the miner's word; the decoded column is not.
        if html.nbytes > min(MAX_ROW_GROUP_BYTES, len(html) * MAX_ROW_BYTES):
            raise UploadTooLarge(f"row group {group} decodes to {html.nbytes} bytes")
        for offset, row in entries:
            row["html"] = html[offset].as_py()


def load_upload(
    data: bytes | None, assigned: list[str], seed: str
) -> tuple[list[dict] | None, dict[str, dict]]:
    """The upload's rows, with HTML only where the integrity check reads it."""
    rows = read_pages(data, len(set(assigned)))
    if rows is None:
        return None, {}
    kept = rows_by_url(rows, assigned)[0]
    try:
        attach_html(data, rows, [kept[url] for url in integrity_urls(kept, seed)])
    except UploadTooLarge:
        return None, {}
    return rows, kept


def sample_seed(task_id: str, validator_hotkey: str, salt: str = "") -> str:
    return hashlib.sha256(f"{task_id}{validator_hotkey}{salt}".encode()).hexdigest()


def sample_order(urls, seed: str) -> list[str]:
    return sorted(
        urls, key=lambda url: hashlib.sha256(f"{seed}{url}".encode()).digest()
    )


def rows_by_url(rows: list[dict], assigned: list[str]) -> tuple[dict[str, dict], int]:
    wanted = set(assigned)
    kept: dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        url = row["url"]
        if url not in wanted:
            continue
        if url in kept:
            duplicates += 1
        else:
            kept[url] = row
    return kept, duplicates


def same_site(a: str, b: str) -> bool:
    try:
        host_a = (urlsplit(a).hostname or "").removeprefix("www.")
        host_b = (urlsplit(b).hostname or "").removeprefix("www.")
    except ValueError:
        return False
    if not host_a or not host_b:
        return False
    return (
        host_a == host_b
        or host_a.endswith("." + host_b)
        or host_b.endswith("." + host_a)
    )


def extraction_url(row: dict) -> str:
    final = row["final_url"]
    return final if final and same_site(row["url"], final) else row["url"]


def derived_from_html(row: dict) -> bool:
    html = row.get("html")
    if (
        not html
        or len(html) > TOO_LARGE_BYTES
        or sha256_hex(html) != row["html_sha256"]
    ):
        return False
    if sha256_hex(row["text"]) != row["text_sha256"]:
        return False
    decoded = html.decode("utf-8", "replace")
    for url in dict.fromkeys((extraction_url(row), row["url"])):
        page = extract(decoded, url)
        if all(getattr(page, name) == row[name] for name in STRUCTURED) and is_match(
            similarity(row["text"], page.text), REEXTRACT_THRESHOLD
        ):
            return True
    return False


def integrity_urls(
    kept: dict[str, dict], seed: str, cap: int = INTEGRITY_CAP
) -> list[str]:
    return [url for url in sample_order(kept, seed) if kept[url]["error"] is None][:cap]


def check_integrity(kept: dict[str, dict], seed: str) -> tuple[int, list[str]]:
    """How many rows were re-extracted, and those whose text was not from their HTML."""
    checked = integrity_urls(kept, seed)
    return len(checked), [url for url in checked if not derived_from_html(kept[url])]


def sample_count(rows: int, minimum: int = MIN_SAMPLES) -> int:
    return max(minimum, math.ceil(rows * SAMPLE_RATE))


def pick_samples(kept: dict[str, dict], seed: str, size: int) -> list[str]:
    order = sample_order(kept, seed)
    size = min(size, len(order))
    if size <= 0:
        return []
    errors = [url for url in order if kept[url]["error"] is not None]
    fine = [url for url in order if kept[url]["error"] is None]
    # Error rows say little about forgery, so sample mostly text.
    error_slots = max(ERROR_SLOTS, int(size * ERROR_SHARE))
    from_errors = min(len(errors), max(1, min(size // 2, error_slots)))
    picked = errors[:from_errors] + fine[: size - from_errors]
    return picked + errors[from_errors : from_errors + size - len(picked)]


def needs_rendered_check(
    kept: dict[str, dict], fetched: Mapping[str, FetchedPage]
) -> list[str]:
    return [
        url
        for url, page in fetched.items()
        if kept[url]["error"] is None
        and judge_sample(kept[url], page)["outcome"] != MATCHED
    ]


def cleared_by_render(
    kept: dict[str, dict], rendered: Mapping[str, FetchedPage]
) -> dict[str, FetchedPage]:
    """Rendering can only clear a sample, never fail one."""
    return {
        url: page
        for url, page in rendered.items()
        if judge_sample(kept[url], page)["outcome"] == MATCHED
    }


def fetch_failed(fetched: FetchedPage) -> bool:
    """No evidence either way: the page was never reached, or the provider failed."""
    if fetched.error in UNREACHED or fetched.error.startswith("fetcher_"):
        return True
    if fetched.via == OWN_IP:
        return False
    return fetched.status in PROVIDER_STATUSES or fetched.status >= 500


def unreachable_both_ways(fetched: FetchedPage) -> bool:
    """Our own page only comes back failed when ScrapingDog failed on it too."""
    return fetched.via == OWN_IP and fetched.error in UNREACHED


def looks_like_html(html: str) -> bool:
    head = html[:4096].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head


def why_unverifiable(fetched: FetchedPage, page: Page) -> str:
    if fetched.error:
        return fetched.error
    if fetched.status != 200:
        return f"http_{fetched.status}"
    if not fetched.html.strip():
        return "empty"
    # Our own fetches already passed the miner's HTML check; only ScrapingDog's did not.
    if fetched.via != OWN_IP and not looks_like_html(fetched.html):
        return "not_html"
    if len(fetched.html) > TOO_LARGE_BYTES // 4 and (
        len(fetched.html.encode("utf-8", "replace")) > TOO_LARGE_BYTES
    ):
        return "too_large"
    if looks_blocked(fetched.status, fetched.html, page.text):
        return "blocked"
    if not page.text.strip():
        return "empty"
    return ""


def texts_match(sim: Similarity, miner: Page, validator: Page, assigned_type) -> bool:
    short = (
        len(miner.text) < SHORT_TEXT_CHARS and len(validator.text) < SHORT_TEXT_CHARS
    )
    if short and (normalize(miner.title) or normalize(validator.title)):
        return sim.title_match
    return is_match(sim) or is_match(sim, page_type=assigned_type())


def judge_sample(row: dict, fetched: FetchedPage | None) -> dict:
    fetched = fetched or FetchedPage(0, error="not_fetched")
    readable = fetched.html and len(fetched.html) <= TOO_LARGE_BYTES
    page = extract(fetched.html, extraction_url(row)) if readable else Page()

    def assigned_type() -> str:
        # Keyed on the assigned URL, not the miner's final_url.
        if extraction_url(row) == row["url"]:
            return page.page_type
        return extract(fetched.html, row["url"]).page_type

    problem = why_unverifiable(fetched, page)
    miner = Page(title=row["title"] or "", text=row["text"] or "")
    sim = None
    page_type = ""

    if row["error"] and unreachable_both_ways(fetched):
        outcome = ERRORS_CONFIRMED
    elif fetch_failed(fetched):
        outcome = NOT_FETCHED
    elif row["error"] is None:
        if problem:
            outcome = UNVERIFIABLE
        else:
            sim = similarity(miner.text, page.text, miner.title, page.title)
            page_type = assigned_type()
            outcome = (
                MATCHED
                if texts_match(sim, miner, page, lambda: page_type)
                else MISMATCHED
            )
    elif problem:
        outcome = ERRORS_CONFIRMED
    else:
        outcome = ERRORS_UNCONFIRMED

    differs = fields_differ(row, page) if outcome == MATCHED else []
    if HARD_FIELDS & set(differs):
        outcome = MISMATCHED
    return {
        "url": row["url"],
        "outcome": outcome,
        "similarity": sim.score if sim else 0.0,
        "precision": sim.precision if sim else None,
        "recall": sim.recall if sim else None,
        "growth": sim.growth if sim else None,
        "page_type": page_type,
        "miner_status": row["status"],
        "validator_status": fetched.status,
        "miner_chars": len(miner.text),
        "validator_chars": len(page.text),
        "miner_error": row["error"] or "",
        "validator_error": problem,
        "via": fetched.via,
        "fields_differ": differs,
        "why": mismatch_reason(
            outcome, row, problem, sim, page_type, len(page.text), differs
        ),
    }


def fields_differ(row: dict, page: Page) -> list[str]:
    return [
        name
        for name in CHECKED_FIELDS
        if normalize(row.get(name) or "") != normalize(getattr(page, name))
    ]


def format_score(value: float) -> str:
    """Round down so a value never reads above its floor."""
    return f"{math.floor(value * 100) / 100:.2f}"


def mismatch_reason(
    outcome: str,
    row: dict,
    problem: str,
    sim: Similarity | None,
    page_type: str,
    their_chars: int,
    differs: list[str] = (),
) -> str:
    claimed = row["error"]
    if outcome == NOT_FETCHED:
        return f"the validator could not fetch the page ({problem}), so it is no evidence either way"
    if outcome == UNVERIFIABLE:
        return f"the validator could not read the page either ({problem})"
    if outcome == ERRORS_CONFIRMED:
        return f"the miner reported {claimed} and the validator was refused too ({problem})"
    if outcome == ERRORS_UNCONFIRMED:
        return f"the miner reported {claimed} but the validator read {their_chars} characters"
    if sim is None:
        return ""
    if differs:
        fields = ", ".join(differs)
        if outcome == MATCHED:
            return f"the text matches but the live page's {fields} differs, so the row is not published"
        return f"the text matches but the live page's {fields} differs, which does not happen between two fetches"
    if outcome == MATCHED:
        if sim.score >= MATCH_THRESHOLD and sim.precision >= MIN_PRECISION:
            return f"the text matches (score {format_score(sim.score)})"
        return f"a {page_type} page that changed within the allowance (kept {format_score(sim.recall)} of the live text)"
    if sim.numbers < NUMBER_FLOOR:
        return (
            f"only {format_score(sim.numbers)} of the figures in the miner's text are on the live page,"
            f" under the {NUMBER_FLOOR:.2f} floor"
        )
    if sim.score >= MATCH_THRESHOLD:
        return (
            f"score {format_score(sim.score)} is fine but only {format_score(sim.precision)} of the miner's text is"
            f" on the live page, under the {MIN_PRECISION:.2f} floor"
        )
    if page_type in CHURN_TYPES:
        return (
            f"a {page_type} page that changed too much: score {format_score(sim.score)},"
            f" kept {format_score(sim.recall)} of the live text, length {sim.growth:.2f}x"
        )
    return f"the texts differ: score {format_score(sim.score)}, under the {MATCH_THRESHOLD:.2f} threshold"


def decide_verdict(
    counts: dict,
    assigned: int,
    checked: int,
    match_ratio: float,
    unconfirmed_share: float,
) -> tuple[str, str]:
    compared = counts[MATCHED] + counts[MISMATCHED]
    judged_errors = counts[ERRORS_CONFIRMED] + counts[ERRORS_UNCONFIRMED]
    if counts["duplicates"] or counts.get("unassigned"):
        return "fail", "extra_rows"
    if assigned and counts["returned"] / assigned < COVERAGE:
        return "fail", "coverage"
    if counts["reextract_mismatch"] > REEXTRACT_TOLERANCE * checked:
        return "fail", "text_not_from_html"
    if counts["sampled"] and counts[NOT_FETCHED] * 2 >= counts["sampled"]:
        return "retry", "provider"
    if not compared and not judged_errors:
        return "void", "inconclusive"
    # Too little the validator could read either way to pay or to fail on.
    if counts["sampled"] and counts[UNVERIFIABLE] * 2 >= counts["sampled"]:
        return "void", "unverifiable"
    if compared and counts[MATCHED] / compared < match_ratio:
        return "fail", "content_mismatch"
    if unconfirmed_share > UNCONFIRMED_SHARE:
        return "fail", "errors_not_reproducible"
    return "pass", "ok"


def empty_result(wanted: set[str], reason: str) -> dict:
    counts = dict.fromkeys(
        (
            "returned",
            "duplicates",
            "unassigned",
            "sampled",
            *OUTCOMES,
            "reextract_mismatch",
        ),
        0,
    )
    return {
        **counts,
        "missing": len(wanted),
        "error_rows": 0,
        "integrity_checked": 0,
        "verdict": "fail",
        "reason": reason,
        "samples": [],
        "rejected": [],
    }


def score(
    rows: list[dict] | None,
    assigned: list[str],
    fetched: Mapping[str, FetchedPage],
    seed: str,
    min_samples: int,
    match_ratio: float,
) -> dict:
    wanted = set(assigned)
    if rows is None:
        return empty_result(wanted, "unreadable")

    kept, duplicates = rows_by_url(rows, assigned)
    checked, not_from_html = check_integrity(kept, seed)
    samples = [
        judge_sample(kept[url], fetched.get(url))
        for url in pick_samples(kept, seed, sample_count(len(kept), min_samples))
    ]
    outcomes = Counter(sample["outcome"] for sample in samples)
    counts = {
        "returned": len(kept),
        "missing": len(wanted) - len(kept),
        "duplicates": duplicates,
        "unassigned": sum(1 for row in rows if row["url"] not in wanted),
        "sampled": len(samples),
        **{outcome: outcomes[outcome] for outcome in OUTCOMES},
        "reextract_mismatch": len(not_from_html),
    }
    error_rows = sum(1 for row in kept.values() if row["error"] is not None)
    judged_errors = outcomes[ERRORS_CONFIRMED] + outcomes[ERRORS_UNCONFIRMED]
    unconfirmed_share = (
        error_rows / len(kept) * outcomes[ERRORS_UNCONFIRMED] / judged_errors
        if kept and judged_errors
        else 0.0
    )
    verdict, reason = decide_verdict(
        counts, len(wanted), checked, match_ratio, unconfirmed_share
    )
    doubted = {
        sample["url"]
        for sample in samples
        if sample["outcome"] == MISMATCHED or sample["fields_differ"]
    }
    return {
        **counts,
        "error_rows": error_rows,
        "integrity_checked": checked,
        "verdict": verdict,
        "reason": reason,
        "samples": samples,
        "rejected": sorted(doubted | set(not_from_html)),
    }


def first_difference(left: str, right: str) -> int | None:
    if left == right:
        return None
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def text_window(text: str, at: int | None) -> str:
    if at is None:
        return ""
    return text[max(0, at - WINDOW_CHARS) : at + WINDOW_CHARS]


def url_log(
    rows: list[dict] | None,
    samples: list[dict],
    texts: dict[str, str],
    rejected: list[str] = (),
) -> list[dict]:
    by_url = {sample["url"]: sample for sample in samples}
    kept_out = set(rejected)
    details = []
    for row in rows or []:
        miner_text = row["text"] or ""
        detail = {
            "url": row["url"],
            "status": row["status"],
            "error": row["error"],
            "text_chars": len(miner_text),
            "sampled": row["url"] in by_url,
            "rejected": row["url"] in kept_out,
        }
        sample = by_url.get(row["url"])
        if sample:
            theirs = texts.get(row["url"], "")
            at = first_difference(miner_text, theirs) if theirs else None
            detail |= {
                "outcome": sample["outcome"],
                "why": sample.get("why"),
                "similarity": sample.get("similarity"),
                "precision": sample.get("precision"),
                "recall": sample.get("recall"),
                "growth": sample.get("growth"),
                "validator_error": sample.get("validator_error"),
                "via": sample.get("via", ""),
                "miner_chars": sample.get("miner_chars"),
                "validator_chars": sample.get("validator_chars"),
                "miner_snippet": miner_text[:SNIPPET_CHARS],
                "validator_snippet": theirs[:SNIPPET_CHARS],
                "diff_at": at,
                "miner_window": text_window(miner_text, at),
                "validator_window": text_window(theirs, at),
            }
        details.append(detail)
    return details
