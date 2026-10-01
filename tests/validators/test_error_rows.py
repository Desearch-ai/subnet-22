import random
from dataclasses import asdict

import pytest

from desearch.extraction import Page, extract
from desearch.extraction.schema import sha256_hex
from neurons.validators.scoring import (
    FetchedPage,
    pick_samples,
    rows_by_url,
    sample_seed,
    score,
)

WORDS = (
    "river market council school harbour winter garden railway museum county "
    "library festival bridge engineer farmer journal harvest mountain village "
    "orchestra factory hospital province theatre election pilgrim lantern "
    "copper meadow tunnel sailor treaty glacier castle monsoon canyon ferry "
    "blacksmith chapel quarry vineyard archive telescope compass voyage "
    "estuary pottery granary lighthouse prairie citadel"
).split()

SEED = sample_seed("task-1", "validator")
SAMPLE = 5
MATCH = 0.8


def page(url: str) -> str:
    rng = random.Random(url)
    paragraphs = "".join(
        "<p>" + " ".join(rng.choice(WORDS) for _ in range(24)) + ".</p>"
        for _ in range(6)
    )
    return (
        f"<html><head><title>Report {url}</title></head>"
        f"<body><main><h1>Report {url}</h1>{paragraphs}</main></body></html>"
    )


def ok_row(url: str) -> dict:
    html = page(url)
    extracted = extract(html, url)
    return {
        "url": url,
        "final_url": url,
        "status": 200,
        "error": None,
        "html": html.encode(),
        "html_sha256": sha256_hex(html.encode()),
        **asdict(extracted),
        "text_sha256": sha256_hex(extracted.text),
    }


def error_row(url: str, error: str = "blocked") -> dict:
    return {
        "url": url,
        "final_url": url,
        "status": 403,
        "error": error,
        "html": None,
        "html_sha256": "",
        **asdict(Page(page_type="")),
        "text_sha256": "",
    }


def batch(total: int, errors: int) -> tuple[list[str], list[dict]]:
    urls = [f"https://site.example/story/{i}" for i in range(total)]
    rows = [error_row(u) if i < errors else ok_row(u) for i, u in enumerate(urls)]
    return urls, rows


def live(urls) -> dict:
    return {u: FetchedPage(200, page(u)) for u in urls}


def run(urls, rows, fetched) -> dict:
    return score(rows, urls, fetched, SEED, SAMPLE, MATCH)


def test_the_generated_pages_are_long_enough_to_judge():
    assert len(ok_row("https://site.example/story/0")["text"]) >= 200


def test_blocked_rows_the_validator_could_fetch_are_unconfirmed_but_do_not_fail():
    urls, rows = batch(20, errors=4)
    result = run(urls, rows, live(urls))
    assert result["verdict"] == "pass"
    assert result["errors_unconfirmed"] >= 1 and result["errors_confirmed"] == 0


def test_errors_that_reproduce_are_confirmed():
    urls, rows = batch(20, errors=4)
    fetched = live(urls[4:]) | {u: FetchedPage(404) for u in urls[:4]}
    result = run(urls, rows, fetched)
    assert result["verdict"] == "pass"
    assert result["errors_confirmed"] >= 1 and result["errors_unconfirmed"] == 0


def test_a_split_sample_judges_each_error_on_its_own():
    urls, rows = batch(20, errors=4)
    kept, _ = rows_by_url(rows, urls)
    sampled_errors = [u for u in pick_samples(kept, SEED, SAMPLE) if kept[u]["error"]]
    assert len(sampled_errors) == 2
    fetched = live(urls) | {sampled_errors[0]: FetchedPage(404)}
    result = run(urls, rows, fetched)
    assert result["verdict"] == "pass"
    assert (result["errors_confirmed"], result["errors_unconfirmed"]) == (1, 1)


def test_claiming_most_of_a_batch_failed_when_it_was_fetchable_fails():
    urls, rows = batch(20, errors=12)
    result = run(urls, rows, live(urls))
    assert result["verdict"] == "fail"
    assert result["reason"] == "errors_not_reproducible"


def test_a_miner_that_fetched_nothing_fails():
    urls, rows = batch(20, errors=20)
    result = run(urls, rows, live(urls))
    assert result["verdict"] == "fail"


def test_errors_on_a_thin_page_the_validator_loaded_are_unconfirmed():
    urls, rows = batch(20, errors=4)
    thin = "<html><body><p>ok</p></body></html>"
    fetched = live(urls[4:]) | {u: FetchedPage(200, thin) for u in urls[:4]}
    result = run(urls, rows, fetched)
    assert result["errors_confirmed"] == 0 and result["errors_unconfirmed"] == 2
    assert result["verdict"] == "pass"


def test_errors_scrapingdog_could_not_judge_are_not_counted_either_way():
    urls, rows = batch(20, errors=4)
    fetched = live(urls[4:]) | {u: FetchedPage(0, error="timeout") for u in urls[:4]}
    result = run(urls, rows, fetched)
    assert result["not_fetched"] == 2
    assert result["errors_confirmed"] == result["errors_unconfirmed"] == 0
    assert result["verdict"] == "pass"


def test_one_caught_mismatch_does_not_fail_the_task():
    urls, rows = batch(20, errors=0)
    kept, _ = rows_by_url(rows, urls)
    forged = pick_samples(kept, SEED, SAMPLE)[0]
    fetched = live(urls) | {forged: FetchedPage(200, page(forged + "/other"))}
    result = run(urls, rows, fetched)
    assert (result["matched"], result["mismatched"]) == (4, 1)
    assert result["verdict"] == "pass"


@pytest.mark.parametrize("errors, verdict", [(10, "pass"), (11, "fail")])
def test_fake_errors_fail_above_half_the_batch(errors, verdict):
    urls, rows = batch(20, errors=errors)
    result = run(urls, rows, live(urls))
    assert result["verdict"] == verdict


def test_an_unreadable_upload_fails():
    result = score(None, ["https://site.example/a"], {}, SEED, SAMPLE, MATCH)
    assert (result["verdict"], result["reason"]) == ("fail", "unreadable")
