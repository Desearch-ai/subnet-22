from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import aiohttp
import pyarrow as pa
import pytest
from aiohttp import web

from desearch.extraction import extract
from desearch.extraction.schema import sha256_hex
from desearch.fetch import ScrapingDog
from neurons.miners.rows import write_parquet
from neurons.validators import crawl, tasks
from neurons.validators.crawl import CrawlValidator, DownloadFailed
from neurons.validators.fetchers import to_page
from neurons.validators.scoring import (
    FetchedPage,
    attach_html,
    check_integrity,
    derived_from_html,
    judge_sample,
    pick_samples,
    read_pages,
    rows_by_url,
    sample_count,
    sample_seed,
)
from tests.local_http import serving
from tests.synthetic import (
    CHALLENGE,
    SEED,
    error_row,
    page_row,
    run,
    synthetic,
    synthetic_html,
    synthetic_url,
    to_parquet,
)


def test_completeness_counts_missing_duplicates_and_unassigned_rows():
    rows, assigned, fetched = synthetic(4)
    extra = page_row("https://elsewhere.example/story", synthetic_html(99))
    uploaded = [rows[0], rows[1], rows[1], rows[2], extra, extra]

    result = run(
        uploaded, assigned, {**fetched, extra["url"]: FetchedPage(200, "")}, 10
    )

    assert (result["returned"], result["missing"], result["duplicates"]) == (3, 1, 1)
    assert result["unassigned"] == 2
    assert result["integrity_checked"] == 3
    assert result["sampled"] == 3
    assert {s["url"] for s in result["samples"]} == set(assigned[:3])
    assert (result["verdict"], result["reason"]) == ("fail", "extra_rows")


@pytest.mark.parametrize("extra", ["duplicate", "unassigned"])
def test_extra_rows_fail_an_otherwise_perfect_task(extra):
    rows, assigned, fetched = synthetic(10)
    forged = page_row("https://spam.example/", synthetic_html(77))
    added = dict(rows[3], text="forged copy") if extra == "duplicate" else forged

    result = run([*rows, added], assigned, fetched)

    assert (result["verdict"], result["reason"]) == ("fail", "extra_rows")


def test_integrity_flags_an_html_hash_mismatch():
    rows, assigned, _ = synthetic(5)
    rows[0]["html_sha256"] = sha256_hex(b"something else")

    assert not derived_from_html(rows[0])
    assert check_integrity(rows_by_url(rows, assigned)[0], SEED) == (5, 1)


def test_integrity_flags_text_not_derived_from_html():
    rows, assigned, _ = synthetic(5)
    rows[0]["text"] = rows[1]["text"]
    rows[0]["text_sha256"] = rows[1]["text_sha256"]

    assert not derived_from_html(rows[0])
    assert check_integrity(rows_by_url(rows, assigned)[0], SEED) == (5, 1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("title", "Buy cheap watches"),
        ("description", "spam"),
        ("page_type", "article"),
        ("published", "1999-01-01"),
        ("canonical", "https://spam.example/"),
        ("headings", ["Injected"]),
    ],
)
def test_integrity_rejects_a_forged_structured_field(field, value):
    rows, assigned, _ = synthetic(1)
    assert derived_from_html(rows[0])
    assert rows[0][field] != value

    assert not derived_from_html(dict(rows[0], **{field: value}))


def test_integrity_rejects_a_text_hash_that_does_not_match_the_text():
    rows, _, _ = synthetic(1)
    assert not derived_from_html(dict(rows[0], text_sha256=sha256_hex("other")))


def test_integrity_counts_an_ok_row_without_html():
    rows, assigned, _ = synthetic(5)
    rows[0] = dict(rows[0], html=None, html_sha256="")

    assert check_integrity(rows_by_url(rows, assigned)[0], SEED) == (5, 1)


def test_integrity_ignores_a_final_url_on_another_site():
    rows, _, _ = synthetic(1)
    row = dict(rows[0], final_url="https://elsewhere.example/2024/01/story.html")
    assert derived_from_html(row)
    assert not derived_from_html(dict(row, page_type="article")), (
        "a borrowed final_url must not choose the page type"
    )


def test_integrity_skips_error_rows_and_caps_at_fifty():
    rows, assigned, _ = synthetic(60, errors=4)

    assert check_integrity(rows_by_url(rows, assigned)[0], SEED) == (50, 0)


def test_errors_unconfirmed_when_the_validator_gets_real_text():
    sample = judge_sample(
        error_row(synthetic_url(1), "timeout", 0), FetchedPage(200, synthetic_html(1))
    )

    assert sample["outcome"] == "errors_unconfirmed"
    assert sample["validator_chars"] >= 200


def test_an_error_against_a_thin_page_the_validator_loaded_is_unconfirmed():
    thin = "<html><head><title>Hi</title></head><body><p>Short page.</p></body></html>"

    assert (
        judge_sample(error_row(synthetic_url(1), "timeout", 0), FetchedPage(200, thin))[
            "outcome"
        ]
        == "errors_unconfirmed"
    )


def test_short_pages_are_decided_on_title():
    def tiny(title: str, body: str) -> str:
        return f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>"

    row = page_row(synthetic_url(1), tiny("Acme Widgets", "Welcome"))

    assert (
        judge_sample(row, FetchedPage(200, tiny("Acme Widgets", "Hello there")))[
            "outcome"
        ]
        == "matched"
    )
    assert (
        judge_sample(row, FetchedPage(200, tiny("Other Site", "Welcome")))["outcome"]
        == "mismatched"
    )


def test_clean_task_passes_with_ok():
    rows, assigned, fetched = synthetic(10)
    result = run(rows, assigned, fetched)

    assert (result["verdict"], result["reason"]) == ("pass", "ok")
    assert (result["sampled"], result["matched"], result["mismatched"]) == (5, 5, 0)
    assert (result["returned"], result["missing"], result["reextract_mismatch"]) == (
        10,
        0,
        0,
    )


def test_a_scrapingdog_outage_is_retried_not_passed():
    rows, assigned, _ = synthetic(10)
    result = run(
        rows, assigned, dict.fromkeys(assigned, FetchedPage(503, error="http_503"))
    )

    assert (result["verdict"], result["reason"]) == ("retry", "provider")
    assert result["not_fetched"] == 5


def test_a_task_with_nothing_judged_is_void_so_it_goes_back_out():
    rows, assigned, _ = synthetic(10)
    result = run(rows, assigned, dict.fromkeys(assigned, FetchedPage(200, CHALLENGE)))

    assert (result["verdict"], result["reason"]) == ("void", "inconclusive")
    assert result["unverifiable"] == 5


@pytest.mark.parametrize("returned, verdict", [(17, "pass"), (16, "fail")])
def test_coverage_needs_85_percent_of_assigned_urls(returned, verdict):
    rows, assigned, fetched = synthetic(20)
    result = run(rows[:returned], assigned, fetched)

    assert result["verdict"] == verdict
    assert result["reason"] == ("coverage" if verdict == "fail" else "ok")
    assert result["missing"] == 20 - returned


@pytest.mark.parametrize("tampered, verdict", [(2, "pass"), (3, "fail")])
def test_text_not_from_html_above_20_percent_of_checked_rows(tampered, verdict):
    rows, assigned, fetched = synthetic(10)
    for row in rows[:tampered]:
        row["html_sha256"] = sha256_hex(b"forged")
    result = run(rows, assigned, fetched)

    assert result["reextract_mismatch"] == tampered
    assert result["verdict"] == verdict
    assert result["reason"] == ("text_not_from_html" if verdict == "fail" else "ok")


def test_fabricated_text_fails_as_text_not_from_html():
    rows, assigned, fetched = synthetic(10)
    for n, row in enumerate(rows[:3]):
        row["text"] = extract(synthetic_html(100 + n), row["url"]).text
    result = run(rows, assigned, fetched)

    assert (result["verdict"], result["reason"]) == ("fail", "text_not_from_html")


@pytest.mark.parametrize("wrong, verdict", [(1, "pass"), (2, "fail")])
def test_content_mismatch_below_the_match_ratio(wrong, verdict):
    rows, assigned, fetched = synthetic(10)
    for n, url in enumerate(
        pick_samples(rows_by_url(rows, assigned)[0], SEED, 5)[:wrong]
    ):
        fetched[url] = FetchedPage(200, synthetic_html(200 + n))
    result = run(rows, assigned, fetched)

    assert (result["matched"], result["mismatched"]) == (5 - wrong, wrong)
    assert result["verdict"] == verdict
    assert result["reason"] == ("content_mismatch" if verdict == "fail" else "ok")


def test_errors_not_reproducible_when_most_claimed_errors_load_fine():
    rows, assigned, fetched = synthetic(5, errors=3)
    result = run(rows, assigned, fetched)

    assert (result["sampled"], result["errors_unconfirmed"], result["matched"]) == (
        5,
        3,
        2,
    )
    assert (result["verdict"], result["reason"]) == ("fail", "errors_not_reproducible")


def test_a_few_unconfirmed_errors_still_pass():
    rows, assigned, fetched = synthetic(10, errors=2)
    result = run(rows, assigned, fetched)

    assert (result["errors_unconfirmed"], result["matched"]) == (2, 3)
    assert (result["verdict"], result["reason"]) == ("pass", "ok")


def test_confirmed_errors_never_count_against_the_miner():
    rows, assigned, _ = synthetic(10, errors=10)
    result = run(
        rows, assigned, dict.fromkeys(assigned, FetchedPage(404, error="http_404"))
    )

    assert result["errors_confirmed"] == 5
    assert (result["verdict"], result["reason"]) == ("pass", "ok")


def test_coverage_is_checked_before_content():
    rows, assigned, _ = synthetic(10)
    wrong = {
        url: FetchedPage(200, synthetic_html(300 + n)) for n, url in enumerate(assigned)
    }
    result = run(rows[:5], assigned, wrong)

    assert result["mismatched"] == 5
    assert (result["verdict"], result["reason"]) == ("fail", "coverage")


def test_unreadable_upload_fails():
    _, assigned, _ = synthetic(4)
    result = run(None, assigned, {})

    assert (result["verdict"], result["reason"]) == ("fail", "unreadable")
    assert (result["returned"], result["missing"], result["sampled"]) == (0, 4, 0)


def test_read_pages_rejects_garbage_and_wrong_schemas():
    rows, _, _ = synthetic(3, errors=1)
    wrong = pa.schema([("url", pa.string()), ("text", pa.large_string())])

    assert read_pages(b"") is None
    assert read_pages(b"PAR1 not really parquet") is None
    assert read_pages(to_parquet([{"url": "u", "text": "t"}], wrong)) is None
    read = read_pages(to_parquet(rows, task_id="t", hotkey="h"))
    assert read == [{k: v for k, v in row.items() if k != "html"} for row in rows]


def test_html_is_read_only_for_the_rows_that_need_it():
    rows, _, _ = synthetic(120)
    upload = write_parquet(rows, "t", "h")
    read = read_pages(upload)
    wanted = [read[3], read[77]]

    attach_html(upload, read, wanted)

    assert [row.get("html") for row in wanted] == [rows[3]["html"], rows[77]["html"]]
    assert sum("html" in row for row in read) == 2


def test_a_row_group_too_big_to_decode_is_refused_unread(monkeypatch):
    rows, _, _ = synthetic(120)
    one_group = to_parquet(rows, task_id="t", hotkey="h")
    small_groups = write_parquet(rows, "t", "h")
    monkeypatch.setattr(
        "neurons.validators.scoring.MAX_ROW_GROUP_BYTES", len(small_groups) * 3
    )

    assert read_pages(one_group, assigned=120) is None
    assert read_pages(small_groups, assigned=120) is not None


def test_sampling_is_deterministic_per_task_and_validator():
    rows, assigned, _ = synthetic(20)
    kept = rows_by_url(rows, assigned)[0]
    shuffled = rows_by_url(rows[::-1], assigned)[0]
    other = sample_seed("task-1", "another-validator")

    assert SEED == hashlib.sha256(b"task-1validator-hotkey").hexdigest()
    assert pick_samples(kept, SEED, 5) == pick_samples(shuffled, SEED, 5)
    assert pick_samples(kept, SEED, 5) != pick_samples(kept, other, 5)
    assert len(set(pick_samples(kept, SEED, 5))) == 5


@pytest.mark.parametrize(
    "count, errors, size, sampled_errors",
    [
        (20, 6, 5, 2),
        (20, 1, 5, 1),
        (20, 0, 5, 0),
        (5, 3, 5, 3),
        (4, 4, 5, 4),
        (20, 3, 1, 1),
    ],
)
def test_sampling_includes_up_to_half_from_error_rows(
    count, errors, size, sampled_errors
):
    rows, assigned, _ = synthetic(count, errors=errors)
    kept = rows_by_url(rows, assigned)[0]
    picked = pick_samples(kept, SEED, size)

    assert len(picked) == min(size, count)
    assert sum(kept[url]["error"] is not None for url in picked) == sampled_errors


def test_scrapingdog_retries_once_on_5xx_and_timeouts_only():
    calls: list[dict] = []
    plan = {
        "https://flaky.example/": [503, 200],
        "https://slow.example/": ["timeout", "timeout"],
        "https://gone.example/": [404, 200],
        "https://down.example/": [500, 502],
    }

    async def handler(request: web.Request) -> web.Response:
        calls.append(dict(request.query))
        step = plan[request.query["url"]].pop(0)
        if step == "timeout":
            await asyncio.sleep(5)
        return web.Response(status=step, text="<html><title>ok</title></html>")

    async def fetch_all():
        async with (
            serving(handler) as endpoint,
            ScrapingDog("secret-key", timeout=0.3, endpoint=endpoint) as dog,
        ):
            return [await to_page(await dog.fetch(url)) for url in list(plan)]

    flaky, slow, gone, down = asyncio.run(fetch_all())

    assert (flaky.status, flaky.error, flaky.html) == (
        200,
        "",
        "<html><title>ok</title></html>",
    )
    assert (slow.status, slow.error) == (0, "timeout")
    assert (gone.status, gone.error) == (404, "http_404")
    assert (down.status, down.error) == (502, "http_502")
    assert len(calls) == 7
    assert all(call["dynamic"] == "false" for call in calls)
    assert all(call["api_key"] == "secret-key" for call in calls)
    assert all("secret-key" not in page.error for page in (flaky, slow, gone, down))


def test_scrapingdog_respects_its_concurrency_limit():
    active, peak = 0, 0

    async def handler(_) -> web.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return web.Response(text="<html></html>", content_type="text/html")

    async def fetch_all():
        async with (
            serving(handler) as endpoint,
            ScrapingDog("k", 2, endpoint=endpoint) as dog,
        ):
            await asyncio.gather(
                *(dog.fetch(f"https://site{n}.example/") for n in range(6))
            )

    asyncio.run(fetch_all())
    assert peak == 2


def _validate_download(status: int, content: bytes, **options) -> dict:
    async def upload(_) -> web.Response:
        return web.Response(status=status, body=content)

    async def score_task():
        async with serving(upload) as base, aiohttp.ClientSession() as http:
            job = {
                "task_id": "t1",
                "miner": "m",
                "urls": ["https://a.example/"],
                "download_url": base + "x",
            }
            return await CrawlValidator(
                SimpleNamespace(hotkey="v"), None, http, **options
            ).score_task(job)

    return asyncio.run(score_task())


def test_validator_treats_a_corrupt_upload_as_unreadable():
    result = _validate_download(200, b"garbage")
    assert (result["verdict"], result["reason"], result["missing"]) == (
        "fail",
        "unreadable",
        1,
    )


def test_a_missing_upload_is_handed_back_not_failed():
    with pytest.raises(crawl.UploadMissing):
        _validate_download(404, b"")


def test_an_oversized_upload_is_unreadable_without_reading_it_all():
    result = _validate_download(200, b"x" * 5000, max_download=1000)
    assert (result["verdict"], result["reason"]) == ("fail", "unreadable")


def test_validator_skips_a_job_when_storage_is_down(monkeypatch):
    monkeypatch.setattr(tasks, "RETRY_DELAY_S", 0)
    attempts = []

    async def down(request) -> web.Response:
        attempts.append(request.path)
        return web.Response(status=503)

    async def download():
        async with serving(down) as base, aiohttp.ClientSession() as http:
            job = {
                "task_id": "t1",
                "miner": "m",
                "urls": [],
                "download_url": base + "x",
            }
            await CrawlValidator(SimpleNamespace(hotkey="v"), None, http).download(
                job["download_url"], job["task_id"]
            )

    with pytest.raises(DownloadFailed):
        asyncio.run(download())
    assert len(attempts) == 2


def test_a_presigned_download_url_is_sent_exactly_as_signed():
    seen = []

    async def upload(request) -> web.Response:
        seen.append(request.raw_path)
        return web.Response(status=404)

    async def download():
        async with serving(upload) as base, aiohttp.ClientSession() as http:
            job = {
                "task_id": "t1",
                "miner": "m",
                "urls": [],
                "download_url": base + "b/k%2Fx.parquet?X-Amz-Credential=a%2Fb&s=%3D",
            }
            await CrawlValidator(SimpleNamespace(hotkey="v"), None, http).download(
                job["download_url"], job["task_id"]
            )

    with pytest.raises(crawl.UploadMissing):
        asyncio.run(download())
    assert seen == ["/b/k%2Fx.parquet?X-Amz-Credential=a%2Fb&s=%3D"]


@pytest.mark.parametrize(
    "rows, samples", [(0, 10), (20, 10), (100, 10), (250, 25), (1000, 100)]
)
def test_samples_grow_with_the_task_but_never_below_the_minimum(rows, samples):
    assert sample_count(rows) == samples


def test_error_rows_get_a_fifth_of_a_large_sample():
    rows, assigned, _ = synthetic(1000, errors=300)
    kept = rows_by_url(rows, assigned)[0]
    picked = pick_samples(kept, SEED, sample_count(len(kept)))

    assert len(picked) == 100
    assert sum(kept[url]["error"] is not None for url in picked) == 20


@pytest.mark.parametrize(
    "error, status, page",
    [
        ("http_4xx", 404, FetchedPage(404, error="http_404", via="own_ip")),
        ("http_4xx", 410, FetchedPage(400, error="http_400", via="own_ip")),
        ("not_html", 200, FetchedPage(200, error="not_html", via="own_ip")),
        ("empty", 200, FetchedPage(200, "<html><body><div></div></body></html>")),
    ],
)
def test_what_our_own_ip_saw_confirms_the_miners_error(error, status, page):
    row = error_row("https://site.example/a", error, status)
    assert judge_sample(row, page)["outcome"] == "errors_confirmed"


def test_a_scrapingdog_failure_is_no_evidence_either_way():
    row = error_row("https://site.example/a")
    page = FetchedPage(400, error="http_400", via="scrapingdog")
    assert judge_sample(row, page)["outcome"] == "not_fetched"


def test_a_page_with_no_text_cannot_prove_a_mismatch():
    row = page_row("https://site.example/a", synthetic_html(4))
    blank = FetchedPage(
        200, "<html><body><div id=app></div></body></html>", via="own_ip"
    )
    assert judge_sample(row, blank)["outcome"] == "unverifiable"


def test_a_page_both_our_address_and_scrapingdog_could_not_reach_confirms_it():
    row = error_row("https://site.example/a", "timeout", 0)
    ours = FetchedPage(0, error="timeout", via="own_ip")
    theirs = FetchedPage(0, error="timeout", via="scrapingdog")

    assert judge_sample(row, ours)["outcome"] == "errors_confirmed"
    assert judge_sample(row, theirs)["outcome"] == "not_fetched"


def test_our_own_fetch_of_a_tagless_page_is_compared_not_refused():
    text = "Obituary notice for a long time resident of the valley, published today."
    row = page_row("https://site.example/a", f"<p>{text}</p>")
    ours = FetchedPage(200, f"<p>{text}</p>", via="own_ip")
    theirs = FetchedPage(200, f"<p>{text}</p>", via="scrapingdog")

    assert judge_sample(row, ours)["outcome"] == "matched"
    assert judge_sample(row, theirs)["outcome"] == "unverifiable"
