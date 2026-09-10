from collections import Counter

import pytest

from desearch_bot.buckets import (
    BUCKETS,
    Buckets,
    BucketStore,
    Changes,
    Resources,
    bucket_of,
    owned_buckets,
    sitemap_id,
)
from desearch_bot.urls import TIMED, Listing, parse


@pytest.fixture(scope="module")
def resources():
    return Resources(cache_bytes=8 << 20, memtable_bytes=64 << 20)


@pytest.fixture
def store(tmp_path, resources):
    with BucketStore(tmp_path / "b", resources) as opened:
        yield opened


def _entries(*paths, domain="example.com", lastmod=0, timed=False):
    return [
        (parse(f"https://{domain}{path}", domain), lastmod, timed) for path in paths
    ]


def test_a_domain_always_lands_in_the_same_bucket():
    assert bucket_of("example.com") == bucket_of("example.com")
    assert 0 <= bucket_of("example.com") < BUCKETS


def test_domains_spread_evenly_over_the_buckets():
    counts = Counter(bucket_of(f"site{i}.com") for i in range(25_600))
    assert len(counts) == BUCKETS and max(counts.values()) < 2 * min(counts.values())


def test_a_sitemap_id_is_stable_and_fits_a_signed_64_bit_field():
    url = "https://example.com/sitemap.xml"
    assert sitemap_id(url) == sitemap_id(url) != sitemap_id(url + "?page=2")
    assert 0 <= sitemap_id(url) < 2**63


def test_domain_and_sitemap_records_come_back_as_written(store):
    changes = Changes()
    changes.domain("example.com", {"state": "active", "failures": 0})
    changes.sitemap("example.com", "https://example.com/a.xml", {"kind": "urlset"})
    changes.sitemap("example.com.au", "https://example.com.au/a.xml", {"kind": "index"})
    store.write(changes)
    assert store.domain("example.com") == {"state": "active", "failures": 0}
    assert store.domain("missing.com") is None
    assert list(store.sitemaps("example.com")) == [
        ("https://example.com/a.xml", {"kind": "urlset"})
    ]


def test_domains_lists_domain_records_and_nothing_else(store):
    changes = Changes()
    for host in ("a.com", "b.com"):
        changes.domain(host, {"state": "new"})
        changes.sitemap(host, f"https://{host}/s.xml", {})
    store.write(changes)
    store.record_listing(1, _entries("/x", domain="a.com"), now=1)
    assert [host for host, _ in store.domains()] == ["a.com", "b.com"]


def test_a_first_listing_is_all_new_and_a_repeat_adds_nothing(store):
    assert store.record_listing(1, _entries("/a", "/b"), now=1000) == Listing(2, 2, 0)
    assert store.record_listing(1, _entries("/a", "/b"), now=2000) == Listing(2, 0, 0)


def test_a_moved_lastmod_is_counted_and_kept(store):
    store.record_listing(1, _entries("/a", lastmod=100), now=1000)
    listing = store.record_listing(2, _entries("/a", lastmod=200, timed=True), now=2000)
    record = store.url(parse("https://example.com/a", "example.com"))
    assert listing.moved == 1 and record.lastmod == 200 and record.flags & TIMED
    assert (record.sitemap_id, record.first_seen, record.last_seen) == (2, 1000, 2000)


def test_a_domains_urls_come_back_together_and_no_others(store):
    store.record_listing(1, _entries("/a", "/b"), now=1)
    store.record_listing(2, _entries("/z", domain="example.com.au"), now=1)
    found = [url.fetchable() for url, _ in store.urls("example.com")]
    assert found == ["https://example.com/a", "https://example.com/b"]


def test_everything_survives_closing_and_reopening(tmp_path, resources):
    with BucketStore(tmp_path / "b", resources) as opened:
        opened.record_listing(1, _entries("/a"), now=1000)
        changes = Changes()
        changes.domain("example.com", {"state": "active"})
        opened.write(changes)
    with BucketStore(tmp_path / "b", resources) as reopened:
        assert (
            reopened.url(parse("https://example.com/a", "example.com")).first_seen
            == 1000
        )
        assert reopened.domain("example.com") == {"state": "active"}


def test_a_process_opens_only_its_buckets_and_routes_each_domain(tmp_path, resources):
    mine = bucket_of("example.com")
    with Buckets(tmp_path, [mine], resources) as buckets:
        assert buckets.owns("example.com")
        assert buckets.store("example.com") is buckets.stores[mine]
        other = next(
            f"site{i}.com" for i in range(1000) if bucket_of(f"site{i}.com") != mine
        )
        assert not buckets.owns(other)
        with pytest.raises(KeyError):
            buckets.store(other)


def test_every_bucket_has_exactly_one_owner_for_any_number_of_workers():
    for workers in (1, 6, 7, 64):
        owned = [b for worker in range(workers) for b in owned_buckets(worker, workers)]
        assert sorted(owned) == list(range(BUCKETS))
