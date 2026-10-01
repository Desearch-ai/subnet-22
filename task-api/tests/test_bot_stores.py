import importlib.util

import pytest
from feeder import bot_stores
from feeder.bot_stores import RECORD, store_index, host_prefix, parse_listed_page

needs_rocksdict = pytest.mark.skipif(
    importlib.util.find_spec("rocksdict") is None,
    reason="reading the crawler's stores needs rocksdict",
)
# Byte for byte as the crawler stores them; refresh from a live store.
KEY = b"U10000tipbook.com\x0010000tipbook.com/aboutus"
RAW = bytes.fromhex("921000000000000087aa395ff625a36afe25a36a000000000000000001")


def test_the_stored_bytes_decode_to_the_crawlers_fields():
    assert RECORD.size == len(RAW)
    assert RECORD.unpack(RAW) == (4242, 1597614727, 1789076982, 1789076990, 0, 0, 1)


def test_a_host_always_lands_in_the_same_bucket():
    assert store_index("tennessean.com") == store_index("tennessean.com")
    assert 0 <= store_index("nypost.com") < 256
    assert store_index("10000tipbook.com") == 7


def test_a_key_and_record_become_the_address_a_miner_requests():
    page = parse_listed_page(KEY, RAW)

    assert page.url == "https://10000tipbook.com/aboutus"
    assert (page.lastmod, page.first_seen, page.crawled_at) == (
        1597614727,
        1789076982,
        0,
    )


def test_the_flags_say_the_scheme_and_whether_the_host_carries_www():
    plain = parse_listed_page(KEY, RECORD.pack(1, 0, 0, 0, 0, 0, 0))
    prefixed = parse_listed_page(KEY, RECORD.pack(1, 0, 0, 0, 0, 0, 1 | 2))

    assert plain.url == "http://10000tipbook.com/aboutus"
    assert prefixed.url == "https://www.10000tipbook.com/aboutus"


def test_a_hosts_range_is_bounded_by_its_own_prefix():
    assert host_prefix("a.example") == b"Ua.example\x00"


class Store:
    def __init__(self, rows: dict[bytes, bytes]):
        self.rows = dict(sorted(rows.items()))
        self.closed = False

    def items(self, from_key, read_opt=None):
        return [(k, v) for k, v in self.rows.items() if k >= from_key]

    def close(self):
        self.closed = True


def entry(host: str, path: str, lastmod: int, first_seen: int = 0, flags: int = 1):
    key = host_prefix(host) + f"{host}/{path}".encode()
    return key, RECORD.pack(1, lastmod, first_seen, 0, 0, 0, flags)


@needs_rocksdict
def test_only_the_newest_pages_of_a_domain_are_taken():
    store = Store(dict(entry("a.example", f"p{n}", lastmod=n) for n in range(10)))

    got = bot_stores.newest_pages(store, "a.example", 3)

    assert [page.url for page in got] == [
        "https://a.example/p9",
        "https://a.example/p8",
        "https://a.example/p7",
    ]


@needs_rocksdict
def test_a_domain_with_fewer_pages_than_asked_for_gives_them_all():
    store = Store(dict([entry("a.example", "one", 5)]))

    assert len(bot_stores.newest_pages(store, "a.example", 25)) == 1


@needs_rocksdict
def test_another_hosts_keys_are_left_alone():
    store = Store(
        {
            **dict([entry("a.example", "mine", 9)]),
            **dict([entry("b.example", "theirs", 9)]),
        }
    )

    got = bot_stores.newest_pages(store, "a.example", 25)

    assert [page.url for page in got] == ["https://a.example/mine"]


@needs_rocksdict
def test_a_bucket_is_opened_once_for_every_domain_it_holds(monkeypatch):
    hosts = ["a0.example", "b0.example", "c1.example"]
    opened = []

    def open_store(root, bucket, secondary):
        store = Store(dict(entry(host, "page", 3) for host in hosts))
        opened.append((bucket, store))
        return store

    monkeypatch.setattr(bot_stores, "open_store", open_store)
    monkeypatch.setattr(bot_stores, "store_index", lambda host: int(host[1]))

    rows = bot_stores.collect_pages("/buckets", "/tmp", hosts, 5)

    assert [bucket for bucket, _ in opened] == [0, 1]
    assert all(store.closed for _, store in opened)
    assert {row["host"] for row in rows} == set(hosts)
