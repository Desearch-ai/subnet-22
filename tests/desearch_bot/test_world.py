import random
from datetime import datetime, timedelta, timezone

import pytest

from desearch_bot.loop import Loop
from desearch_bot.urls import parse
from desearch_bot.visit import Visitor

from .fakeweb import ALLOW, ENGLISH, PAGES, FakeWeb, index, urlset
from .harness import MemoryRegistry, open_buckets, seed

START = datetime(2026, 9, 7, tzinfo=timezone.utc)
STEP = timedelta(minutes=5)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


class World(FakeWeb):
    """Sites that change as simulated time passes, and when each address was asked for."""

    def __init__(self):
        super().__init__()
        self.now = START
        self.log = []
        self.changes = []

    def get(self, url, headers=None, **kwargs):
        self.log.append((self.now, url))
        return super().get(url, headers=headers, **kwargs)

    def update(self):
        for change in self.changes:
            change(self)

    def hits(self, fragment, since=START, until=None):
        return [
            at
            for at, url in self.log
            if fragment in url and since <= at and (until is None or at < until)
        ]


def plain_site(world, host):
    world.page(
        f"https://{host}/robots.txt",
        f"User-agent: *\nSitemap: https://{host}/sitemap.xml\n".encode(),
    )
    world.page(f"https://{host}/sitemap.xml", urlset(*PAGES, host=host), etag='"v1"')
    world.page(f"https://{host}/", ENGLISH)


def hourly(host, path, **markup):
    """A new page every hour on the hour; the sitemap lists the latest twenty."""
    world_path = f"https://{host}{path}"

    def change(world):
        latest = int((world.now - START) / HOUR)
        pages = [f"/a/{i}" for i in range(latest - 19, latest + 1)]
        world.page(world_path, urlset(*pages, host=host, **markup), etag=f'"{latest}"')

    return change


def outage(host, start, end):
    def change(world):
        if start <= world.now < end:
            world.dead.add(host)
        else:
            world.dead.discard(host)

    return change


def crawler(root, world, hosts, later=()):
    buckets = open_buckets(root, [*hosts, *later])
    seed(buckets, hosts, START)
    visitor = Visitor(
        world,
        buckets,
        detect_language=lambda text: "en" if "english" in text else "fr",
        registrable=lambda host: host.removeprefix("www."),
        floor=0.0,
    )
    crawl = Loop(
        buckets, visitor, 10, MemoryRegistry(), frozenset(), clock=lambda: world.now
    )
    crawl.load()
    return crawl


@pytest.fixture
def make_crawl(tmp_path):
    made = []

    def make(world, hosts, later=()):
        made.append(crawler(tmp_path / f"crawl{len(made)}", world, hosts, later))
        return made[-1]

    yield make
    for crawl in made:
        crawl.buckets.close()


async def advance(crawl, world, until, step=STEP):
    """Move simulated time forward, visiting whatever falls due at each step."""
    while world.now < until:
        world.update()
        for _ in range(50):
            hosts = crawl.timetable.take(100, world.now)
            if not hosts:
                break
            crawl.save(
                [await crawl.once(known) for known in map(crawl.known, hosts) if known]
            )
            await crawl.sync()
        else:
            raise AssertionError(f"domains kept falling due at {world.now}")
        world.now += step


def state(crawl, host):
    return crawl.buckets.store(host).domain(host)["state"]


def stored(crawl, url, host):
    return crawl.buckets.store(host).url(parse(url, host))


async def test_a_week_of_news_quiet_sites_an_outage_and_a_redirect(make_crawl):
    random.seed(7)
    world = World()
    for host in ("quiet.com", "flaky.com", "landing.com"):
        plain_site(world, host)
    world.page(
        "https://news.com/robots.txt",
        b"User-agent: *\nSitemap: https://news.com/news.xml\n",
    )
    world.page("https://news.com/", ENGLISH)
    world.page(
        "https://moved.com/robots.txt",
        status=301,
        location="https://landing.com/robots.txt",
    )
    down, up = START + 2 * DAY, START + 3 * DAY
    world.changes = [
        hourly("news.com", "/news.xml", news=True),
        hourly("flaky.com", "/sitemap.xml", changefreq="hourly"),
        outage("flaky.com", down, up),
    ]
    crawl = make_crawl(
        world,
        ["news.com", "quiet.com", "flaky.com", "moved.com"],
        later=["landing.com"],
    )
    await advance(crawl, world, down + 2 * HOUR)
    assert state(crawl, "flaky.com") == "failing"
    await advance(crawl, world, up + timedelta(hours=4, minutes=30))
    assert state(crawl, "flaky.com") == "active"
    await advance(crawl, world, START + 7 * DAY)

    for i in range(7 * 24):
        record = stored(crawl, f"https://news.com/a/{i}", "news.com")
        published = (START + i * HOUR).timestamp()
        assert record is not None and record.first_seen - published <= 65 * 60, i
    assert 7 * 24 <= len(world.hits("news.com/news.xml")) <= 7 * 24 * 6

    assert len(world.hits("quiet.com/sitemap.xml")) <= 5
    assert len(world.hits("quiet.com/robots.txt")) <= 9

    assert 5 <= len(world.hits("flaky.com", down, up)) <= 12
    assert stored(crawl, f"https://flaky.com/a/{26 * 3 - 5}", "flaky.com")

    assert world.hits("moved.com") == [START]
    assert state(crawl, "moved.com") == "redirects"
    assert state(crawl, "landing.com") == "active"
    assert all(
        record["state"] != "down"
        for store in crawl.buckets.stores.values()
        for _, record in store.domains()
    )


async def test_a_sitemap_added_later_is_found_at_the_monthly_recheck(make_crawl):
    random.seed(7)
    world = World()
    world.page("https://late.com/robots.txt", ALLOW)
    world.page("https://late.com/", ENGLISH)

    def launch(world):
        if world.now >= START + 10 * DAY:
            world.page("https://late.com/sitemap.xml", urlset(*PAGES, host="late.com"))

    world.changes = [launch]
    crawl = make_crawl(world, ["late.com"])
    await advance(crawl, world, START + 26 * DAY, HOUR)
    assert state(crawl, "late.com") == "no_sitemap"
    assert len(world.hits("late.com")) == 3
    await advance(crawl, world, START + 34 * DAY, HOUR)
    assert state(crawl, "late.com") == "active"


async def test_an_index_bigger_than_one_visit_is_read_to_the_end(make_crawl):
    random.seed(7)
    world = World()
    host = "big.com"
    children = [f"https://{host}/s{i}.xml" for i in range(100)]
    world.page(
        f"https://{host}/robots.txt",
        f"User-agent: *\nSitemap: https://{host}/index.xml\n".encode(),
    )
    world.page(
        f"https://{host}/index.xml", index(*[(child, None) for child in children])
    )
    for i, child in enumerate(children):
        world.page(child, urlset(*[f"/p{i}-{j}" for j in range(12)], host=host))
    world.page(f"https://{host}/", ENGLISH)
    crawl = make_crawl(world, [host])
    await advance(crawl, world, START + HOUR)
    assert all(stored(crawl, f"https://{host}/p{i}-0", host) for i in range(100))
    assert len(world.hits(".xml")) == len(children) + 1
