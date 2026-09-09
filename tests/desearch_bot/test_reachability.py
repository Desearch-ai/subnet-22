import pytest

from desearch_bot.reachability import Canonicaliser
from desearch_bot.suffixes import PublicSuffixList


@pytest.fixture
def psl(tmp_path):
    rules = tmp_path / "psl.dat"
    rules.write_text("com\nnet\nuk\nco.uk\n", encoding="utf-8")
    return PublicSuffixList(rules)


class _Response:
    def __init__(self, status, location=None):
        self.status = status
        self.headers = {"Location": location} if location else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def release(self):
        return None


class _Session:
    """Answers each requested URL from a scripted map of redirects."""

    def __init__(self, redirects=None, fail=()):
        self.redirects = redirects or {}
        self.fail = set(fail)
        self.requested = []

    def get(self, url, **_):
        self.requested.append(url)
        if url in self.fail:
            raise OSError("refused")
        location = self.redirects.get(url)
        return _Response(301, location) if location else _Response(200)


async def test_a_domain_serving_itself_has_no_canonical_host(psl):
    session = _Session()
    host, canonical, error = await Canonicaliser(session, psl).final_host("example.com")
    assert (canonical, error) == (None, None)


async def test_redirect_to_another_domain_is_recorded(psl):
    session = _Session({"https://old.com/": "https://new.com/"})
    _, canonical, error = await Canonicaliser(session, psl).final_host("old.com")
    assert canonical == "new.com"
    assert error is None


async def test_www_is_not_a_redirect(psl):
    """example.com -> www.example.com is the same registrable domain, not an alias."""
    session = _Session({"https://example.com/": "https://www.example.com/"})
    _, canonical, _ = await Canonicaliser(session, psl).final_host("example.com")
    assert canonical is None


async def test_relative_location_is_resolved_against_the_current_url(psl):
    session = _Session({"https://example.com/": "/en/"})
    _, canonical, _ = await Canonicaliser(session, psl).final_host("example.com")
    assert canonical is None
    assert session.requested[-1] == "https://example.com/en/"


async def test_a_chain_ending_on_another_domain_follows_through(psl):
    session = _Session({
        "https://a.com/": "https://b.com/",
        "https://b.com/": "https://c.com/landing",
    })
    _, canonical, _ = await Canonicaliser(session, psl).final_host("a.com")
    assert canonical == "c.com"


async def test_https_failure_falls_back_to_http(psl):
    session = _Session({"http://example.com/": "https://other.net/"},
                       fail=["https://example.com/"])
    _, canonical, _ = await Canonicaliser(session, psl).final_host("example.com")
    assert canonical == "other.net"


async def test_both_schemes_failing_reports_the_error(psl):
    session = _Session(fail=["https://example.com/", "http://example.com/"])
    _, canonical, error = await Canonicaliser(session, psl).final_host("example.com")
    assert canonical is None
    assert error == "OSError"


async def test_a_redirect_loop_is_not_followed_forever(psl):
    loop = {
        "https://a.com/": "https://b.com/", "https://b.com/": "https://a.com/",
        "http://a.com/": "http://b.com/", "http://b.com/": "http://a.com/",
    }
    _, canonical, error = await Canonicaliser(_Session(loop), psl).final_host("a.com")
    assert (canonical, error) == (None, "RuntimeError")


async def test_an_https_loop_still_falls_back_to_http(psl):
    """Only https loops, so the http answer is the one that counts."""
    session = _Session({"https://a.com/": "https://b.com/", "https://b.com/": "https://a.com/"})
    _, canonical, error = await Canonicaliser(session, psl).final_host("a.com")
    assert (canonical, error) == (None, None)
