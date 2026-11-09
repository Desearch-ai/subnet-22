import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "desearch-bot" / "tools" / "recheck_hosts.py"
)
SPEC = importlib.util.spec_from_file_location("recheck_hosts", SOURCE)
recheck_hosts = importlib.util.module_from_spec(SPEC)
sys.modules["recheck_hosts"] = recheck_hosts
SPEC.loader.exec_module(recheck_hosts)

Answer = recheck_hosts.Answer
ENGLISH = (
    "<html lang='en'><body>"
    + "Reporting on the council vote and the budget. " * 20
    + "</body></html>"
)


def site(pages):
    def fetch(url, limit):
        found = pages.get(url)
        if found is None:
            return Answer(404, url, b"")
        if isinstance(found, Exception):
            raise found
        status, final, body = found
        return Answer(status, final or url, body.encode())

    return fetch


def english(text):
    return "en" if "council" in text else "fr"


def check(pages, own_host=False, host="paper.example"):
    return recheck_hosts.check(host, site(pages), english, own_host)


def test_robots_follow_our_group_else_the_star_group():
    paths, delay = recheck_hosts.rules(
        "User-agent: *\nDisallow: /\n\nUser-agent: DesearchBot\nAllow: /\nCrawl-delay: 5\n"
    )
    assert recheck_hosts.allows(paths, "/") and delay == 5
    paths, _ = recheck_hosts.rules(
        "User-agent: *\nDisallow: /private\nAllow: /private/open$\n"
    )
    assert recheck_hosts.allows(paths, "/") and not recheck_hosts.allows(
        paths, "/private/x"
    )
    assert recheck_hosts.allows(paths, "/private/open")
    assert not recheck_hosts.allows(
        recheck_hosts.rules("User-agent: desearchbot\nDisallow: /\n")[0], "/"
    )


def test_an_english_site_that_lets_us_in_is_ok_with_its_listing():
    found = check(
        {
            "https://paper.example/robots.txt": (200, None, "Sitemap: /news.xml\n"),
            "https://paper.example/": (200, "https://www.paper.example/", ENGLISH),
            "https://paper.example/news.xml": (
                200,
                None,
                "<urlset><url><loc>a</loc></url></urlset>",
            ),
        }
    )
    assert (found.verdict, found.language, found.declared) == ("ok", "en", "en")
    assert found.listing == "sitemap:1 https://paper.example/news.xml"


def test_walls_robots_and_challenges_keep_a_site_out():
    robots = {
        "https://paper.example/robots.txt": (200, None, "User-agent: *\nDisallow: /\n")
    }
    assert (check(robots).verdict, check(robots).why) == ("walled", "robots_disallow")
    for status in (401, 402, 403, 429):
        found = check({"https://paper.example/": (status, None, "")})
        assert (found.verdict, found.why) == ("walled", f"homepage_{status}")
    challenge = "<html><title>Just a moment...</title>" + "x " * 300
    assert check({"https://paper.example/": (503, None, challenge)}).why == "challenge"
    assert (
        check({"https://paper.example/": (200, None, "Access denied " + ENGLISH)}).why
        == "challenge"
    )
    assert (
        check({"https://paper.example/robots.txt": (429, None, "")}).verdict == "walled"
    )


def test_a_site_that_moved_is_named_by_the_bots_rule():
    moved = {"https://paper.example/": (200, "https://paper.com/", ENGLISH)}
    assert (check(moved).verdict, check(moved).why) == ("moved", "paper.com")
    edition = {
        "https://paper.example/": (200, "https://edition.paper.example/", ENGLISH)
    }
    assert check(edition).verdict == "ok"
    docs = {
        "https://docs.paper.example/": (200, "https://docs.paper.example/en/", ENGLISH)
    }
    assert check(docs, own_host=True, host="docs.paper.example").verdict == "ok"
    assert check(edition, own_host=True).verdict == "moved", (
        "a subdomain host stays itself only on its own host"
    )
    parent = {
        "https://docs.paper.example/": (200, "https://paper.example/docs/", ENGLISH)
    }
    assert (
        check(parent, own_host=True, host="docs.paper.example").why == "paper.example"
    )


def test_a_page_declaring_another_language_is_not_english_whatever_the_model_says():
    spanish_template = ENGLISH.replace("lang='en'", "lang='es'")
    assert (
        check({"https://paper.example/": (200, None, spanish_template)}).verdict
        == "not_english"
    )


def test_language_text_and_errors_are_read_as_the_bot_reads_them():
    french = "<html lang='en'>" + "Le conseil municipal a voté le budget. " * 20
    assert (
        check({"https://paper.example/": (200, None, french)}).verdict
    ) == "not_english"
    assert (
        check({"https://paper.example/": (200, None, "<p>Loading</p>")}).verdict
        == "no_text"
    )
    assert (
        check({"https://paper.example/": (404, None, ENGLISH)}).verdict
        == "homepage_error"
    )
    assert check({"https://paper.example/": (502, None, "")}).verdict == "unreachable"
    down = {
        "https://paper.example/robots.txt": OSError("reset"),
        "http://paper.example/robots.txt": OSError("reset"),
    }
    assert (check(down).verdict, check(down).why) == ("unreachable", "robots:OSError")


def test_private_addresses_are_never_fetched():
    with pytest.raises(recheck_hosts.PrivateAddress):
        recheck_hosts.public_connection(("localhost", 80), 1)
    pacer = recheck_hosts.Pacer(0)
    answer, error = recheck_hosts.first_answer(
        "localhost",
        "/robots.txt",
        100,
        lambda url, limit: recheck_hosts.http_fetch(url, limit, pacer),
    )
    assert (answer, error) == (None, "PrivateAddress")


def test_odd_answers_do_not_stop_the_run():
    found = check(
        {
            "https://paper.example/robots.txt": (
                200,
                None,
                "Sitemap: http://[oops/sitemap.xml\n",
            ),
            "https://paper.example/": (
                200,
                None,
                "<meta charset='undefined'>" + ENGLISH,
            ),
        }
    )
    assert found.verdict == "ok"

    def broken(text):
        raise RuntimeError("model")

    found = recheck_hosts.checked(
        "paper.example",
        site({"https://paper.example/": (200, None, ENGLISH)}),
        broken,
        False,
        None,
    )
    assert (found.verdict, found.why) == ("error", "RuntimeError")


def test_the_pace_is_shared_with_the_www_name_and_slows_to_crawl_delay(monkeypatch):
    pacer = recheck_hosts.Pacer(1.0)
    monkeypatch.setattr(recheck_hosts.time, "sleep", lambda seconds: None)
    pacer.wait("www.paper.example")
    first = pacer.last["paper.example"]
    pacer.slow_to("www.paper.example", 600)
    pacer.wait("paper.example")
    assert pacer.last["paper.example"] - first == recheck_hosts.MAX_CRAWL_DELAY, (
        "the next request waits the delay from the last one"
    )


def serve(answer):
    """A local server that answers one connection with answer(conn), and its port."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _ = server.accept()
        with conn:
            try:
                answer(conn)
            except OSError:
                pass
        server.close()

    threading.Thread(target=run, daemon=True).start()
    return server.getsockname()[1]


def drip(prefix):
    def answer(conn):
        conn.recv(4096)
        conn.sendall(prefix)
        for _ in range(40):
            conn.sendall(b"x")
            time.sleep(0.1)

    return answer


def test_a_slow_answer_is_cut_off_and_only_http_is_followed(monkeypatch):
    monkeypatch.setattr(recheck_hosts, "public_address", lambda address: True)
    monkeypatch.setattr(recheck_hosts, "MAX_SECONDS", 0.5)
    pacer = recheck_hosts.Pacer(0)
    for prefix in (
        b"HTTP/1.1 200 OK\r\nX-Slow: ",
        b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\n\r\n",
    ):
        port = serve(drip(prefix))
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            recheck_hosts.http_fetch(f"http://127.0.0.1:{port}/", 1000, pacer)
        assert time.monotonic() - started < 2.5

    def to_ftp(conn):
        conn.recv(4096)
        conn.sendall(
            b"HTTP/1.1 302 Found\r\nLocation: ftp://127.0.0.1/\r\nContent-Length: 0\r\n\r\n"
        )

    port = serve(to_ftp)
    with pytest.raises(OSError, match="ftp redirect"):
        recheck_hosts.http_fetch(f"http://127.0.0.1:{port}/", 1000, pacer)
