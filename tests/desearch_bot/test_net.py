import pytest

from desearch_bot.net import public_host, public_only


def test_private_answers_are_dropped_and_public_ones_kept():
    answers = [{"host": "10.0.0.1"}, {"host": "93.184.216.34"}]
    assert public_only("example.com", answers) == [{"host": "93.184.216.34"}]


def test_a_name_that_only_resolves_privately_is_refused():
    with pytest.raises(OSError):
        public_only(
            "intranet.example.com", [{"host": "192.168.1.10"}, {"host": "127.0.0.1"}]
        )


def test_literal_private_addresses_are_not_public():
    for host in (
        "10.1.2.3",
        "127.0.0.1",
        "169.254.169.254",
        "192.168.0.1",
        "::1",
        "100.64.0.1",
    ):
        assert not public_host(host)


def test_names_and_public_addresses_pass():
    for host in ("example.com", "8.8.8.8", "2606:4700:4700::1111"):
        assert public_host(host)
