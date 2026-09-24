from datetime import datetime, timedelta, timezone

from desearch_bot.states import State
from desearch_bot.timetable import Timetable

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def test_nothing_is_handed_out_before_its_time():
    table = Timetable()
    table.set("a.com", State.ACTIVE, NOW + timedelta(minutes=5), 1)
    assert table.take(10, NOW) == []
    assert table.take(10, NOW + timedelta(minutes=5)) == ["a.com"]


def test_known_sites_come_before_discovery_and_discovery_goes_by_rank():
    table = Timetable()
    table.set("late-rank.com", State.NEW, NOW, 900)
    table.set("top-rank.com", State.NEW, NOW, 3)
    table.set("known.com", State.ACTIVE, NOW, 500_000)
    table.set("unranked.com", State.NEW, NOW, None)
    assert table.take(10, NOW) == [
        "known.com",
        "top-rank.com",
        "late-rank.com",
        "unranked.com",
    ]


def test_a_taken_domain_is_gone_until_it_is_scheduled_again():
    table = Timetable()
    table.set("a.com", State.ACTIVE, NOW, 1)
    assert table.take(10, NOW) == ["a.com"] and table.take(10, NOW) == []
    table.set("a.com", State.ACTIVE, NOW, 1)
    assert table.take(10, NOW) == ["a.com"]


def test_rescheduling_replaces_the_earlier_time_and_none_removes_it():
    table = Timetable()
    table.set("a.com", State.ACTIVE, NOW, 1)
    table.set("a.com", State.ACTIVE, NOW + timedelta(hours=1), 1)
    table.set("b.com", State.NEW, NOW, 2)
    table.set("b.com", State.EXCLUDED, None, 2)
    assert table.take(10, NOW) == [] and len(table) == 1
    assert table.next_at() == int((NOW + timedelta(hours=1)).timestamp())


def test_the_limit_is_respected_across_both_kinds():
    table = Timetable()
    for i in range(5):
        table.set(f"known{i}.com", State.ACTIVE, NOW, i)
        table.set(f"new{i}.com", State.NEW, NOW, i)
    assert len(table.take(7, NOW)) == 7 and len(table.take(7, NOW)) == 3
