from desearch_bot.cli import RESTART_FIRST, RESTART_MAX, restart_delay


def test_a_crashing_worker_waits_longer_each_time_up_to_a_limit():
    assert restart_delay(1) == RESTART_FIRST
    assert restart_delay(2) == 2 * RESTART_FIRST
    assert restart_delay(20) == RESTART_MAX
