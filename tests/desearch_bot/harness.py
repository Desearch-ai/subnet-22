"""A crawl loop's surroundings for tests: temporary bucket stores and an in-memory registry."""

from desearch_bot import records
from desearch_bot.buckets import Buckets, Changes, Resources, bucket_of
from desearch_bot.registry import Change
from desearch_bot.states import Outcome, State

RESOURCES = Resources(cache_bytes=8 << 20, memtable_bytes=64 << 20)


class MemoryRegistry:
    """Keeps what visits reported, and hands redirect destinations back on the next sync."""

    def __init__(self):
        self.reported = []
        self.waiting = []

    async def report(self, visits, now):
        self.reported.extend(visits)
        for write, _ in visits:
            if write.visit.outcome is Outcome.REDIRECT and write.canonical_host:
                self.waiting.append(
                    Change(
                        write.canonical_host,
                        None,
                        "big_generic",
                        State.NEW.value,
                        None,
                        None,
                    )
                )

    async def changes(self):
        waiting, self.waiting = self.waiting, []
        return waiting


def open_buckets(root, hosts):
    """Stores for every bucket the given domains fall into."""
    return Buckets(root, sorted({bucket_of(host) for host in hosts}), RESOURCES)


def seed(buckets, hosts, due, state=State.NEW):
    """Put domains in their stores as if the registry had just delivered them."""
    for rank, host in enumerate(hosts, 1):
        changes = Changes()
        changes.domain(
            host, records.new_domain(rank, "big_generic", state=state, due=due)
        )
        buckets.store(host).write(changes)
