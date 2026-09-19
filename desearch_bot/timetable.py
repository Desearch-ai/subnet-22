"""When each domain a process owns is next due, kept in memory."""

from __future__ import annotations

import heapq
from datetime import datetime

from .states import State

REFRESH = frozenset({State.ACTIVE, State.FAILING, State.DOWN})
UNRANKED = 2**31 - 1


class Timetable:
    """Domains in the order they fall due: known sites first, then discovery by rank."""

    def __init__(self):
        self._heaps: tuple[list, list] = ([], [])
        self._due: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._due)

    def set(
        self, host: str, state: State, due: datetime | None, rank: int | None
    ) -> None:
        """Schedule a domain's next visit, replacing any earlier time; None takes it off."""
        if due is None:
            self._due.pop(host, None)
            return
        when = int(due.timestamp())
        self._due[host] = when
        heap = self._heaps[0 if state in REFRESH else 1]
        heapq.heappush(heap, (when, UNRANKED if rank is None else rank, host))

    def take(self, limit: int, now: datetime) -> list[str]:
        """Up to limit domains whose time has come; each leaves the timetable until set again."""
        cutoff, taken = int(now.timestamp()), []
        for heap in self._heaps:
            while heap and len(taken) < limit and heap[0][0] <= cutoff:
                when, _, host = heapq.heappop(heap)
                if self._due.get(host) == when:
                    del self._due[host]
                    taken.append(host)
        return taken

    def next_at(self) -> int | None:
        """The earliest scheduled time, as a Unix timestamp."""
        times = [heap[0][0] for heap in self._heaps if heap]
        return min(times) if times else None
