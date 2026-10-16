"""How much work the system can take, from how fast the publisher has been finishing it."""

from __future__ import annotations

from collections import deque

# Work in the system the bot may fill to, in seconds of publishing at the measured rate; well above the ~10 min a task takes end to end.
LAG_TARGET_S = 1800
# Claims stop once this much is waiting to be published.
LAG_LIMIT_S = 1800
# Assumed until the publisher has been measured, so a fresh start is not held at zero.
RATE_FLOOR = 100 / 60
WINDOW_S = 300


class PublishRate:
    """Tasks a second the publisher finished over the last few minutes."""

    def __init__(
        self, lag_s: float = LAG_TARGET_S, limit_s: float = LAG_LIMIT_S
    ) -> None:
        self.lag_s, self.limit_s = lag_s, limit_s
        self.samples: deque[tuple[float, int]] = deque()

    def note(self, acked: int, at: float) -> None:
        self.samples.append((at, acked))
        while len(self.samples) > 2 and self.samples[1][0] <= at - WINDOW_S:
            self.samples.popleft()

    def per_second(self) -> float:
        if len(self.samples) < 2:
            return RATE_FLOOR
        (first_at, first), (last_at, last) = self.samples[0], self.samples[-1]
        if last_at <= first_at:
            return RATE_FLOOR
        return max(RATE_FLOOR, (last - first) / (last_at - first_at))

    def room(self, in_system: int) -> int:
        """Tasks that may still enter before the work ahead of the publisher passes the target."""
        return max(0, int(self.per_second() * self.lag_s) - in_system)

    def overloaded(self, waiting: int) -> bool:
        return waiting > self.per_second() * self.limit_s
