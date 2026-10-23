"""Dates from meta tags and JSON-LD, written as ISO 8601 in UTC."""

import re
from datetime import datetime, timedelta, timezone

DATE_TIME = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?)?",
    re.I,
)


def read_date(value: object) -> str:
    """RFC 3339 in UTC when the value has a time and an offset, the date alone when it has no offset."""
    found = DATE_TIME.search(value[:64]) if isinstance(value, str) else None
    if not found:
        return ""
    year, month, day = (int(part) for part in found.groups()[:3])
    if not 1900 <= year <= 2100:
        return ""
    try:
        date = datetime(year, month, day)
    except ValueError:
        return ""
    hour, minute, second, offset = found.groups()[3:]
    if hour is None or offset is None:
        return date.strftime("%Y-%m-%d")
    try:
        moment = datetime(
            year,
            month,
            day,
            int(hour),
            int(minute),
            int(second or 0),
            tzinfo=read_offset(offset),
        )
    except ValueError:
        return date.strftime("%Y-%m-%d")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_offset(offset: str) -> timezone:
    if offset.upper() == "Z":
        return timezone.utc
    digits = offset[1:].replace(":", "")
    hours, minutes = int(digits[:2]), int(digits[2:])
    if hours > 14 or minutes > 59:
        raise ValueError(offset)
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(-delta if offset[0] == "-" else delta)
