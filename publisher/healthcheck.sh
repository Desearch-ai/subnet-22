#!/bin/sh
# Fail when the publisher heartbeat is missing or older than PUBLISHER_HEARTBEAT_MAX_AGE_S.
#
# debian:bookworm-slim already provides this script's tools: /bin/sh (dash), sed, date, and head
# (coreutils). find and stat are in that image too; the check reads unix_secs from the file
# instead of the inode mtime, so touching the file cannot hide a stall.
# The same comparison lives in heartbeat::is_stale: age > limit fails, age == limit is healthy.
#
# There is no HTTP listener. Coolify must not use an HTTP health check against this container.
set -u

path="${PUBLISHER_HEARTBEAT_PATH:-/data/heartbeat}"
max="${PUBLISHER_HEARTBEAT_MAX_AGE_S:-600}"

case "$max" in
  ''|*[!0-9]*)
    echo "PUBLISHER_HEARTBEAT_MAX_AGE_S is not a whole number of seconds: $max" >&2
    exit 1
    ;;
esac

if [ ! -f "$path" ]; then
  echo "heartbeat missing: $path" >&2
  exit 1
fi

ts=$(sed -n 's/^unix_secs=//p' "$path" | head -n 1)
case "$ts" in
  ''|*[!0-9]*)
    echo "heartbeat unreadable: $path" >&2
    exit 1
    ;;
esac

now=$(date +%s)
age=$((now - ts))
if [ "$age" -gt "$max" ]; then
  echo "heartbeat stale: $path is ${age}s old, limit ${max}s" >&2
  exit 1
fi

exit 0
