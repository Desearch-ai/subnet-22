#!/bin/bash
# Forced command for the planner box's refresh-pull key on the crawler box: list, get <run>, done <run> (receipt on stdin).
set -euo pipefail
home=${REFRESH_HOME:-$HOME/refresh}
read -r verb name _ <<<"${SSH_ORIGINAL_COMMAND:-} x"
ready() { [[ $name =~ ^refresh-[a-z0-9-]{1,80}$ ]] && [ -e "$home/outbox/$name.ready" ] && [ -d "$home/outbox/$name" ]; }
case $verb in
  list)
    for f in "$home"/outbox/*.ready; do [ -e "$f" ] && basename "$f" .ready; done
    exit 0 ;;
  get)
    ready || { echo "no ready run $name" >&2; exit 2; }
    tar -C "$home/outbox" -cf - "$name" ;;
  done)
    ready || { echo "no ready run $name" >&2; exit 2; }
    mkdir -p "$home/receipts" "$home/sent"
    head -c 65536 > "$home/receipts/$name.txt.part"
    mv "$home/outbox/$name" "$home/sent/$name"
    rm -f "$home/outbox/$name.ready"
    mv "$home/receipts/$name.txt.part" "$home/receipts/$name.txt" ;;
  *)
    echo "list, get <run> or done <run>" >&2
    exit 2 ;;
esac
