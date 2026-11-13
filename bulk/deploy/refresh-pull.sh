#!/bin/bash
# On the planner box: publishes the runs the crawler box's refresh runner left in its outbox, and sends back each publish's output.
set -euo pipefail
. /etc/refresh-pull.env
dir=/root/bulk-runs
remote() { ssh -T -i /root/.ssh/refresh-pull -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new "$CRAWLER_HOST" "$@"; }
exec 9>"$dir/.refresh-pull.lock"
flock -n 9 || exit 0
for name in $(remote list); do
  [[ $name =~ ^refresh-[a-z0-9-]{1,80}$ ]] || continue
  if [ ! -d "$dir/$name" ]; then
    rm -rf "$dir/$name.part" && mkdir "$dir/$name.part"
    remote get "$name" | tar -C "$dir/$name.part" --no-same-owner -xf -
    mv "$dir/$name.part/$name" "$dir/$name"
    rmdir "$dir/$name.part"
    chmod 777 "$dir/$name"
  fi
  while docker ps -a --filter name=^/bulk-pub$ -q | grep -q .; do sleep 20; done
  out=$("$dir/publish-one.sh" "$name" 2>&1 || true)
  printf "%s\n" "$out" | remote done "$name"
  echo "$name: $(printf "%s\n" "$out" | grep -m1 '^rc=')"
done
