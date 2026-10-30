#!/bin/sh
# Builds v1-set.json and the v1 crawl list into ~ubuntu/crawl-1b/v1 from the registry (read-only) and the staged inputs; NEWS=B takes every news site from 2020, UNDATED=article takes undated news articles; RESEND_WAVES=N gives the sites of resend waves 1 to N resend_before 2026-10-27 (none by default); NODE2_MIN_WEIGHT=2 leaves the weight-1 sites outside the set off the list (the default; 1 keeps them), and NODE2_HOURS=H holds each site outside the set that stays to its stored pages divided by H an hour (0, no limit, by default); REJOIN=FILE puts the send_cap of the hosts in FILE one up, so the list walker queues their stored pages again; RECHECKED=FILE (recheck_hosts.py's TSV) lets the hosts it read as ok pass the homepage language, text and error checks; VERDICTS=FILE (the site check's TSV) puts its out hosts out for quality.
set -e
v=/home/ubuntu/crawl-1b/v1
f=/var/lib/task-api-feeder
news=${NEWS:-A}
undated=${UNDATED:-out}
resend_waves=${RESEND_WAVES:-0}
node2_min_weight=${NODE2_MIN_WEIGHT:-2}
node2_hours=${NODE2_HOURS:-0}
fail() { echo "build-v1.sh: $*" >&2; exit 1; }
case $resend_waves in '' | *[!0-9]*) fail "RESEND_WAVES=$resend_waves is not a number" ;; esac
rows=$v/benchmark-set-rows.json
[ "$news" = B ] && rows=$v/benchmark-set-rows-B.json
judged=/home/ubuntu/crawl-1b/judged-final-2026-10-27.tsv
waves=""
i=1
while [ $i -le $resend_waves ]; do
  waves="$waves $v/resend-waves-$news/resend-wave-$(printf %02d $i).txt"
  i=$((i + 1))
done
# Every input is checked before anything is written, so a bad one leaves v1-set.json and the list as they were.
for input in $f/backing.tsv /home/ubuntu/crawl-1b/overrides-new.tsv $f/excluded-domains.txt $judged $v/v1-hand.tsv $v/bulk-set-rows.json $rows $v/extra-set-rows.json \
  $v/news-domains.tsv.gz $v/news-outlets-2026-10-28.tsv $v/v1-store-counts-2026-10-28.tsv.gz $v/news-counts-2026-10-28b.tsv.gz $v/too-big.tsv $waves $REJOIN $RECHECKED $VERDICTS; do
  [ -s "$input" ] || fail "$input is missing or empty"
done
set -a; . /opt/desearch-bot/.env; set +a
rm -f $v/domains-v1.json $v/v1-set.json.next
common="--backing $f/backing.tsv --overrides /home/ubuntu/crawl-1b/overrides-new.tsv --excluded $f/excluded-domains.txt --judged $judged ${RECHECKED:+--rechecked $RECHECKED}"
python3 $v/crawl_list.py --out $v/domains-judged.json --decisions $v/decisions-judged.tsv.gz $common > $v/judged.out
python3 $v/v1_set.py --out $v/v1-set.json.next --news $news --undated $undated --decisions $v/decisions-judged.tsv.gz --hand $v/v1-hand.tsv \
  --rows $v/bulk-set-rows.json --rows $rows --rows $v/extra-set-rows.json --backing $f/backing.tsv --news-domains $v/news-domains.tsv.gz --outlets $v/news-outlets-2026-10-28.tsv \
  --judged $judged --counts $v/v1-store-counts-2026-10-28.tsv.gz --news-counts $v/news-counts-2026-10-28b.tsv.gz --too-big $v/too-big.tsv ${VERDICTS:+--site-verdicts $VERDICTS} > $v/v1-set.out
: > $v/resend-hosts.txt
[ -z "$waves" ] || cat $waves > $v/resend-hosts.txt
python3 $v/crawl_list.py --out $v/domains-v1.json --decisions $v/decisions-v1.tsv.gz --v1-set $v/v1-set.json.next $common --resend-before 2026-10-27 \
  ${waves:+--resend-hosts $v/resend-hosts.txt} --node2-min-weight $node2_min_weight --node2-hours $node2_hours ${REJOIN:+--rejoin-hosts $REJOIN} > $v/v1-list.out
mv $v/v1-set.json.next $v/v1-set.json
chown ubuntu:ubuntu $v/*
