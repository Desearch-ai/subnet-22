#!/bin/sh
# Builds the v1 set and crawl list on the bot box; README.md lists its variables.
set -e
v=/home/ubuntu/crawl-1b/v1
f=/var/lib/task-api-feeder
news=${NEWS:-A}
undated=${UNDATED:-out}
resend_waves=${RESEND_WAVES:-0}
node2_min_weight=${NODE2_MIN_WEIGHT:-2}
node2_hours=${NODE2_HOURS:-0}
landing_rank=${LANDING_RANK:-}
deep_rank=${DEEP_RANK:-}
whole_rank=${WHOLE_RANK:-}
fail() { echo "build-v1.sh: $*" >&2; exit 1; }
case $resend_waves in '' | *[!0-9]*) fail "RESEND_WAVES=$resend_waves is not a number" ;; esac
case $landing_rank in *[!0-9]*) fail "LANDING_RANK=$landing_rank is not a number" ;; esac
case $deep_rank$whole_rank in *[!0-9]*) fail "DEEP_RANK=$deep_rank or WHOLE_RANK=$whole_rank is not a number" ;; esac
case ${HUBS:-0}${SUBDOMAINS:-0} in 0 | 00 | 01 | 10 | 11) ;; *) fail "HUBS and SUBDOMAINS are 0 or 1" ;; esac
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
  $v/news-domains.tsv.gz $v/news-outlets-2026-10-28.tsv $v/v1-store-counts-2026-10-28.tsv.gz $v/news-counts-2026-10-28b.tsv.gz $v/too-big.tsv $waves $REJOIN $RECHECKED $VERDICTS \
  ${PSL:-/opt/desearch-bot/data/public_suffix_list.dat}; do
  [ -s "$input" ] || fail "$input is missing or empty"
done
set -a; . /opt/desearch-bot/.env; set +a
rm -f $v/domains-v1.json $v/v1-set.json.next
common="--backing $f/backing.tsv --overrides /home/ubuntu/crawl-1b/overrides-new.tsv --excluded $f/excluded-domains.txt --judged $judged ${RECHECKED:+--rechecked $RECHECKED}"
python3 $v/crawl_list.py --out $v/domains-judged.json --decisions $v/decisions-judged.tsv.gz $common > $v/judged.out
python3 $v/v1_set.py --out $v/v1-set.json.next --news $news --undated $undated --decisions $v/decisions-judged.tsv.gz --hand $v/v1-hand.tsv \
  --rows $v/bulk-set-rows.json --rows $rows --rows $v/extra-set-rows.json --backing $f/backing.tsv --news-domains $v/news-domains.tsv.gz --outlets $v/news-outlets-2026-10-28.tsv \
  --judged $judged --counts $v/v1-store-counts-2026-10-28.tsv.gz --news-counts $v/news-counts-2026-10-28b.tsv.gz --too-big $v/too-big.tsv ${VERDICTS:+--site-verdicts $VERDICTS} ${landing_rank:+--landing-rank $landing_rank} \
  ${deep_rank:+--deep-rank $deep_rank} ${whole_rank:+--whole-rank $whole_rank} $([ "${HUBS:-0}" = 1 ] && echo --hubs) $([ "${SUBDOMAINS:-0}" = 1 ] && echo --subdomains) \
  --public-suffixes ${PSL:-/opt/desearch-bot/data/public_suffix_list.dat} > $v/v1-set.out
: > $v/resend-hosts.txt
[ -z "$waves" ] || cat $waves > $v/resend-hosts.txt
python3 $v/crawl_list.py --out $v/domains-v1.json --decisions $v/decisions-v1.tsv.gz --v1-set $v/v1-set.json.next $common --resend-before 2026-10-27 \
  ${waves:+--resend-hosts $v/resend-hosts.txt} --node2-min-weight $node2_min_weight --node2-hours $node2_hours ${REJOIN:+--rejoin-hosts $REJOIN} ${WIDENED_BEFORE:+--widened-before $WIDENED_BEFORE} > $v/v1-list.out
mv $v/v1-set.json.next $v/v1-set.json
chown ubuntu:ubuntu $v/*
