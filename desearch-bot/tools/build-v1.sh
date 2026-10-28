#!/bin/sh
# Builds v1-set.json and the v1 crawl list into ~ubuntu/crawl-1b/v1 from the registry (read-only) and the staged inputs; NEWS=B takes every news site from 2020, UNDATED=article takes undated news articles; RESEND_WAVES=N gives the sites of resend waves 1 to N resend_before 2026-10-27 (none by default).
set -e
v=/home/ubuntu/crawl-1b/v1
f=/var/lib/task-api-feeder
news=${NEWS:-A}
undated=${UNDATED:-out}
resend_waves=${RESEND_WAVES:-0}
rows=$v/benchmark-set-rows.json
[ "$news" = B ] && rows=$v/benchmark-set-rows-B.json
set -a; . /opt/desearch-bot/.env; set +a
common="--backing $f/backing.tsv --overrides /home/ubuntu/crawl-1b/overrides-new.tsv --excluded $f/excluded-domains.txt --judged /home/ubuntu/crawl-1b/judged-final-2026-10-27.tsv"
python3 $v/crawl_list.py --out $v/domains-judged.json --decisions $v/decisions-judged.tsv.gz $common > $v/judged.out
python3 $v/v1_set.py --out $v/v1-set.json --news $news --undated $undated --decisions $v/decisions-judged.tsv.gz --hand $v/v1-hand.tsv \
  --rows $v/bulk-set-rows.json --rows $rows --rows $v/extra-set-rows.json --backing $f/backing.tsv --news-domains $v/news-domains.tsv.gz --outlets $v/news-outlets-2026-10-28.tsv \
  --judged /home/ubuntu/crawl-1b/judged-final-2026-10-27.tsv --counts $v/v1-store-counts-2026-10-28.tsv.gz --news-counts $v/news-counts-2026-10-28b.tsv.gz --too-big $v/too-big.tsv > $v/v1-set.out
: > $v/resend-hosts.txt
i=1
while [ $i -le $resend_waves ]; do
  cat $v/resend-waves-$news/resend-wave-$(printf %02d $i).txt >> $v/resend-hosts.txt
  i=$((i + 1))
done
python3 $v/crawl_list.py --out $v/domains-v1.json --decisions $v/decisions-v1.tsv.gz --v1-set $v/v1-set.json $common --resend-before 2026-10-27 --resend-hosts $v/resend-hosts.txt > $v/v1-list.out
chown ubuntu:ubuntu $v/*
