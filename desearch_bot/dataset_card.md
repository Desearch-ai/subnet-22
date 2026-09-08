---
license: other
license_name: derived-from-public-lists
language:
  - en
pretty_name: Desearch crawlable domains
tags:
  - web-crawl
  - domains
  - sitemaps
  - bittensor
configs:
  - config_name: domains
    data_files: domains/domains.parquet
---

# Desearch crawlable domains

The English-language domains Desearch miners crawl. Every host listed here was visited by the
Desearch Bot: its robots.txt was read and obeyed, a sitemap was fetched and parsed, and its
homepage was checked for readable English text. `sitemap_url` is where a crawl of that domain
starts.

`domains/domains.parquet` is rewritten as the bot works through its candidate list, so it always
reflects the current qualified set. `domains/stats.json` carries the counts and the rejection
breakdown for the same moment.

## Columns

| column | meaning |
|---|---|
| `host` | registrable domain, lowercase, no `www.` |
| `sitemap_url` | the sitemap that parsed; where a crawl of this domain starts |
| `sitemap_kind` | `index` when it points at more sitemaps, `urlset` when it points at pages |
| `url_count` | URLs discovered across this domain's sitemap tree |
| `crawl_delay` | seconds between requests this host asks for in robots.txt, when set; respect it |

Rank, source type, language and the visit timestamp are kept in the bot's own database rather
than here, since they say nothing about how to crawl a domain.

## How a domain qualifies

1. **robots.txt** is fetched and parsed. A `Disallow: /` for `DesearchBot`, or for `*` when we are
   not named, rejects the domain. A declared `Crawl-delay` is recorded and respected.
2. **A sitemap** is taken from robots.txt, or tried at `/sitemap.xml` and `/sitemap_index.xml`. It
   must parse, and when it lists pages directly it must hold at least ten.
3. **The homepage** must return 200, show at least 200 characters of visible text, not be a bot
   wall, and be detected as English.

## What is excluded before any visit

Adult, malware, phishing, cryptomining, stalkerware, hacking, warez, gambling, link shorteners, ad
networks, dynamic DNS and proxy hosts, from the UT1 category lists, matched on the exact host.
Search engines, social networks, forums, chat, webmail, file hosting, media streaming and
marketplaces, whose pages are generated per user rather than published as documents. CDNs,
resolvers, certificate authorities, registrars and analytics endpoints. Country-code domains
outside English-speaking markets.

## Crawling policy

The bot identifies itself as
`Mozilla/5.0 (compatible; DesearchBot/1.0; +https://www.desearch.ai/crawler)`, obeys robots.txt
for that token, and requests at most one page per second per host unless a longer `Crawl-delay` is
declared. It reads sitemaps and homepages only; page content is fetched by miners under the same
rules.

## Sources

Candidate hosts come from Tranco, Majestic Million, Open PageRank, BuiltWith Top 1M and Cisco
Umbrella. Categories come from the UT1 blacklists. Registrable domains follow the Public Suffix
List. Each source carries its own terms; this dataset is derived data with attribution, and the
ranks are each source's own and are not comparable between sources.

## Reproducing

The bot is open source in [subnet-22](https://github.com/Desearch-ai/subnet-22) under
`desearch_bot/`:

```bash
python -m desearch_bot.cli candidates --out build
python -m desearch_bot.cli load --candidates build/candidates.parquet
python -m desearch_bot.cli discover --limit 100000
python -m desearch_bot.cli publish --push
```
