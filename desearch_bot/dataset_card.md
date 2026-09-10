---
license: other
license_name: derived-from-public-lists
language:
  - en
pretty_name: Desearch domains
tags:
  - web-crawl
  - domains
  - bittensor
size_categories:
  - 1M<n<10M
configs:
  - config_name: domains
    data_files: domains/domains.parquet
---

# Desearch domains

The domains in scope for crawling by [Desearch](https://www.desearch.ai), Bittensor subnet 22.

## Contents

`domains/domains.parquet`

| column | type | description |
| --- | --- | --- |
| `host` | string | Registrable domain (eTLD+1), lowercase, without scheme or `www.` |

## Sources

The union of five public domain rankings, each entry resolved to its registrable domain using the
Public Suffix List:

- [Tranco](https://tranco-list.eu/)
- [Majestic Million](https://majestic.com/reports/majestic-million)
- [Open PageRank](https://www.domcop.com/openpagerank/)
- [BuiltWith Top 1M](https://builtwith.com/top-1m)
- [Cisco Umbrella Popularity List](https://umbrella-static.s3-us-west-1.amazonaws.com/index.html)

## Filtering

Removed from the union:

- Country-code TLDs outside English-speaking markets
- Adult, gambling, malware, phishing, cryptojacking, stalkerware, warez, hacking and DDoS domains
- Banking portals, URL shorteners, redirectors, advertising and tracking endpoints, dynamic DNS,
  DNS-over-HTTPS resolvers and residential proxies
- Social networks, forums, chat, webmail and file hosting, where pages are generated per user
  rather than published
- CDNs, certificate authorities, registrars and other infrastructure hostnames

Classification uses the [UT1 blacklists](https://dsi.ut-capitole.fr/blacklists/) from Université
Toulouse 1 Capitole together with public adult-domain blocklists.

Every surviving domain is then checked twice more:

- **It must resolve.** A domain with no address record is dropped. Rankings are built from
  historical traffic, so a meaningful share of any of them has since lapsed.
- **It must serve under its own name.** A domain that redirects to a different registrable domain
  is dropped in favour of its destination. Listing both would send two crawlers to one server
  under two names, and would give the same pages to two different miners.

Inclusion means a domain passed all of these. It does not mean the domain has been crawled.
`robots.txt` is read and obeyed at request time, and a domain that disallows the `DesearchBot`
token is never fetched.

## Crawler

Desearch identifies as `DesearchBot` and signs its requests with
[Web Bot Auth](https://developers.cloudflare.com/bots/concepts/bot/verified-bots/web-bot-auth/),
so operators can cryptographically verify the traffic. Requests to a host are sent no more than
one per second, and less often where `robots.txt` specifies a longer `Crawl-delay`.

To request removal, see [desearch.ai/crawler](https://www.desearch.ai/crawler).

## License

Derived from the public sources listed above, each of which retains its own terms.
