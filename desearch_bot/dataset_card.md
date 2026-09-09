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

The domain list for Desearch subnet-22. **{{DOMAINS}} domains**, one column, one row each.

```
domains/domains.parquet
  host: string   registrable domain, lowercase, no scheme and no www.
```

That is the whole file on purpose. Sitemap locations, crawl delays, refresh schedules and
categories are operational state that changes as the crawler works; they live in Desearch's
database, not here. This list answers one question: which domains are in scope.

## How it was built

Five public domain rankings were merged on the registrable domain (eTLD+1, resolved with the
Public Suffix List):

| Source | Domains |
| --- | ---: |
| Open PageRank | 7,015,762 |
| Tranco | 999,031 |
| Majestic Million | 997,296 |
| BuiltWith Top 1M | 979,149 |
| Cisco Umbrella | 256,847 |
| **Union** | **8,541,592** |

The union was then filtered. Domains on a non-English country-code TLD were dropped, as were
domains categorised as adult, gambling, malware, phishing, cryptojacking, stalkerware, warez,
hacking, DDoS, banking portals, URL shorteners, redirectors, ad and tracking endpoints, dynamic
DNS, DNS-over-HTTPS resolvers, residential proxies, social networks, forums, chat, webmail and
file hosting. Categories come from the [UT1 blacklists](https://dsi.ut-capitole.fr/blacklists/)
maintained by Université Toulouse 1 Capitole, combined with four public adult-domain blocklists.
CDNs, certificate authorities, registrars and other infrastructure hostnames were removed by name.

Being on this list means a domain passed those filters. It does not mean the domain has been
crawled, or that it will be: robots.txt is read and obeyed at crawl time, and a domain that
disallows the `DesearchBot` token is never fetched.

## Crawler

Desearch crawls as `DesearchBot`, signing requests with
[Web Bot Auth](https://developers.cloudflare.com/bots/concepts/bot/verified-bots/web-bot-auth/)
so operators can verify the traffic is ours. Requests to a host are paced at least one second
apart, and longer when robots.txt asks. To have a domain removed, see
[desearch.ai/crawler](https://www.desearch.ai/crawler).

Built {{BUILT_AT}}.
