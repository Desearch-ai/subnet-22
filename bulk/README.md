# Bulk ingest

Some sources come from their own bulk data, not from crawling, such as Wikipedia snapshots, PubMed, PMC and arXiv. The
ingest lane turns a source's records into page rows in the bulk feed, so the compactor, the planner, miners' embedding
and the index treat them like crawled pages.

```
source files ─► bulk <source> ─► run folder ─► publisher bulk ─► bulk feed ─┬─► planner's bulk lane
                                                                             ├─► compactor ─► current/
                                                                             └─► index
```

## A bulk page

- **Same row as a crawled page.** `bulk` writes each page in the miners' upload schema, with v2 extraction.
  `publisher bulk` builds the record with the publisher's own code, so key, version, `doc_id` and `content_sha1` match
  a crawl of the same URL.
- **Its own feed.** Bulk files are numbered in `bulk/`, apart from the change log, so a large run never holds up
  crawled pages.
  - The compactor folds both feeds. Bulk pages sit before every change-log row in log order, and pages go by fetch time
    either way.
  - The planner reads the bulk feed in a lane of its own, at most `PLANNER_BULK_PAGES_PER_HOUR` pages an hour beside
    the change log's budget, and takes the pages the set's `bulk` rule admits. While the lane is on, the backfill
    leaves it the bulk files it has yet to read.
  - The indexer tails the bulk feed beside the change log.
- **One id a row:** `bulk-<source>-<run>-<file>-<row>`, no miner, no validator. The task API never hands out such an id,
  so no miner is credited for it and no take-back of a miner's task touches it.
- **Fetch time is the source's time,** the snapshot's or the file's, not the time the run is published. A later crawl of
  the URL replaces the bulk version, and an older crawl gives way to it. A corrected run is published with a later
  `--as-of`.
- **Pages only.** The bulk feed carries no removals. Records a source deletes (PubMed `DeleteCitation`, arXiv deleted
  records) are kept in the run's `removed.txt` for the change log's removal path.
- **Papers** have the page type `paper`. The planner embeds a paper's page text and its first 4 parts, the abstract and
  the start of the body; the whole text stays in the page for keyword search and highlights.

## Run

`bulk` is the Rust binary in `bulk/` (`cargo build --release -p bulk`). Each run gets its own new folder. `--as-of`
is when the source's data was taken.

```bash
bulk pmc-inventory --out inventory
bulk pmc-pmids --inventory inventory/*.csv.gz --out pmc-pmids.txt
bulk pubmed --out runs/pubmed-n0001 --as-of 2026-01-29T00:00:00Z --skip-pmids pmc-pmids.txt pubmed26n0001.xml.gz
bulk pmc --out runs/pmc-1027 --as-of 2026-10-27T01:00:00Z --inventory inventory/*.csv.gz
curl -O https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz
bulk pmc-order --inventory inventory/*.csv.gz --pmc-ids PMC-ids.csv.gz --out pmc-order.txt
bulk pmc --out runs/pmc-1110-01 --run 20261110-01 --as-of 2026-11-10T00:00:00Z --names pmc-order-01.txt
bulk arxiv-harvest --out oai --from 2026-10-01 --until 2026-10-27
bulk arxiv --out runs/arxiv-1027 --as-of 2026-10-27T00:00:00Z oai/*.xml
bulk arxiv-text --out runs/arxiv-text-1110 --as-of 2024-09-04T00:00:00Z --oai oai *_arxiv-papers.jsonl.gz
bulk hidden-categories enwiki_namespace_14.tar.gz > hidden.txt
bulk wikipedia --out runs/enwiki-1101 --as-of 2026-11-01T00:00:00Z --hidden hidden.txt enwiki_namespace_0.tar.gz
```

The shared extraction runs in Python: `bulk wikipedia`, `medlineplus`, `eurlex`, `subnet-sites`, `subnet-github` and
`bittensor-fetch` start `python -m bulk.page_worker` processes, and `bulk pmc` starts `python -m bulk.pmc_render`
processes. `--python` names the interpreter (one with lxml), `--python-path` the repo root, and `--workers`
(`--render-workers` for PMC) how many.

Then publish the folder into the bulk feed, in a local folder laid out as the pages bucket (`--log`) or in the pages
bucket itself (`--r2`):

```bash
publisher bulk runs/pubmed-n0001 --log /tmp/log
publisher bulk runs/pubmed-n0001 --r2
```

`--r2` writes to the S3-compatible store named by the same environment as `publisher serve`. A run that stops picks up
where it left off: `published.txt` in the run folder lists the batch files already published. Bulk files hold at most
10,000 pages and 32 MB of text.

## Sources

| Source | Input | Pages | Kept |
|---|---|---|---|
| `pubmed` | baseline and daily update files | one per citation | citations with an abstract; `--skip-pmids` leaves out those PMC has in full |
| `arxiv` | OAI-PMH pages or the Kaggle metadata file | one abstract page per paper | every paper (arXiv metadata is CC0) |
| `arxiv-text` | Common Pile's `arxiv_papers` files, with the OAI-PMH pages for each paper's licence | full text without references, at the paper's HTML view (`arxiv.org/html/<id>` from 2312 on, `ar5iv.labs.arxiv.org/html/<id>` before) | papers whose arXiv record is CC BY, CC BY-SA or CC0 |
| `pmc` | the `pmc-oa-opendata` bucket, newest version of each article | full text without references, with its DOI | open access under CC0, CC BY, CC BY-SA or CC BY-ND, not retracted |
| `wikipedia` | Wikimedia Enterprise snapshots | the article as the live page shows it | namespace 0, no disambiguation pages |
| `factbook` | factbook.json, the World Factbook's final edition | one per entity, at its worldfactbookarchive.org page | entities with a page of their own on the archive |
| `medlineplus` | the daily health topics XML and the Genetics summaries | one per topic, condition, gene and chromosome | English; the summaries only |
| `uscode` | an OLRC release point's all-titles zip | one per section | single sections with statutory text |
| `ecfr` | govinfo's eCFR title XML, with eCFR's structure for each title's addresses | one per section | sections the current structure lists |
| `fedreg` | govinfo's daily issues, with an index from federalregister.gov's API | one per document | documents the index gives an address |
| `eurlex` | Cellar's English texts, listed through its SPARQL endpoint | one per act or judgment | acts in force and CJEU judgments |
| `dailymed` | DailyMed's full release or daily update zips | one per label | human prescription and OTC labels, newest version, without images |
| `ctgov` | ClinicalTrials.gov's all-studies zip | one per study | without contact details or posted results tables |
| `uniprot` | UniProtKB/Swiss-Prot's release XML | one per reviewed entry | reviewed entries only |
| `pubchem` | PUG-View's Record Description annotations | one per compound | descriptions from sources whose terms allow reuse, each credited |
| `gutenberg` | the plain-text archive and catalog | one per book, its opening | English, public domain in the USA, without the licence or the project's name in the text |

`pmc-order` lists the newest version of every PMC article, most cited per year since publication first, by
OpenAlex's citation counts (joined by PMID through NCBI's PMCID table), so a run fed slices of that list embeds the
most-read articles first.

Wikipedia articles are framed as the live page frames them and read by the same `extract()` a crawl uses, so the text
matches a crawl of the live page. Hidden categories come from the category-namespace snapshot.

Sources carry their own terms: Wikimedia content is CC BY-SA (credit, link, license), PMC asks users to credit NLM
as the source, and an arXiv full text keeps its paper's CC licence (credit the authors, link the paper).

## Refresh

The Rust binary `refresh` (crate `refresh/`) keeps a source current. `refresh run <source>` runs the source's fetch and
build commands, keeps only the pages that are new or changed since the last run, publishes them and hides the pages
the source dropped. The sources and their commands are in `refresh/src/sources.rs`.

- **State:** `~/refresh/state/<source>.sqlite` holds each page's URL, a hash of its row (every column but the fetch's
  own and the build's volatile ones, such as a credit line naming the run's date), the run that last published it and
  whether it is gone. A page is compared with its last hash under any build of its source. `refresh seed <source> <build> <run folders>` records
  runs published before the source's first refresh; `refresh diff` checks a build folder against the state.
- **Guards:** a full-list build with under 95% of the pages it had publishes and hides nothing. More than 30% of a
  build changed is flagged in the report.
- **Incremental sources** such as arXiv fetch from `{since}`, the day of their last run that published, to `{until}`,
  today. A first run takes `--since`.
- **Publish:** the runner leaves each diff in `~/refresh/outbox`. `refresh-pull` on the planner box takes it every 10
  minutes, publishes it with the publish script kept on that box and sends the output back as a receipt in
  `~/refresh/receipts`.
- **Hide:** pages a source dropped go to index-1 as exact-path rules in the source's own section of
  `hidden-urls.txt` (`refresh hide-section`, the forced command of the crawler box's key), rewritten whole each run,
  so a page that comes back is shown again.
- **Report:** one JSON line a run in `~/refresh/status.jsonl`, and one line to the ops channel when
  `REFRESH_DISCORD_WEBHOOK` is set.

`deploy/` holds the crawler box's `refresh@.service` and timers, the planner box's `refresh-pull` and the crawler
box's forced command for it, `refresh-send.sh`.
