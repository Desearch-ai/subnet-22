# Bulk ingest

Some sources come from their own bulk data, not from crawling: Wikipedia snapshots, PubMed, PMC and arXiv. The ingest
lane turns a source's records into page rows in the bulk feed, so the compactor, the planner, miners' embedding and the
index treat them like crawled pages.

```
source files ─► python -m bulk <source> ─► run folder ─► publisher bulk ─► bulk feed ─┬─► planner's bulk lane
                                                                                           ├─► compactor ─► current/
                                                                                           └─► index
```

## A bulk page

- **Same row as a crawled page.** `python -m bulk` writes each page in the miners' upload schema, with v2 extraction.
  `publisher bulk` builds the record with the publisher's own code, so key, version, `doc_id` and `content_sha1` match
  a crawl of the same URL.
- **Its own feed.** Bulk files are numbered in `bulk/`, apart from the change log, so a large run never holds up
  crawled pages.
  - The compactor folds both feeds. Bulk pages sit before every change-log row in log order, and pages go by fetch time
    either way.
  - The planner reads the bulk feed in a lane of its own, at most `PLANNER_BULK_PAGES_PER_HOUR` pages an hour beside
    the change log's budget, and takes the pages the set's `bulk` rule admits. Its backfill leaves them to that lane.
  - The indexer tails the bulk feed beside the change log.
- **One id a row:** `bulk-<source>-<run>-<file>-<row>`, no miner, no validator. The task API never hands out such an id,
  so no miner is credited for it and no take-back of a miner's task touches it.
- **Fetch time is the source's time,** the snapshot's or the file's, not the time the run is published. A later crawl of
  the URL replaces the bulk version, and an older crawl gives way to it. A corrected run is published with a later
  `--as-of`.
- **Pages only.** The bulk feed carries no removals. Records a source deletes (PubMed `DeleteCitation`, arXiv deleted
  records) are kept in the run's `removed.txt` for the change log's removal path.
- **Papers** have the page type `paper`, and the planner keeps a paper's first 12 parts.

## Run

Each run gets its own new folder. `--as-of` is when the source's data was taken.

```bash
python -m bulk pmc-inventory --out inventory
python -m bulk pmc-pmids --inventory inventory/*.csv.gz --out pmc-pmids.txt
python -m bulk pubmed --out runs/pubmed-n0001 --as-of 2026-01-29T00:00:00Z --skip-pmids pmc-pmids.txt pubmed26n0001.xml.gz
python -m bulk pmc --out runs/pmc-1027 --as-of 2026-10-27T01:00:00Z --inventory inventory/*.csv.gz
python -m bulk arxiv-harvest --out oai --from 2026-10-01 --until 2026-10-27
python -m bulk arxiv --out runs/arxiv-1027 --as-of 2026-10-27T00:00:00Z oai/*.xml
python -m bulk hidden-categories enwiki_namespace_14.tar.gz > hidden.txt
python -m bulk wikipedia --out runs/enwiki-1101 --as-of 2026-11-01T00:00:00Z --hidden hidden.txt enwiki_namespace_0.tar.gz
```

Then publish the folder, into a local folder laid out as the pages bucket or into the change feed `serve` writes:

```bash
publisher bulk runs/pubmed-n0001 --log /tmp/log
publisher bulk runs/pubmed-n0001 --r2
```

`--r2` reads the same environment as `publisher serve`. A run that stops picks up where it left off: `published.txt`
in the run folder lists the batch files already published. Bulk files hold at most 10,000 pages and 32 MB of text.

## Sources

| Source | Input | Pages | Kept |
|---|---|---|---|
| `pubmed` | baseline and daily update files | one per citation | citations with an abstract; `--skip-pmids` leaves out those PMC has in full |
| `arxiv` | OAI-PMH pages or the Kaggle metadata file | one abstract page per paper | every paper (arXiv metadata is CC0) |
| `pmc` | the `pmc-oa-opendata` bucket, newest version of each article | full text without references | open access under CC0, CC BY, CC BY-SA or CC BY-ND, not retracted |
| `wikipedia` | Wikimedia Enterprise snapshots | the article as the live page shows it | namespace 0, no disambiguation pages |

Wikipedia articles are framed as the live page frames them and read by the same `extract()` a crawl uses, so the text
matches a crawl of the live page. Hidden categories come from the category-namespace snapshot.

Sources carry their own terms: Wikimedia content is CC BY-SA (credit, link, license), and PMC asks users to credit NLM
as the source.
