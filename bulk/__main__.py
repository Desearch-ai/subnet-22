"""python -m bulk: turn one bulk source into a folder of batch files for `publisher bulk`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyarrow.fs import S3FileSystem

from bulk import arxiv, pmc, pubmed, wikipedia
from bulk.rows import Batches, parse_as_of

USER_AGENT = "DesearchBulk/1.0 (+https://desearch.ai)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bulk")
    commands = parser.add_subparsers(dest="command", required=True)

    def source(name: str, help: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help)
        command.add_argument(
            "--out",
            type=Path,
            required=True,
            help="a new folder for this run's batch files",
        )
        command.add_argument(
            "--as-of",
            required=True,
            help="when the source's data was taken, as an ISO time with a zone; the pages' fetch time",
        )
        command.add_argument(
            "--run",
            default="",
            help="the run's name in its task id, bulk-<source>-<run>; the as-of date by default",
        )
        command.add_argument(
            "--limit",
            type=int,
            default=0,
            help="stop after this many pages, for trials",
        )
        return command

    found = source("pubmed", "PubMed baseline or update files (.xml.gz)")
    found.add_argument("files", type=Path, nargs="+")
    found.add_argument(
        "--skip-pmids",
        type=Path,
        help="PMIDs to leave out, one per line, such as those the PMC run took",
    )
    found = source(
        "arxiv", "arXiv OAI-PMH pages (.xml) or the Kaggle metadata snapshot (.json)"
    )
    found.add_argument("files", type=Path, nargs="+")
    found = source(
        "arxiv-text",
        "Common Pile's arXiv full texts (.jsonl.gz), for papers under CC BY, CC BY-SA or CC0",
    )
    found.add_argument("files", type=Path, nargs="+")
    found.add_argument(
        "--oai",
        type=Path,
        required=True,
        help="a folder of OAI-PMH pages from `arxiv-harvest`, whose licences decide which papers are taken",
    )
    found = source("pmc", "PMC open-access articles under commercial-use licenses")
    found.add_argument(
        "--inventory",
        type=Path,
        nargs="*",
        help="the bucket's inventory CSV files; read from the bucket",
    )
    found.add_argument(
        "--folder",
        type=Path,
        help="a folder of <name>.json and <name>.xml pairs instead of the bucket",
    )
    found.add_argument(
        "--names",
        type=Path,
        help="version names to read from the bucket in this order, one per line, as `pmc-order` writes them",
    )
    found.add_argument("--workers", type=int, default=32)
    found.add_argument(
        "--pmids-out",
        type=Path,
        help="write the PMIDs taken here, for the PubMed run's --skip-pmids",
    )
    found = source(
        "wikipedia",
        "Wikimedia Enterprise snapshots of English Wikipedia and its sister projects",
    )
    found.add_argument("files", type=Path, nargs="+")
    found.add_argument(
        "--hidden",
        type=Path,
        required=True,
        help="hidden category names, one per line, from `hidden-categories`",
    )
    found.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes that frame and extract articles",
    )

    harvest = commands.add_parser(
        "arxiv-harvest", help="save arXiv OAI-PMH pages for a date range"
    )
    harvest.add_argument("--out", type=Path, required=True)
    harvest.add_argument("--from", dest="since", required=True)
    harvest.add_argument("--until", required=True)
    inventory = commands.add_parser(
        "pmc-inventory", help="download the PMC bucket's newest inventory CSV files"
    )
    inventory.add_argument("--out", type=Path, required=True)
    ordered = commands.add_parser(
        "pmc-order",
        help="the newest version of each PMC article, most cited per year first, by OpenAlex's counts",
    )
    ordered.add_argument("--inventory", type=Path, nargs="+", required=True)
    ordered.add_argument(
        "--pmc-ids", type=Path, required=True, help=f"NCBI's PMCID table, from {pmc.PMC_IDS}"
    )
    ordered.add_argument("--out", type=Path, required=True)
    ordered.add_argument("--workers", type=int, default=8)
    pmids = commands.add_parser("pmc-pmids", help="PMIDs of the PMC versions taken, from metadata alone")
    pmids.add_argument("--inventory", type=Path, nargs="+", required=True)
    pmids.add_argument("--out", type=Path, required=True)
    pmids.add_argument("--workers", type=int, default=64)
    hidden = commands.add_parser(
        "hidden-categories",
        help="print the hidden categories of a category-namespace snapshot",
    )
    hidden.add_argument("files", type=Path, nargs="+")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "arxiv-harvest":
        print(
            f"{arxiv.harvest(args.out, args.since, args.until, USER_AGENT)} pages saved in {args.out}"
        )
        return 0
    if args.command == "pmc-inventory":
        for path in pmc.latest_inventory(args.out, USER_AGENT):
            print(path)
        return 0
    if args.command == "pmc-order":
        names = pmc.latest_versions(pmc.inventory_keys(args.inventory))
        ids = pmc.pmc_ids(args.pmc_ids)
        wanted = {ids[p][0] for p in names if p in ids and ids[p][0]}
        files = pmc.openalex_files(USER_AGENT)
        cited = pmc.citations(files, S3FileSystem(anonymous=True, region="us-east-1"), wanted, args.workers)
        ordered = pmc.order(names, ids, cited, datetime.now(timezone.utc).year)
        args.out.write_text("".join(f"{name}\n" for name in ordered))
        print(json.dumps({"versions": len(ordered), "with_pmid": len(wanted), "cited": len(cited), "out": str(args.out)}))
        return 0
    if args.command == "pmc-pmids":
        names = pmc.latest_versions(pmc.inventory_keys(args.inventory))
        with open(args.out, "w") as out:
            for pmid in pmc.taken_pmids(sorted(names.values()), USER_AGENT, args.workers):
                out.write(f"{pmid}\n")
        return 0
    if args.command == "hidden-categories":
        for path in args.files:
            for name in wikipedia.hidden_categories(wikipedia.articles(path)):
                print(name)
        return 0

    as_of = parse_as_of(args.as_of)
    batches = Batches(
        args.out, args.command, args.run or as_of.strftime("%Y%m%d"), as_of
    )
    if args.command == "pubmed":
        skip = set(args.skip_pmids.read_text().split()) if args.skip_pmids else set()
        counts = pubmed.run(args.files, batches, as_of, skip, args.limit)
    elif args.command == "arxiv":
        counts = arxiv.run(args.files, batches, as_of, args.limit)
    elif args.command == "arxiv-text":
        papers = arxiv.open_papers(sorted(args.oai.glob("*.xml")))
        counts = arxiv.run_text(args.files, papers, batches, as_of, args.limit)
    elif args.command == "pmc":
        if args.folder:
            found = pmc.from_folder(args.folder)
        elif args.names:
            found = pmc.from_bucket(args.names.read_text().split(), USER_AGENT, args.workers)
        else:
            names = pmc.latest_versions(pmc.inventory_keys(args.inventory or []))
            found = pmc.from_bucket(sorted(names.values()), USER_AGENT, args.workers)
        pmids: list[str] = []
        counts = pmc.run(found, batches, as_of, args.limit, pmids)
        if args.pmids_out:
            args.pmids_out.write_text("".join(f"{p}\n" for p in pmids))
    else:
        hidden = set(args.hidden.read_text().splitlines())
        counts = wikipedia.run(args.files, batches, as_of, hidden, args.limit, args.workers)
    batches.close(dict(counts))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "files": len(batches.files),
                "rows": batches.rows,
                "removed": len(batches.removed),
                **counts,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
