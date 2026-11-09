"""Lists the listed hosts the list walker marked joined with nothing queued and nothing ever sent: their stored pages queue again only when their send_cap changes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rocksdict import AccessType, Options, Rdict

JOINED, READY_COUNT, SENT_COUNT = b"J", b"Q", b"C"


def bucket_of(host: str) -> int:
    return hashlib.blake2b(host.encode(), digest_size=2).digest()[1]


def count(raw: bytes | None) -> int:
    return int.from_bytes(raw, "little") if raw and len(raw) == 8 else 0


def unqueued(db, host: str) -> bool:
    key = host.encode()
    return (
        db.get(JOINED + key) is not None
        and count(db.get(READY_COUNT + key)) == 0
        and count(db.get(SENT_COUNT + key)) == 0
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument(
        "--secondary-dir",
        required=True,
        help="a directory for the read-only instances, one folder a store",
    )
    parser.add_argument(
        "--list", required=True, help="a crawl list; its hosts are checked"
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        default=4,
        help="check only hosts weighted at most this",
    )
    parser.add_argument(
        "--buckets", help="buckets to read, such as 1,5,41; all 256 by default"
    )
    parser.add_argument("--out", required=True, help="the hosts found, one a line")
    args = parser.parse_args(argv)
    wanted = (
        {int(b) for b in args.buckets.split(",")} if args.buckets else set(range(256))
    )
    by_bucket: dict[int, list[str]] = {}
    for entry in json.load(open(args.list)):
        if (
            entry.get("weight", 1) <= args.max_weight
            and bucket_of(entry["host"]) in wanted
        ):
            by_bucket.setdefault(bucket_of(entry["host"]), []).append(entry["host"])
    found = checked = 0
    temporary = Path(f"{args.out}.new")
    with open(temporary, "w") as out:
        for bucket in sorted(by_bucket):
            secondary = Path(args.secondary_dir) / f"{bucket:03d}"
            secondary.mkdir(parents=True, exist_ok=True)
            options = Options(raw_mode=True)
            options.set_max_open_files(-1)
            db = Rdict(
                f"{args.buckets_dir}/{bucket:03d}",
                options=options,
                access_type=AccessType.secondary(str(secondary)),
            )
            for host in by_bucket[bucket]:
                checked += 1
                if unqueued(db, host):
                    found += 1
                    out.write(host + "\n")
            db.close()
    if not checked:
        temporary.unlink()
        sys.exit(f"no host in {args.list} to check; {args.out} left as it was")
    temporary.replace(args.out)
    print(
        f"{found} of {checked} hosts joined with nothing queued or sent",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
