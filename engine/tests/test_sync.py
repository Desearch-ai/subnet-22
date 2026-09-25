import hashlib
import io
import json
import re
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from engine import build, service, sync
from engine.build import CorpusStats, canon, segment_source, write_index
from engine.chunking import doc_full, doc_head, para_chunks
from engine.search import Index, Unified

WIDTH = 4096


def toy(text: str) -> np.ndarray:
    """Hashed words: pages sharing words get close vectors, like a real model would."""
    vector = np.zeros(WIDTH, dtype=np.float32)
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % WIDTH] += 1
    return vector / np.linalg.norm(vector)


def page(url: str, topic: str, title: str = "") -> dict:
    text = "\n".join((f"{topic} story part{i} " * 12).strip() for i in range(3))
    return {
        "url": url,
        "title": title or topic.title(),
        "published": "2026-09-20",
        "text": text,
        "content_sha1": hashlib.sha1(text.encode()).hexdigest(),
    }


def rows_of(doc: dict) -> list[np.ndarray]:
    units = [doc_head(doc["title"], doc["text"]), doc_full(doc["title"], doc["text"])]
    return [toy(unit) for unit in units + para_chunks(doc["text"])]


def main_index(root, docs: list[dict]) -> Unified:
    vectors = [row for doc in docs for row in rows_of(doc)]
    shard = root / "shard.npy"
    np.save(shard, np.array(vectors, dtype=np.float16))

    def source():
        pos = 0
        for doc in docs:
            n = 2 + len(para_chunks(doc["text"]))
            yield doc, [(0, pos + i) for i in range(n)]
            pos += n

    write_index(root / "main", [("bench", source())], [str(shard)])
    return Unified(root / "main")


class Bucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def publish(
        self, docs: list[dict], task: str, stale: set[str] = frozenset()
    ) -> None:
        """What the publisher leaves on R2: page records, and one vectors file per embed task."""
        rows = []
        for doc in docs:
            key = f"pages/{canon(doc['url'])}"
            current = {
                **doc,
                "content_sha1": "newer" if doc["url"] in stale else doc["content_sha1"],
            }
            self.objects[key] = zstandard.ZstdCompressor().compress(
                json.dumps(current).encode()
            )
            kinds = [("head", 0), ("full", 0)] + [
                ("chunk", i) for i in range(len(para_chunks(doc["text"])))
            ]
            for (kind, index), vector in zip(kinds, rows_of(doc), strict=True):
                rows.append(
                    {
                        "text_id": f"{key}#{kind}{index}",
                        "page_key": key,
                        "url": doc["url"],
                        "content_sha1": doc["content_sha1"],
                        "kind": kind,
                        "index": index,
                        "text": "",
                        "model": "m",
                        "vector": vector.astype("<f2").tobytes(),
                    }
                )
        sink = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), sink)
        day = datetime.now(UTC).date().isoformat()
        self.objects[f"vectors/model=m/dt={day}/task={task}.parquet"] = sink.getvalue()


@pytest.fixture
def world(tmp_path, monkeypatch):
    main = main_index(
        tmp_path,
        [
            page("https://old.example/a", "harbor"),
            page("https://old.example/b", "garden"),
        ],
    )
    monkeypatch.setattr(service, "OUT", main.root)
    monkeypatch.setattr(service, "LIVE", tmp_path / "live")
    places = {
        "model": "m",
        "live": tmp_path / "live",
        "store": tmp_path / "store",
        "main": main.root,
    }
    synced = sync.Synced(tmp_path / "live" / "sync.db")
    yield tmp_path, Bucket(), synced, places
    synced.close()


def search(index: Index, question: str) -> list[str]:
    arms, _ = index.arms(question, toy(question))
    ranked, _ = index.fuse(arms, service.PROFILES["balanced"], 0, None)
    return [index.meta[doc]["url"] for doc in ranked]


def test_verified_vectors_become_searchable_without_a_restart(world):
    _, bucket, synced, places = world
    bucket.publish([page("https://new.example/c", "volcano")], "t1")

    name = sync.sync_once(bucket, synced, **places)

    assert (places["live"] / name / "READY").exists()
    index = service.load_index()
    assert [part.root.name for part in index.segments] == ["main", name]
    assert search(index, "volcano story")[0] == "https://new.example/c"
    assert sync.sync_once(bucket, synced, **places) is None, "a file is synced once"


def test_a_recrawled_page_replaces_its_older_copy(world):
    _, bucket, synced, places = world
    bucket.publish([page("https://old.example/a", "comet", "Comet")], "t1")
    sync.sync_once(bucket, synced, **places)

    index = service.load_index()

    found = search(index, "comet story")
    assert (
        found.count("https://old.example/a") == 1
        and found[0] == "https://old.example/a"
    )
    assert index.meta[index.key_ix["old.example/a"]]["title"] == "Comet"
    assert "harbor" not in " ".join(
        index.text(doc) for doc in range(len(index.meta)) if not index.hidden[doc]
    )


def test_vectors_for_a_page_that_has_since_changed_wait_for_their_own(world):
    _, bucket, synced, places = world
    bucket.publish(
        [
            page("https://new.example/c", "volcano"),
            page("https://new.example/d", "desert"),
        ],
        "t1",
        stale={"https://new.example/d"},
    )

    name = sync.sync_once(bucket, synced, **places)

    segment = Unified(places["live"] / name)
    assert [d["url"] for d in segment.meta] == ["https://new.example/c"]


def test_a_segments_keywords_score_on_the_main_indexs_scale(world):
    _, bucket, synced, places = world
    bucket.publish([page("https://new.example/c", "harbor")], "t1")
    name = sync.sync_once(bucket, synced, **places)
    main, segment = Unified(places["main"]), Unified(places["live"] / name)

    def weight(index, word):
        ids = index._vocab_ids({word})
        a, b = index.indptr[ids[0]], index.indptr[ids[0] + 1]
        return float(index.data[a:b].max())

    assert weight(segment, "harbor") == pytest.approx(weight(main, "harbor"), rel=0.05)
    stats = CorpusStats.of(places["main"])
    assert stats.chunks == len(main.owner) and stats.avgdl


def test_a_rebuild_folds_segments_in_and_the_newest_copy_wins(world):
    tmp_path, bucket, synced, places = world
    bucket.publish([page("https://old.example/a", "comet", "Comet")], "t1")
    name = sync.sync_once(bucket, synced, **places)
    files: list[str] = []
    old = json.loads((places["main"] / "files.json").read_text())

    def main_source():
        base = len(files)
        files.extend(old)
        yield from (
            (doc, rows) for doc, rows in segment_source_of_main(places["main"], base)
        )

    write_index(
        tmp_path / "rebuilt",
        [
            ("live", segment_source(places["live"] / name, files)),
            ("bench", main_source()),
        ],
        files,
    )

    rebuilt = Unified(tmp_path / "rebuilt")
    by_url = {d["url"]: d for d in rebuilt.meta}
    assert by_url["https://old.example/a"]["title"] == "Comet"
    assert set(by_url) == {"https://old.example/a", "https://old.example/b"}


def segment_source_of_main(root, base: int):
    """The test's main index was written in the segment layout, so it reads back the same way."""
    pos = 0
    with open(root / "docs.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            n = 2 + len(para_chunks(d["text"]))
            yield d, [(base, pos + i) for i in range(n)]
            pos += n


def test_the_service_skips_segments_a_rebuild_already_folded_in(world):
    _, bucket, synced, places = world
    bucket.publish([page("https://new.example/c", "volcano")], "t1")
    name = sync.sync_once(bucket, synced, **places)
    (places["main"] / "merged.json").write_text(json.dumps([name]))

    assert [part.root.name for part in service.load_index().segments] == ["main"]


def test_ready_segments_are_listed_oldest_first(tmp_path):
    for name in ("0000000002-b", "0000000001-a", "0000000003-c.tmp"):
        (tmp_path / name).mkdir()
    for name in ("0000000002-b", "0000000001-a"):
        (tmp_path / name / "READY").touch()

    assert [p.name for p in build.ready_segments(tmp_path)] == [
        "0000000001-a",
        "0000000002-b",
    ]


def test_the_running_service_picks_up_a_new_segment(world, monkeypatch):
    _, bucket, synced, places = world
    monkeypatch.setattr(service, "_state", {})
    before = service.engine()
    assert not service.refresh()

    bucket.publish([page("https://new.example/c", "volcano")], "t1")
    sync.sync_once(bucket, synced, **places)

    assert service.refresh()
    after = service.engine()
    assert after.segments[0] is before.segments[0], "the main index is not reloaded"
    assert search(after, "volcano story")[0] == "https://new.example/c"
