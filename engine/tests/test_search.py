from types import SimpleNamespace

import numpy as np

from engine.search import Unified, prefix, unit


def test_per_doc_keeps_the_best_chunk_of_each_document():
    engine = SimpleNamespace(owner=np.array([0, 0, 1, 1, 2]))
    rows = np.array([0, 1, 2, 3, 4])
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.7], dtype=np.float32)

    docs, best = Unified._per_doc(engine, rows, scores)

    assert docs.tolist() == [0, 1, 2]
    assert np.allclose(best, [0.9, 0.5, 0.7])


def test_per_doc_handles_an_arm_that_matched_nothing():
    """A query whose terms are absent from the corpus leaves BM25 with no rows."""
    engine = SimpleNamespace(owner=np.array([0, 1, 2]))

    docs, best = Unified._per_doc(engine, np.array([], dtype=np.int64), np.array([]))

    assert docs.size == 0 and best.size == 0


def test_prefix_keeps_full_queries_and_renormalises_cut_ones():
    q = unit(np.arange(1, 9, dtype=np.float32))

    assert prefix(q, 8) is q
    cut = prefix(q, 4)
    assert cut.shape == (4,) and np.isclose(np.linalg.norm(cut), 1.0)


def test_rescore_uses_the_stored_width():
    stored = unit(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    engine = SimpleNamespace(
        cfile=np.array([0, 0]), crow=np.array([0, 1]), files=[stored.astype(np.float16)]
    )
    q = unit(np.array([1, 0, 0, 0, 5, 5, 5, 5], dtype=np.float32))

    scores = Unified._rescore_chunks(engine, np.array([0, 1]), q)

    assert np.allclose(scores, [1.0, 0.0], atol=1e-3)
