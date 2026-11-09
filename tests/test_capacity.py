import hashlib

import numpy as np
import pytest

from desearch import capacity
from desearch.capacity import (
    LENGTHS,
    SAMPLES,
    WORDS,
    Test as CapacityTest,
)
from desearch.embedding import encode_vector

DIMS = 8
HOTKEY = "5Honest"
OTHER = "5Other"
SEED = hashlib.sha256(b"seed").digest()
SAMPLE_HASH = hashlib.sha256(b"sample block").digest()
TEST = CapacityTest(1000, SEED)


def vector_of(text: str) -> np.ndarray:
    raw = hashlib.shake_256(text.encode()).digest(DIMS)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5


def committed(hotkey: str, count: int, honest=lambda index: True):
    raws = []
    for index in range(count):
        text = capacity.text_at(SEED, hotkey, index) if honest(index) else f"garbage {index}"
        raws.append(encode_vector(vector_of(text)))
    levels = capacity.tree_levels([capacity.leaf_hash(i, raw) for i, raw in enumerate(raws)])
    tree_root = levels[-1][0]
    return raws, levels, tree_root, capacity.commit_root(SEED, hotkey, count, tree_root)


def reveal_of(raws, levels, tree_root, count, hotkey=HOTKEY):
    picked = capacity.sample_indices(SAMPLE_HASH, hotkey, count)
    items = {i: (raws[i], capacity.tree_proof(levels, i)) for i in picked}
    return capacity.decode_reveal(capacity.encode_reveal(tree_root, count, items))


def verdict(hotkey, count, root, reveal):
    opened = capacity.opened_vectors(SEED, hotkey, count, root, SAMPLE_HASH, reveal, DIMS)
    if isinstance(opened, str):
        return opened
    picked, vectors = opened
    reference = np.array([vector_of(capacity.text_at(SEED, hotkey, i)) for i in picked])
    return "pass" if capacity.matches(capacity.similarities(vectors, reference)) else "mismatch"


def test_the_spec_stays_the_same_everywhere():
    """Validators and miners must rebuild the same bytes on any machine; a change here is a new version of the test."""
    assert len(LENGTHS) == 64 and max(LENGTHS) == 6000
    assert len(WORDS) == 512 and len(set(WORDS)) == 512
    assert hashlib.sha256(" ".join(WORDS).encode()).hexdigest() == (
        "7f40287e49eb5919afe2a7a35db0498d575b34d5ff2335f18e65cff0810b8193"
    )
    assert hashlib.sha256(capacity.text_at(SEED, HOTKEY, 0).encode()).hexdigest() == (
        "f061391bb5c478ca5b4a3904c92f789569b51e8c1e7379bf84c3503a1619ff9e"
    )
    assert capacity.chars_of(SEED, HOTKEY, 100) == 196550
    assert capacity.sample_indices(SAMPLE_HASH, HOTKEY, 1000)[:4] == [864, 813, 630, 945]


def test_a_text_has_exactly_its_length_and_is_ascii():
    for index in range(300):
        text = capacity.text_at(SEED, HOTKEY, index)
        assert len(text) == capacity.text_length(SEED, HOTKEY, index)
        assert text.isascii()


def test_texts_differ_by_index_hotkey_and_seed():
    assert capacity.text_at(SEED, HOTKEY, 1) != capacity.text_at(SEED, HOTKEY, 2)
    assert capacity.text_at(SEED, HOTKEY, 1) != capacity.text_at(SEED, OTHER, 1)
    assert capacity.text_at(SEED, HOTKEY, 1) != capacity.text_at(SAMPLE_HASH, HOTKEY, 1)


def test_texts_share_no_useful_prefix():
    starts = [capacity.text_at(SEED, HOTKEY, index)[:24] for index in range(2000)]
    assert len(set(starts)) == len(starts)


def test_a_test_block_needs_a_quiet_run_before_it():
    candidates, quiet = [], []
    for n in range(200_000):
        block_hash = hashlib.sha256(n.to_bytes(8, "big")).digest()
        if capacity.is_candidate(block_hash):
            candidates.append((n, block_hash))
        else:
            quiet.append(block_hash)
        if len(candidates) > 200 and len(quiet) > 100:
            break
    block, block_hash = candidates[0]
    assert capacity.test_at(block, block_hash, quiet[:60]) == CapacityTest(block, block_hash)
    assert capacity.test_at(block, block_hash, [*quiet[:59], candidates[1][1]]) is None
    assert capacity.test_at(block, quiet[0], quiet[1:61]) is None


def test_about_one_block_in_three_hundred_starts_a_test():
    hashes = [hashlib.sha256(n.to_bytes(8, "big")).digest() for n in range(240_000)]
    flags = [capacity.is_candidate(h) for h in hashes]
    tests = sum(1 for n in range(60, len(hashes)) if flags[n] and not any(flags[n - 60 : n]))
    assert 240_000 / 340 < tests < 240_000 / 280


def test_the_test_timeline_follows_from_its_block():
    assert (TEST.deadline, TEST.sample_block, TEST.reveal_by) == (1006, 1010, 1035)


def test_a_tag_round_trips_and_rejects_anything_else():
    assert capacity.parse_tag(capacity.make_tag(TEST, 4000)) == (1000, SEED[:4], 4000)
    assert len(capacity.make_tag(CapacityTest(999_999_999_999, SEED), 999_999_999_999)) <= 128
    for bad in (b"", b"sn22c1:", b"sn22c1:0:00000000:1", b"sn22c1:1:zzzzzzzz:1", b"sn22c1:1:00000000:01", b"other:1:00000000:1"):
        assert capacity.parse_tag(bad) is None


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 8, 9, 47, 48, 49, 100, 257])
def test_an_honest_commitment_passes_at_every_tree_shape(count):
    raws, levels, tree_root, root = committed(HOTKEY, count)
    assert verdict(HOTKEY, count, root, reveal_of(raws, levels, tree_root, count)) == "pass"


def test_samples_are_distinct_and_within_the_count():
    picked = capacity.sample_indices(SAMPLE_HASH, HOTKEY, 1000)
    assert len(picked) == SAMPLES == len(set(picked)) and all(0 <= i < 1000 for i in picked)
    assert picked != capacity.sample_indices(SAMPLE_HASH, OTHER, 1000)


def test_an_inflated_count_cannot_open_its_samples():
    raws, levels, tree_root, _ = committed(HOTKEY, 300)
    root = capacity.commit_root(SEED, HOTKEY, 600, tree_root)
    picked = capacity.sample_indices(SAMPLE_HASH, HOTKEY, 600)
    items = {i: (raws[i], capacity.tree_proof(levels, i)) for i in picked if i < 300}
    assert verdict(HOTKEY, 600, root, capacity.Reveal(tree_root, 600, items)) in ("sample_missing", "bad_proof")
    # Vectors made up for the missing samples are not in the committed tree.
    for i in picked:
        items.setdefault(i, (raws[0], capacity.tree_proof(levels, 0)))
    assert verdict(HOTKEY, 600, root, capacity.Reveal(tree_root, 600, items)) == "bad_proof"


def test_mostly_fake_vectors_are_caught():
    raws, levels, tree_root, root = committed(HOTKEY, 400, honest=lambda i: i % 2 == 0)
    assert verdict(HOTKEY, 400, root, reveal_of(raws, levels, tree_root, 400)) == "mismatch"


def test_another_hotkeys_commitment_means_nothing():
    raws, levels, tree_root, root = committed(HOTKEY, 100)
    assert verdict(OTHER, 100, root, reveal_of(raws, levels, tree_root, 100, OTHER)) == "wrong_root"


def test_a_copied_tree_under_its_own_root_fails_on_the_vectors():
    raws, levels, tree_root, _ = committed(HOTKEY, 100)
    root = capacity.commit_root(SEED, OTHER, 100, tree_root)
    assert verdict(OTHER, 100, root, reveal_of(raws, levels, tree_root, 100, OTHER)) == "mismatch"


def test_a_vector_changed_after_the_commitment_fails_its_proof():
    raws, levels, tree_root, root = committed(HOTKEY, 100)
    reveal = reveal_of(raws, levels, tree_root, 100)
    index = next(iter(reveal.items))
    raw, proof = reveal.items[index]
    changed = encode_vector(vector_of("something else"))
    assert changed != raw
    items = {**reveal.items, index: (changed, proof)}
    assert verdict(HOTKEY, 100, root, capacity.Reveal(tree_root, 100, items)) == "bad_proof"


def test_a_vector_of_the_wrong_length_or_not_unit_is_malformed():
    bad = np.full(DIMS, 3.0, dtype=np.float16).tobytes()
    levels = capacity.tree_levels([capacity.leaf_hash(0, bad)])
    root = capacity.commit_root(SEED, HOTKEY, 1, levels[-1][0])
    reveal = capacity.Reveal(levels[-1][0], 1, {0: (bad, [])})
    assert verdict(HOTKEY, 1, root, reveal) == "malformed_vector"


def test_spare_proof_hashes_do_not_pass():
    raws, levels, tree_root, root = committed(HOTKEY, 8)
    reveal = reveal_of(raws, levels, tree_root, 8)
    items = {i: (raw, [*proof, b"\0" * 32]) for i, (raw, proof) in reveal.items.items()}
    assert verdict(HOTKEY, 8, root, capacity.Reveal(tree_root, 8, items)) == "bad_proof"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"nonsense",
        b"SN22R1" + b"\0" * 32,
        b"SN22R1" + b"\0" * 32 + (1).to_bytes(8, "big") + (5).to_bytes(2, "big"),
        b"x" * (capacity.MAX_REVEAL_BYTES + 1),
    ],
)
def test_a_broken_reveal_file_is_refused(data):
    with pytest.raises(ValueError):
        capacity.decode_reveal(data)


def test_a_reveal_with_bytes_after_it_is_refused():
    raws, levels, tree_root, _ = committed(HOTKEY, 4)
    good = capacity.encode_reveal(tree_root, 4, {0: (raws[0], capacity.tree_proof(levels, 0))})
    assert capacity.decode_reveal(good).count == 4
    with pytest.raises(ValueError):
        capacity.decode_reveal(good + b"\0")


def test_a_real_reveal_at_full_size_fits_the_limit():
    raw = b"\0" * 5120
    items = {index: (raw, [b"\0" * 32] * 20) for index in range(SAMPLES)}
    assert len(capacity.encode_reveal(b"\0" * 32, 1 << 20, items)) < capacity.MAX_REVEAL_BYTES


def test_similarity_is_cosine_whatever_the_reference_length():
    vector = np.array([[0.6, 0.8]], dtype=np.float32)
    assert capacity.similarities(vector, np.array([[3.0, 4.0]]))[0] == pytest.approx(1.0)
    assert capacity.similarities(vector, np.array([[-4.0, 3.0]]))[0] == pytest.approx(0.0, abs=1e-6)
    assert not capacity.matches([1.0, 0.99])
    assert not capacity.matches([])
    assert capacity.median_gap([1.0, 0.99, 0.98]) == pytest.approx(0.01)
