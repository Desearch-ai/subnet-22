"""The capacity test: what a test is, its texts, what a miner commits and reveals, and how a reveal is judged.

Everyone sits a test at once, so a GPU shared by several hotkeys counts once. Nothing here needs the task API: a test follows
from block hashes alone, and anyone can rebuild its texts and judge a reveal. Everything validators must agree on works on bytes.
docs/capacity-test.md explains it for miners.
"""

from __future__ import annotations

import hashlib
import re
import statistics
import struct
from dataclasses import dataclass

import numpy as np

from desearch.embedding import decode_vector

# A block is a candidate one time in this many, and a test when no candidate came in the blocks before it: about 24 tests a day.
CANDIDATE_ONE_IN = 240
QUIET_BLOCKS = 60
# Commitments count up to the deadline block: the test block plus this.
WINDOW_BLOCKS = 6
# The reference miner stops this many blocks before the deadline, leaving a block to send its commitment again.
REFERENCE_STOP_BLOCKS = 2
SECONDS_PER_BLOCK = 12
# The block whose hash picks the samples comes this long after the deadline, so no commitment can follow from it.
SAMPLE_AFTER_BLOCKS = 4
REVEAL_BLOCKS = 25
SAMPLES = 48
MATCH_SIMILARITY = 0.999
# Honest vectors sit about 1e-6 from the reference. Reported, not enforced, until real GPUs show where a tighter rule can stand.
TIGHT_MEDIAN_GAP = 1e-5
NORM_TOLERANCE = 0.01
# This share of the coldkeys that committed having no reveal means the fault was not theirs, and the test counts for nobody.
VOID_MISSING_SHARE = 0.2

DOMAIN = b"sn22-capacity-1:"
TAG = b"sn22c1:"
TAG_PATTERN = re.compile(
    rb"\Asn22c1:([1-9][0-9]{0,11}):([0-9a-f]{8}):(0|[1-9][0-9]{0,11})\Z"
)
REVEAL_MAGIC = b"SN22R1"
MAX_PROOF = 48
MAX_REVEAL_BYTES = 600_000

# Lengths of real embed texts in bytes, up to the longest a test uses, one for each 64th of them.
LENGTHS = (
    157, 249, 298, 335, 370, 400, 431, 464, 497, 533, 571, 609, 644, 666, 683, 696,
    707, 719, 731, 745, 760, 776, 792, 809, 829, 849, 871, 895, 920, 949, 978, 1014,
    1056, 1104, 1156, 1209, 1270, 1342, 1426, 1508, 1582, 1666, 1785, 1902, 2043, 2181, 2336, 2506,
    2672, 2863, 3051, 3246, 3467, 3707, 3955, 4214, 4487, 4846, 5264, 5751, 6000, 6000, 6000, 6000,
)  # fmt: skip

WORDS = tuple("""
    able about above accept across action active actor actual adapt add address adult advice affect after again against
    age agency agent agree ahead air alarm album alert allow almost alone along already also always among amount ancient
    angle animal annual answer any apart apple apply area argue arm army around arrive art article artist ask aspect assume
    attack attend author autumn avoid award aware away baby back badge bag balance ball band bank barrier base basic basket
    battle beach bear beauty because become bed before begin behind believe bell belong below bench benefit best better
    between beyond bicycle big bill bird birth bit black blade blank block blood blue board boat body bone book border both
    bottle bottom box boy brain branch brave bread break bridge brief bright bring broad brother brown budget build burn bus
    business button buy cable cake call calm camera camp can capital captain car card care carry case cash cast castle cat
    catch cause cell center central century chain chair chance change chapter charge chart check chief child choice choose
    church circle city civil claim class clean clear climb clock close cloud club coach coast code coffee cold collect color
    come common company compare complex concern confirm connect consider contain content context control cook copy corner
    cost cotton count country couple course court cover craft create credit crew crop cross crowd culture cup current
    custom cut cycle daily damage dance danger dark data date daughter day deal debate decade decide deep degree delay
    demand depend depth desert design desk detail develop device differ dinner direct discuss doctor dog door double down
    draft drama draw dream dress drink drive drop dry during duty each early earth east easy edge editor effect effort eight
    either elder electric element else empty end energy engine enjoy enough enter entire equal error escape even evening
    event ever every exact example exist expect expert explain express extend extra eye face fact factor fair fall family
    famous far farm fast father favor fear feature feed feel few field fifty figure file fill film final find fine finger
    finish fire firm first fish five fix flag flat flight floor flow flower fly focus follow food foot force foreign forest
    forget form former forward four frame free fresh friend front fruit full fund future game garden gas gate gather general
    gentle gift girl give glad glass global goal gold good grain grand grass great green ground group grow guard guess guide
    habit hair half hall hand happen happy hard harvest hat head health hear heart heat heavy help here high hill history
    hold hole home hope horse hospital hot hotel hour house human hundred idea image impact import improve inch include
    income index inside invest iron island issue item job join joy judge jump just keep key kind king kitchen knee know
    label lady lake land language large last late laugh law layer lead learn least leave left legal lemon length lesson
    letter level library life lift light like limit line link list listen little live local lock long look lose lot love
    lunch machine magic
    """.split())
assert len(WORDS) == 512 and len(set(WORDS)) == 512


@dataclass(frozen=True)
class Test:
    block: int
    seed: bytes

    @property
    def deadline(self) -> int:
        return self.block + WINDOW_BLOCKS

    @property
    def sample_block(self) -> int:
        return self.deadline + SAMPLE_AFTER_BLOCKS

    @property
    def reveal_by(self) -> int:
        return self.sample_block + REVEAL_BLOCKS


@dataclass(frozen=True)
class Reveal:
    tree_root: bytes
    count: int
    items: dict[int, tuple[bytes, list[bytes]]]


def _digest(tag: bytes, *parts: bytes) -> bytes:
    hasher = hashlib.sha256(DOMAIN + tag)
    for part in parts:
        hasher.update(len(part).to_bytes(4, "big"))
        hasher.update(part)
    return hasher.digest()


def is_candidate(block_hash: bytes) -> bool:
    return (
        int.from_bytes(_digest(b"candidate", block_hash)[:8], "big") % CANDIDATE_ONE_IN
        == 0
    )


def test_at(block: int, block_hash: bytes, earlier_hashes: list[bytes]) -> Test | None:
    """The test this block starts, given the hashes of the QUIET_BLOCKS before it."""
    if not is_candidate(block_hash) or any(map(is_candidate, earlier_hashes)):
        return None
    return Test(block, block_hash)


def text_length(seed: bytes, hotkey: str, index: int) -> int:
    return LENGTHS[
        _digest(b"length", seed, hotkey.encode(), index.to_bytes(8, "big"))[0]
        % len(LENGTHS)
    ]


def text_at(seed: bytes, hotkey: str, index: int) -> str:
    """A hotkey's text at an index: words drawn from hashes, so it cannot be known before the test or reused from another text."""
    length = text_length(seed, hotkey, index)
    draw = hashlib.shake_256(
        DOMAIN + b"text" + seed + hotkey.encode() + index.to_bytes(8, "big")
    )
    # Each word with its space takes at least 3 of the length.
    drawn = draw.digest(2 * (length // 3 + 2))
    parts: list[str] = []
    size = 0
    for at in range(0, len(drawn), 2):
        word = WORDS[drawn[at] | (drawn[at + 1] & 1) << 8]
        # The top bits end a sentence about one word in sixteen.
        parts.append(f"{word}. " if drawn[at + 1] >> 4 == 0 else f"{word} ")
        size += len(parts[-1])
        if size >= length:
            break
    return "".join(parts)[:length]


def chars_of(seed: bytes, hotkey: str, count: int) -> int:
    """The characters of texts 0..count-1; the texts are ASCII."""
    return sum(text_length(seed, hotkey, index) for index in range(count))


def leaf_hash(index: int, raw: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + index.to_bytes(8, "big") + raw).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def tree_levels(leaves: list[bytes]) -> list[list[bytes]]:
    """The tree from its leaves up; a last node without a sibling moves up as it is."""
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        level = levels[-1]
        parents = [
            _node(level[at], level[at + 1]) for at in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            parents.append(level[-1])
        levels.append(parents)
    return levels


def tree_proof(levels: list[list[bytes]], index: int) -> list[bytes]:
    proof = []
    for level in levels[:-1]:
        if index % 2:
            proof.append(level[index - 1])
        elif index + 1 < len(level):
            proof.append(level[index + 1])
        index //= 2
    return proof


def proof_holds(
    leaf: bytes, index: int, count: int, proof: list[bytes], root: bytes
) -> bool:
    """The proof's shape follows from index and count, so no extra or missing hash passes."""
    if not 0 <= index < count:
        return False
    node, used = leaf, 0
    while count > 1:
        if index % 2 or index + 1 < count:
            if used == len(proof):
                return False
            node = _node(proof[used], node) if index % 2 else _node(node, proof[used])
            used += 1
        index //= 2
        count = (count + 1) // 2
    return used == len(proof) and node == root


def commit_root(seed: bytes, hotkey: str, count: int, tree_root: bytes) -> bytes:
    """Binds the tree to its test, hotkey and count, so a commitment means nothing for another hotkey."""
    return _digest(b"root", seed, hotkey.encode(), count.to_bytes(8, "big"), tree_root)


def make_tag(test: Test, count: int) -> bytes:
    return TAG + b"%d:%s:%d" % (test.block, test.seed[:4].hex().encode(), count)


def parse_tag(tag: bytes) -> tuple[int, bytes, int] | None:
    """The test block, seed prefix and count of a commitment, or None when it is not one."""
    found = TAG_PATTERN.match(tag)
    if not found:
        return None
    return int(found[1]), bytes.fromhex(found[2].decode()), int(found[3])


def sample_indices(
    sample_hash: bytes, hotkey: str, count: int, samples: int = SAMPLES
) -> list[int]:
    """Which texts a hotkey must reveal, from a block hash nobody knew when it committed."""
    if count <= samples:
        return list(range(count))
    picked: list[int] = []
    taken: set[int] = set()
    draw = 0
    while len(picked) < samples:
        digest = _digest(
            b"sample", sample_hash, hotkey.encode(), draw.to_bytes(8, "big")
        )
        index = int.from_bytes(digest, "big") % count
        draw += 1
        if index not in taken:
            taken.add(index)
            picked.append(index)
    return picked


def reveal_key(test_block: int, hotkey: str) -> str:
    """Where a reveal sits in the public bucket; the task API's reveal route uses the same key."""
    return f"capacity/{test_block}/{hotkey}.reveal"


def encode_reveal(
    tree_root: bytes, count: int, items: dict[int, tuple[bytes, list[bytes]]]
) -> bytes:
    parts = [REVEAL_MAGIC, tree_root, struct.pack(">QH", count, len(items))]
    for index, (raw, proof) in sorted(items.items()):
        parts += [struct.pack(">QIB", index, len(raw), len(proof)), raw, *proof]
    return b"".join(parts)


def decode_reveal(data: bytes) -> Reveal:
    """The reveal in a file; ValueError when the file is not one."""
    if len(data) > MAX_REVEAL_BYTES:
        raise ValueError(f"{len(data)} bytes, over {MAX_REVEAL_BYTES}")
    head = len(REVEAL_MAGIC) + 32 + 10
    if len(data) < head or not data.startswith(REVEAL_MAGIC):
        raise ValueError("not a reveal")
    tree_root = data[len(REVEAL_MAGIC) : len(REVEAL_MAGIC) + 32]
    count, entries = struct.unpack(">QH", data[len(REVEAL_MAGIC) + 32 : head])
    items: dict[int, tuple[bytes, list[bytes]]] = {}
    at = head
    for _ in range(entries):
        if at + 13 > len(data):
            raise ValueError("a reveal cut short")
        index, size, proofs = struct.unpack(">QIB", data[at : at + 13])
        at += 13
        end = at + size + 32 * proofs
        if proofs > MAX_PROOF or end > len(data) or index in items:
            raise ValueError("a malformed reveal")
        raw = data[at : at + size]
        proof = [
            data[at + size + 32 * n : at + size + 32 * (n + 1)] for n in range(proofs)
        ]
        items[index] = (raw, proof)
        at = end
    if at != len(data):
        raise ValueError("bytes after a reveal")
    return Reveal(tree_root, count, items)


def unit_vector(raw: bytes, dims: int) -> np.ndarray | None:
    """A revealed vector, or None when it is not finite and of length 1."""
    vector = decode_vector(raw, dims)
    if (
        vector is None
        or not np.all(np.isfinite(vector))
        or abs(float(np.linalg.norm(vector)) - 1.0) > NORM_TOLERANCE
    ):
        return None
    return vector


def opened_vectors(
    seed: bytes,
    hotkey: str,
    count: int,
    root: bytes,
    sample_hash: bytes,
    reveal: Reveal,
    dims: int,
) -> tuple[list[int], np.ndarray] | str:
    """The sampled texts and their vectors when the reveal proves them against the commitment, else why it does not."""
    if (
        reveal.count != count
        or commit_root(seed, hotkey, count, reveal.tree_root) != root
    ):
        return "wrong_root"
    picked = sample_indices(sample_hash, hotkey, count)
    vectors = []
    for index in picked:
        if index not in reveal.items:
            return "sample_missing"
        raw, proof = reveal.items[index]
        if not proof_holds(
            leaf_hash(index, raw), index, count, proof, reveal.tree_root
        ):
            return "bad_proof"
        vector = unit_vector(raw, dims)
        if vector is None:
            return "malformed_vector"
        vectors.append(vector)
    return picked, np.array(vectors, dtype=np.float32).reshape(len(picked), dims)


def similarities(vectors: np.ndarray, reference: np.ndarray) -> list[float]:
    """Cosine similarity of each revealed vector with the validator's own."""
    norms = np.linalg.norm(reference, axis=1)
    unit = reference / np.maximum(norms, 1e-12)[:, None]
    return [
        float(value)
        for value in np.sum(vectors * unit, axis=1)
        / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
    ]


def matches(values: list[float]) -> bool:
    return bool(values) and min(values) >= MATCH_SIMILARITY


def median_gap(values: list[float]) -> float:
    """How far the median similarity is from 1, kept in shadow mode to set a tighter rule from real data."""
    return 1.0 - statistics.median(values) if values else 0.0
