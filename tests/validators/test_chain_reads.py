from desearch.manifest import seed_from_hash
from neurons.validators import chain_reads
from neurons.validators.validator import Validator


class PrunedNode:
    """A default node: every finalized block's hash and body, no state older than the last 256 blocks."""

    def __init__(self, finalized: int = 1_000):
        self.finalized = finalized
        self.asked = []

    async def rpc(self, method, params):
        self.asked.append(method)
        if method == "chain_getFinalizedHead":
            return "0xhead"
        if method == "chain_getHeader":
            return {"number": hex(self.finalized)}
        if method == "chain_getBlockHash":
            return f"0x{params[0]:064x}"
        if method == "chain_getBlock":
            return {"block": {"extrinsics": ["0x0400", "0x0800"]}}
        raise RuntimeError("State discarded for 0x...")


async def test_blocks_are_read_by_number_which_a_pruned_node_answers_for_any_block():
    node = PrunedNode()
    assert await chain_reads.block_hash(node, 10) == f"0x{10:064x}"
    assert await chain_reads.block_hash(node, 1_001) is None, "not finalized yet"
    assert await chain_reads.block_extrinsics(node, 10) == [b"\x04\x00", b"\x08\x00"]
    assert not any(method.startswith("state_") for method in node.asked)

    # The seed of an upload whose block is long past, on a validator whose node keeps no state that old.
    validator = Validator.__new__(Validator)
    validator.subtensor = node
    assert await validator.seed_for(10) == seed_from_hash(f"0x{10:064x}")
    assert await validator.seed_for(1_001) is None
