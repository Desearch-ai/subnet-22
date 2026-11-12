"""The validator. Importing any of it first makes the pinned SDK read an axon it cannot parse as no axon (below)."""

import bittensor.metagraph as _sdk

# bittensor 11.1.0 parses every neuron's axon in `metagraph._axon_endpoint` while building the metagraph, so one legacy record
# with `ip_type` 4 and an ip past 2**32 (testnet netuid 41 has one) raises `AddressValueError` and no metagraph can be read at
# all. That neuron serves no axon instead. tests/validators/test_sdk_axon.py fails if the SDK moves or renames the function.
_sdk_axon_endpoint = getattr(_sdk, "_axon_endpoint", None)


def _axon_endpoint(axon):
    try:
        return _sdk_axon_endpoint(axon)
    except ValueError:
        return None


if _sdk_axon_endpoint is not None:
    _sdk._axon_endpoint = _axon_endpoint
