import bittensor as bt
from bittensor.core.metagraph import AsyncMetagraph
import torch
from bittensor.core.chain_data import (
    AxonInfo,
    NeuronInfoLite,
    PrometheusInfo,
)
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.utils.balance import Balance
from .wallet import MOCK_WALLET_KEY

NEURON_COUNT = 3


def generateMockNeurons(count: int = 2):
    neurons = [
        NeuronInfoLite(
            hotkey=MOCK_WALLET_KEY,
            coldkey="coldkey0",
            uid=0,
            netuid=41,
            active=True,
            stake=Balance(0),
            axon_info=AxonInfo(
                version=8005001,
                ip="0.0.0.0",
                hotkey=MOCK_WALLET_KEY,
                coldkey="coldkey0",
                port=8091,
                ip_type=4,
            ),
            stake_dict={"coldkey0": Balance(0)},
            total_stake=Balance(0),
            rank=0.0,
            emission=0.0,
            incentive=0.0,
            consensus=0.0,
            trust=0.0,
            validator_trust=0.0,
            dividends=0.0,
            last_update=3764741,
            validator_permit=True,
            prometheus_info=PrometheusInfo(
                block=0, version=0, ip="0.0.0.0", port=0, ip_type=0
            ),
            pruning_score=0,
            is_null=False,
        ),
    ]

    neurons.extend(
        [
            NeuronInfoLite(
                hotkey=f"hotkey{i}",
                coldkey=f"coldkey{i}",
                uid=i,
                netuid=41,
                active=True,
                stake=Balance(0),
                axon_info=AxonInfo(
                    version=8005001,
                    ip="0.0.0.0",
                    hotkey=f"hotkey{i}",
                    coldkey=f"coldkey{i}",
                    port=8091,
                    ip_type=4,
                ),
                stake_dict={f"coldkey{i}": Balance(0)},
                total_stake=Balance(0),
                rank=0.0,
                emission=0.0,
                incentive=0.0,
                consensus=0.0,
                trust=0.0,
                validator_trust=0.0,
                dividends=0.0,
                last_update=3764741,
                validator_permit=True,
                prometheus_info=PrometheusInfo(
                    block=0, version=0, ip="0.0.0.0", port=0, ip_type=0
                ),
                pruning_score=0,
                is_null=False,
            )
            for i in range(1, count)
        ]
    )

    return neurons


class Metagraph(AsyncMetagraph):
    def __init__(
        self,
        netuid: int,
        network: str = "local",
        lite: bool = True,
        sync: bool = True,
        subtensor=None,
    ):
        super().__init__(
            netuid=netuid, network=network, lite=lite, sync=sync, subtensor=subtensor
        )
        self.neurons = generateMockNeurons(NEURON_COUNT)

        self.lite = lite
        self.netuid = netuid
        self.network = network

    async def sync(self, subtensor, block=None, lite=False):
        self.lite = lite

        self.n = self._create_tensor(len(self.neurons), dtype=torch.int64)
        self.version = self._create_tensor([1], dtype=torch.int64)
        self.block = self._create_tensor(
            block if block else (await subtensor.block), dtype=torch.int64
        )
        self.uids = self._create_tensor(
            [neuron.uid for neuron in self.neurons], dtype=torch.int64
        )
        self.trust = self._create_tensor(
            [neuron.trust for neuron in self.neurons], dtype=torch.float32
        )
        self.consensus = self._create_tensor(
            [neuron.consensus for neuron in self.neurons], dtype=torch.float32
        )
        self.incentive = self._create_tensor(
            [neuron.incentive for neuron in self.neurons], dtype=torch.float32
        )
        self.dividends = self._create_tensor(
            [neuron.dividends for neuron in self.neurons], dtype=torch.float32
        )
        self.ranks = self._create_tensor(
            [neuron.rank for neuron in self.neurons], dtype=torch.float32
        )
        self.emission = self._create_tensor(
            [neuron.emission for neuron in self.neurons], dtype=torch.float32
        )
        self.active = self._create_tensor(
            [neuron.active for neuron in self.neurons], dtype=torch.int64
        )
        self.last_update = self._create_tensor(
            [neuron.last_update for neuron in self.neurons], dtype=torch.int64
        )
        self.validator_permit = self._create_tensor(
            [neuron.validator_permit for neuron in self.neurons], dtype=torch.bool
        )
        self.validator_trust = self._create_tensor(
            [neuron.validator_trust for neuron in self.neurons], dtype=torch.float32
        )
        self.total_stake = self._create_tensor(
            [neuron.total_stake.tao for neuron in self.neurons], dtype=torch.float32
        )
        self.stake = self._create_tensor(
            [neuron.stake for neuron in self.neurons], dtype=torch.float32
        )
        self.axons = [n.axon_info for n in self.neurons]
