import argparse
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import bittensor as bt
import torch
from bittensor.core.metagraph import AsyncMetagraph


class AbstractNeuron(ABC):
    @abstractmethod
    def __init__(self):
        self.subtensor: "bt.AsyncSubtensor" = None
        self.wallet: "bt.Wallet" = None
        self.metagraph: "AsyncMetagraph" = None
        self.dendrite: "bt.Dendrite" = None
        self.dendrite1: "bt.Dendrite" = None
        self.dendrite2: "bt.Dendrite" = None
        self.dendrite3: "bt.Dendrite" = None
        self.dendrites: itertools.cycle[bt.Dendrite]

    @classmethod
    @abstractmethod
    def add_args(cls, parser: "argparse.ArgumentParser"):
        pass

    @classmethod
    @abstractmethod
    def config(cls) -> "bt.config":
        pass

    @abstractmethod
    async def initialize_components(self):
        pass

    @abstractmethod
    async def check_uid(self, axon, uid: int):
        pass

    @abstractmethod
    async def get_uids(self, axon, uid: int):
        pass

    @abstractmethod
    async def get_random_miner(
        self, uid: Optional[int] = None
    ) -> Tuple[int, bt.AxonInfo]:
        pass

    @abstractmethod
    async def update_scores(self, scores: torch.Tensor, wandb_data):
        pass

    @abstractmethod
    async def update_scores_for_basic(self, scores: torch.Tensor, wandb_data):
        pass

    @abstractmethod
    async def update_moving_averaged_scores(self, uids, rewards):
        pass

    @abstractmethod
    async def run_query_and_score(self):
        pass

    @abstractmethod
    def run(self):
        pass
