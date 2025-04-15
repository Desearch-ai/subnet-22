from abc import ABC, abstractmethod
import asyncio
import itertools
import torch
import bittensor as bt
import argparse


class AbstractNeuron(ABC):
    @abstractmethod
    def __init__(self):
        self.subtensor: "bt.subtensor" = None
        self.wallet: "bt.wallet" = None
        self.metagraph: "bt.metagraph" = None
        self.dendrite: "bt.dendrite" = None
        self.dendrite1: "bt.dendrite" = None
        self.dendrite2: "bt.dendrite" = None
        self.dendrite3: "bt.dendrite" = None
        self.dendrites: itertools.cycle[bt.dendrite]

    @classmethod
    @abstractmethod
    def check_config(cls, config: "bt.config"):
        pass

    @classmethod
    @abstractmethod
    def add_args(cls, parser: "argparse.ArgumentParser"):
        pass

    @classmethod
    @abstractmethod
    def config(cls) -> "bt.config":
        pass

    @abstractmethod
    def initialize_components(self):
        pass

    @abstractmethod
    async def check_uid(self, axon, uid: int):
        pass

    @abstractmethod
    async def get_uids(self, axon, uid: int):
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
    async def query_synapse(self):
        pass

    @abstractmethod
    def run(self):
        pass
