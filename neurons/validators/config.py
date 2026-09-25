import argparse
import os
from pathlib import Path

import bittensor as bt

ENV_FILE = Path(__file__).resolve().parent / ".env"


def check_config(config: "bt.Config"):
    bt.logging.check_config(config)

    config.neuron.full_path = os.path.expanduser(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    )
    os.makedirs(config.neuron.full_path, exist_ok=True)


def add_args(cls, parser):
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=22)

    parser.add_argument("--wandb.off", action="store_false", dest="wandb_on")

    parser.set_defaults(wandb_on=True)

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.task_api_url",
        type=str,
        help="The crawl task API: validation jobs are claimed from it and weights set from its shares.",
        default="https://task-api.desearch.ai",
    )


def config(cls):
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.AsyncSubtensor.add_args(parser)

    os.environ["BT_LOGGING_DEBUG"] = "True"
    bt.logging.add_args(parser)

    cls.add_args(parser)
    return bt.Config(parser)
