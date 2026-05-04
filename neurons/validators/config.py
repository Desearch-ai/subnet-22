# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os

import bittensor as bt

from desearch.protocol import ScoringModel


def check_config(config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    if config.mock:
        config.neuron.mock_dataset = False
        config.wallet._mock = True

    config.neuron.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/validator".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
        )
    )
    os.makedirs(config.neuron.full_path, exist_ok=True)


def add_args(cls, parser):
    parser.add_argument(
        "--netuid", type=int, help="Prompting network netuid", default=22
    )

    parser.add_argument("--wandb.off", action="store_false", dest="wandb_on")

    parser.set_defaults(wandb_on=True)

    parser.add_argument(
        "--neuron.disable_log_rewards",
        action="store_true",
        help="Disable all reward logging, suppresses reward functions and their values from being logged to wandb.",
        default=False,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.offline",
        action="store_true",
        help="Run validator in offline mode",
        default=False,
    )

    parser.add_argument(
        "--neuron.scoring_model",
        type=ScoringModel,
        help="Name of llm model used for scoring.",
        default=ScoringModel.OPENAI_GPT4_1_NANO,
    )

    parser.add_argument(
        "--neuron.utility_api_url",
        type=str,
        help="Base URL of the utility API that provides scoring questions.",
        default="https://utility-api.desearch.ai",
    )


def config(cls):
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.AsyncSubtensor.add_args(parser)

    os.environ["BT_LOGGING_DEBUG"] = "True"
    bt.logging.add_args(parser)

    bt.Axon.add_args(parser)
    cls.add_args(parser)
    return bt.Config(parser)
