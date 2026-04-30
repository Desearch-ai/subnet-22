import argparse
import os
from distutils.util import strtobool

import bittensor as bt


def str2bool(v):
    return bool(strtobool(v))


def check_config(cls, config: "bt.Config"):
    bt.Axon.check_config(config)
    bt.logging.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/{}".format(
            config.logging.logging_dir,
            config.wallet.get("name", bt.DEFAULTS["wallet"]["name"]),
            config.wallet.get("hotkey", bt.DEFAULTS["wallet"]["hotkey"]),
            config.miner.name,
        )
    )
    config.miner.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.miner.full_path):
        os.makedirs(config.miner.full_path)


def get_config() -> "bt.Config":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axon.port",
        type=int,
        default=int(os.environ.get("AXON_PORT", 8098)),
        help="Port to run the axon on.",
    )
    # External IP
    parser.add_argument(
        "--axon.external_ip",
        type=str,
        default=bt.utils.networking.get_external_ip(),
        help="IP for the metagraph",
    )
    # Subtensor network to connect to
    parser.add_argument(
        "--subtensor.network",
        default=os.environ.get("SUBTENSOR_NETWORK", "finney"),
        help="Bittensor network to connect to.",
    )
    # Chain endpoint to connect to
    parser.add_argument(
        "--subtensor.chain_endpoint",
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="Chain endpoint to connect to.",
    )
    # Adds override arguments for network and netuid.
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.environ.get("NETUID", 22)),
        help="The chain subnet uid.",
    )

    parser.add_argument(
        "--miner.root",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.bittensor/miners/",
    )
    parser.add_argument(
        "--miner.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="Bittensor Miner",
    )

    parser.add_argument(
        "--miner.config_path",
        type=str,
        help="Path to miner manifest JSON (per-search-type concurrency).",
        default="./neurons/miners/manifest.json",
    )

    # Run config.
    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=str,
        help="Blocks until the miner sets weights on chain",
        default=100,
    )

    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.Subtensor.add_args(parser)

    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)

    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.Wallet.add_args(parser)

    # Adds axon specific arguments i.e. --axon.port ...
    bt.Axon.add_args(parser)

    # Override wallet defaults from .env (CLI flags still win).
    parser.set_defaults(
        **{
            "wallet.name": os.environ.get("WALLET_NAME", "miner"),
            "wallet.hotkey": os.environ.get("WALLET_HOTKEY", "default"),
        }
    )

    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 desearch/miner.py --help
    config = bt.Config(parser)

    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config
