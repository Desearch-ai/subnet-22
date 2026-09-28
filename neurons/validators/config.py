import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_LOG_DIR = "~/.bittensor/miners"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=22)
    parser.add_argument("--wallet.name", default="default")
    parser.add_argument("--wallet.hotkey", default="default")
    parser.add_argument("--wallet.path", default="~/.bittensor/wallets")
    parser.add_argument(
        "--subtensor.network",
        default="finney",
        help="finney, test, or a ws:// endpoint",
    )
    parser.add_argument(
        "--subtensor.chain_endpoint", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--logging.logging_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--logging.info", action="store_true")
    parser.add_argument("--logging.debug", action="store_true")
    parser.add_argument("--logging.trace", action="store_true")
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
        help="The task API: verdicts are reported to it and the coverage gate read from it.",
        default="https://task-api.desearch.ai",
    )
    parser.add_argument(
        "--neuron.storage_url",
        type=str,
        help="Public URL of the bucket uploads land in; the open list and the uploads are read from it.",
        default="https://r2.desearch.ai",
    )
    return parser


def nested(flat: dict) -> SimpleNamespace:
    """`wallet.name` style keys become `config.wallet.name`."""
    tree: dict = {}
    for key, value in flat.items():
        node = tree
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    return SimpleNamespace(
        **{
            k: nested_namespace(v) if isinstance(v, dict) else v
            for k, v in tree.items()
        }
    )


def nested_namespace(tree: dict) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            k: nested_namespace(v) if isinstance(v, dict) else v
            for k, v in tree.items()
        }
    )


def flat(config: SimpleNamespace, prefix: str = "") -> dict:
    out = {}
    for key, value in vars(config).items():
        name = f"{prefix}{key}"
        if isinstance(value, SimpleNamespace):
            out.update(flat(value, name + "."))
        else:
            out[name] = value
    return out


def config(argv: list[str] | None = None) -> SimpleNamespace:
    made = nested(vars(build_parser().parse_args(argv)))
    made.neuron.full_path = os.path.expanduser(
        f"{made.logging.logging_dir}/{made.wallet.name}/{made.wallet.hotkey}"
        f"/netuid{made.netuid}/validator"
    )
    os.makedirs(made.neuron.full_path, exist_ok=True)
    return made


def setup_logging(made: SimpleNamespace) -> None:
    if made.logging.debug or made.logging.trace:
        level = logging.DEBUG
    elif made.logging.info:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("bittensor").setLevel(max(level, logging.INFO))
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
