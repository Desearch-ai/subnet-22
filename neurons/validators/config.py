import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from desearch.trust import TASK_API_ANCHOR_HOTKEY, TASK_API_SIGNER

ENV_FILE = Path(__file__).resolve().parent / ".env"
REPO = Path(__file__).resolve().parents[2]
# Where `run.sh` builds the task API's replay tool from this release's source.
REPLAY_BIN = REPO / "task-api" / "target" / "release" / "task-api-replay"
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
        default="https://api-22.desearch.ai",
    )
    parser.add_argument(
        "--neuron.task_api_anchor",
        type=str,
        help="The hotkey whose commitment on the subnet anchors the task API's log.",
        default=TASK_API_ANCHOR_HOTKEY,
    )
    parser.add_argument(
        "--neuron.replay_bin",
        type=str,
        help="The task API's replay tool, which run.sh builds from this release (task-api/target/release/task-api-replay);"
        " a validator that sets weights does not start without it.",
        default=os.environ.get("TASK_API_REPLAY_BIN", str(REPLAY_BIN)),
    )
    parser.add_argument(
        "--neuron.no_replay",
        action="store_true",
        help="Run without replaying the task API's log: the API's budgets, verdicts, shares and upload log then never reach the"
        " weights, which come from this validator's own checks only.",
        default=False,
    )
    parser.add_argument(
        "--neuron.storage_url",
        type=str,
        help="Public URL of the bucket uploads land in; the open list and the uploads are read from it.",
        default="https://subnet-22.s3.us-east-va.io.cloud.ovh.us",
    )
    parser.add_argument(
        "--relay.port",
        type=int,
        default=0,
        help="Serve a relay for miners' task API requests on this port, witnessing each one; 0 serves none.",
    )
    parser.add_argument(
        "--relay.host", default="0.0.0.0", help="The address the relay listens on."
    )
    parser.add_argument(
        "--relay.publish_delay_blocks",
        type=int,
        default=5,
        help="Blocks before a relayed request's witness record is listed publicly.",
    )
    parser.add_argument(
        "--relay.check_after_blocks",
        type=int,
        default=50,
        help="Blocks after which a witnessed request whose outcome the task API's anchored log lacks or contradicts is a censorship proof.",
    )
    parser.add_argument(
        "--relay.late_blocks",
        type=int,
        default=10,
        help="Blocks after it was forwarded within which the task API must have judged a witnessed request; later is a censorship proof.",
    )
    parser.add_argument(
        "--relay.signer",
        default="",
        help="The ss58 key the task API's receipts must be signed with; the custody chain's holder (or the pinned signer) when empty.",
    )
    parser.add_argument(
        "--witness.storage_off",
        action="store_false",
        dest="witness_storage",
        help="Do not look at every completed upload in storage.",
    )
    parser.set_defaults(witness_storage=True)
    parser.add_argument(
        "--witness.probe_every_s",
        type=float,
        default=60.0,
        dest="witness_probe_every_s",
        help="Seconds between this validator's own probes of storage through a link the task API hands validators; failures are reported to it. 0 probes nothing.",
    )
    parser.add_argument(
        "--neuron.archive_network",
        type=str,
        help="A chain endpoint holding old state, for approvals at custody records' blocks; the subtensor's own archive fallback when empty.",
        default="",
    )
    parser.add_argument(
        "--neuron.legacy_signer",
        "--neuron.task_api_signer",
        dest="neuron.legacy_signer",
        type=str,
        help="The task API signer from before custody records, pinned in this release and never read from the API; "
        "the only one that may hand over first.",
        default=TASK_API_SIGNER,
    )
    parser.add_argument(
        "--neuron.tee_allow_any_os",
        action="store_true",
        help="Testnets only: accept a task API attestation from any OS image. Ignored on finney.",
        default=False,
    )
    parser.add_argument(
        "--neuron.tee_simulated",
        action="store_true",
        help="Testnets only: accept the dstack simulator's quotes, whose signature cannot verify. Ignored on finney.",
        default=False,
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
    known, unknown = build_parser().parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown arguments: {' '.join(unknown)}", file=sys.stderr)
    made = nested(vars(known))
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
