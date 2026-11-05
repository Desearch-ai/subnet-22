"""What a validator takes as the task API's signer and anchor: only what its release pins or the custody chain names."""

import asyncio
from types import SimpleNamespace

from neurons.validators import validator as module
from neurons.validators.validator import Validator

SIGNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
ANCHOR = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def validator(legacy_signer: str = "", anchor: str = "") -> Validator:
    made = Validator.__new__(Validator)
    neuron = SimpleNamespace(
        legacy_signer=legacy_signer,
        task_api_anchor=anchor,
        task_api_url="https://api.example",
        storage_url="https://storage.example",
        full_path="/tmp/unused",
        tee_allow_any_os=False,
        tee_simulated=False,
    )
    made.config = SimpleNamespace(
        neuron=neuron, netuid=22, subtensor=SimpleNamespace(network="finney")
    )
    return made


def test_anchors_are_never_read_from_the_signers_hotkey(monkeypatch):
    monkeypatch.setattr(module, "LEGACY_SIGNER", SIGNER)
    assert validator(anchor=ANCHOR).anchor_hotkey() == ANCHOR
    assert validator(anchor=SIGNER).anchor_hotkey() == "", "the release's signer"
    assert validator(legacy_signer=ANCHOR, anchor=ANCHOR).anchor_hotkey() == ""
    assert validator().anchor_hotkey() == "", "none pinned: anchors are not checked"


def test_without_a_pinned_signer_the_key_the_api_names_is_never_taken(monkeypatch):
    class Unpinned:
        def __init__(self, *args, fallback="", **kwargs):
            self.records, self.fallback = [], fallback

        async def poll(self):
            return False

    class NoHttp:
        def get(self, *args, **kwargs):
            raise AssertionError("asked the task API which key to follow")

    monkeypatch.setattr(module, "Custody", Unpinned)
    monkeypatch.setattr(module, "LEGACY_SIGNER", "")
    made = validator()
    made.http, made.archive = NoHttp(), None
    custody = asyncio.run(made.follow_custody())
    assert custody.fallback == "" and not hasattr(made, "key_from_api")
