import bittensor as bt

from substrateinterface import Keypair

MOCK_WALLET_KEY = "5EsrMfo7UcPs6AqAotU47VmYGfLHntS9JzhEwbY2EJMcWQxQ"


class Wallet(bt.wallet):
    @property
    def hotkey(self):
        return Keypair(MOCK_WALLET_KEY)

    pass
