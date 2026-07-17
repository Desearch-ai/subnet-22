import bittensor as bt
from bittensor_wallet import Keypair

MOCK_WALLET_KEY = "5EsrMfo7UcPs6AqAotU47VmYGfLHntS9JzhEwbY2EJMcWQxQ"


class Wallet(bt.Wallet):
    @property
    def hotkey(self):
        return Keypair(MOCK_WALLET_KEY)

    pass
