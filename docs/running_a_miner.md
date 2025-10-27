# Bittensor (Desearch) Miner Setup Guide

This guide details the process for setting up and running a Bittensor Desearch miner using the Desearch repository.

## Prerequisites
Before starting, ensure you have:

- **PM2:** A process manager to maintain your miner. If not installed, see [PM2 Installation](https://pm2.io/docs/runtime/guide/installation/).

- **Environment Variables:** Set the necessary variables as per the [Environment Variables Guide](./env_variables.md).

## Setup Process

## 1. Clone the desearch repository and install dependencies
Clone and install the Desearch repository in editable mode:

```sh
git clone https://github.com/Desearch-ai/subnet-22.git
cd desearch
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Configure and Run the Miner
Configure and launch the miner using PM2:

```sh
pm2 start neurons/miners/miner.py \
--miner.name desearch \
--interpreter <path-to-python-binary> -- \
--wallet.name <wallet-name> \
--wallet.hotkey <wallet-hotkey> \
--netuid <netuid> \
--subtensor.network <network> \
--axon.port <port>

# Example
pm2 start neurons/miners/miner.py --interpreter /usr/bin/python3 --name miner_1 -- --wallet.name miner --wallet.hotkey default --subtensor.network testnet --netuid 41 --axon.port 14001
```

#### Variable Explanation
- `--wallet.name`: Your wallet's name.
- `--wallet.hotkey`: Your wallet's hotkey.
- `--netuid`: Network UID, `41` for testnet.
- `--subtensor.network`: Choose network (`finney`, `test`, `local`, etc).
- `--logging.debug`: Set logging level.
- `--axon.port`: Desired port number.

- `--miner.name`: Path for miner data (miner.root / (wallet_cold - wallet_hot) / miner.name).
- `--miner.mock_dataset`: Set to True to use a mock dataset.
- `--miner.blocks_per_epoch`: Number of blocks until setting weights on chain.
- `--miner.openai_summary_model`: OpenAI model used for summarizing content. Default gpt-3.5-turbo-0125
- `--miner.openai_query_model`: OpenAI model used for generating queries. Default gpt-3.5-turbo-0125
- `--miner.openai_fix_query_model`: "OpenAI model used for fixing queries. Default gpt-4-1106-preview


## Conclusion
Following these steps, your desearch miner should be operational. Regularly monitor your processes and logs for any issues. For additional information or assistance, consult the official documentation or community resources.