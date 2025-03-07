<div align="center">

# **Desearch (Subnet 22) on Bittensor**
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

## Introduction

**Bittensor Desearch (Subnet 22):** 

Welcome to Desearch, the AI-powered search engine built on Bittensor. Designed for the Bittensor community and general internet users, Desearch delivers an unbiased and verifiable search experience. Through our API, developers and AI builders are empowered to integrate AI search capabilities into their products, with access to metadata from platforms like X, Reddit, Arxiv and general web search.


### Key Features

- **AI-powered Analysis:** Utilizes decentralized AI models to deliver relevant, contextual, and unfiltered search results.
- **Real-time Access to Diverse Data Sources:** Access metadata from platforms like X, Reddit, Arxiv, and broader web data.
- **Sentiment and Metadata Analysis:** Determines the emotional tone of social posts while analyzing key metadata to provide a comprehensive understanding of public sentiment.
- **Time-efficient:** Minimizes manual data sorting, saving valuable research time.
- **User-friendly Design:** Suitable for both beginners and experts.

### Advantages

- **Decentralized Platform:** Built on the Bittensor network, ensures unbiased and highly relevant search results through decentralization.
- **Customizability:** Tailors data analysis to meet specific user requirements.
- **Versatility:** Applicable for diverse research fields, from market analysis to academic studies.
- **Community-driven Innovation:** Built and optimized by a decentralized network of Bittensor miners, validators, and users for continuous search result enhancement.

---

## Installation

**Requirements:** Python 3.10 or higher

1. Clone the repository:
   ```bash
   git clone https://github.com/Datura-ai/desearch.git
   ```
2. Install the requirements:
   ```bash
   cd desearch
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

---

## Preparing Your Environment

Before running a miner or validator, ensure to:

- [Create a wallet](https://github.com/opentensor/docs/blob/main/reference/btcli.md).
- [Register the wallet to a netuid](https://github.com/opentensor/docs/blob/main/subnetworks/registration.md).


### Environment Variables Configuration

For setting up the necessary environment variables for your miner or validator, please refer to the [Environment Variables Guide](./docs/env_variables.md).

# Running the Miner

  ```bash
  python -m neurons/miners/miner.py 
      --netuid 22
      --subtensor.network finney
      --wallet.name <your miner wallet>
      --wallet.hotkey <your validator hotkey>
      --axon.port 14000
  ```

# Running the Validator API with Automatic Updates

These validators are designed to run and update themselves automatically. To run a validator, follow these steps:

1. Install this repository, you can do so by following the steps outlined in [the installation section](#installation).
2. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and run `wandb login` within this repository. This will initialize Weights and Biases, enabling you to view KPIs and Metrics on your validator. (Strongly recommended to help the network improve from data sharing)
3. Install [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/).
4. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```
5. Run the `run.sh` script which will handle running your validator and pulling the latest updates as they are issued. 
   ```bash
   pm2 start run.sh --name smart_scrape_validators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
   ```

This will run **two** PM2 process: one for the validator which is called `smart_scrape_validators_main_process` by default (you can change this in `run.sh`), and one for the run.sh script (in step 4, we named it `smart_scrape_validators_autoupdate`). The script will check for updates every 30 minutes, if there is an update then it will pull it, install it, restart `smart_scrape_validators_main_process` and then restart itself.

### Detailed Setup Instructions

For step-by-step guidance on setting up and running a miner, validator, or operating on the testnet or mainnet, refer to the following guides:
- [Miner Setup](./docs/running_a_miner.md)
- [Validator Setup](./docs/running_a_validator.md)
- [Testnet Operations](./docs/running_on_testnet.md)
- [Mainnet Operations](./docs/running_on_mainnet.md)
- [Setting Up and Running the Web Application with Validator Integration](./ui/README.md)

---


## Real-time Monitoring with wandb Integration

The text prompting validator sends data to wandb, allowing real-time monitoring with key metrics like:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

Data is publicly available at [this link](https://wandb.ai/smart-scrape/smart-scrape-1.0). Note that [data from anonymous users is deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon).

</div>