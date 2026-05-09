# Running on Bittensor Mainnet

SN22 runs on Bittensor mainnet (`finney`) with `netuid 22`. This guide covers the mainnet-specific wallet, registration, and launch checks for miners and validators.

Use the role-specific runbooks for full process management details:

- [Running a Miner](./running_a_miner.md)
- [Running a Validator](./running_a_validator.md)
- [Environment Variables](./env_variables.md)

## 1. Install the repo

```bash
git clone https://github.com/Desearch-ai/subnet-22.git
cd subnet-22
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## 2. Create wallets

Create a coldkey/hotkey pair for the role you will run.

```bash
# Miner wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Validator wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

If your installed `btcli` uses the older command names, run `btcli --help` and use the equivalent wallet commands for that version.

## 3. Register the hotkey on SN22

Register the hotkey you will run on `netuid 22` against `finney`.

```bash
# Miner
btcli subnet register \
  --netuid 22 \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney

# Validator
btcli subnet register \
  --netuid 22 \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network finney
```

Check registration and wallet state:

```bash
btcli wallet overview --wallet.name miner --subtensor.network finney
btcli wallet overview --wallet.name validator --subtensor.network finney
```

## 4. Configure credentials

### Miner

```bash
cp neurons/miners/.env.template neurons/miners/.env
$EDITOR neurons/miners/.env
```

Required miner secrets include `OPENAI_API_KEY`, `APIFY_API_KEY`, and `SERPAPI_API_KEY`. Set `WALLET_NAME`, `WALLET_HOTKEY`, `SUBTENSOR_NETWORK=finney`, `NETUID=22`, and `AXON_PORT` as needed.

Create `neurons/miners/manifest.json` from the template and declare the concurrency you can serve for each search type.

### Validator

Export validator secrets in the shell/session that PM2 will inherit:

```bash
export OPENAI_API_KEY="***"
export APIFY_API_KEY="***"
export SCRAPINGDOG_API_KEY="***"
export WANDB_API_KEY="***"
export EXPECTED_ACCESS_KEY="$(python3 scripts/generate_access_key.py)"
```

`EXPECTED_ACCESS_KEY` protects the validator API. Share it only with trusted Desearch services that call the validator with the `access-key` header.

## 5. Start the role

### Miner

```bash
pm2 start neurons/miners/miner.py \
  --interpreter /usr/bin/python3 \
  --name desearch_miner \
  -- \
  --wallet.name miner \
  --wallet.hotkey default \
  --netuid 22 \
  --subtensor.network finney \
  --axon.port 8098
```

### Validator

```bash
pm2 start run.sh --name desearch_autoupdate -- \
  --wallet.name validator \
  --wallet.hotkey default \
  --netuid 22 \
  --subtensor.network finney
```

`run.sh` manages `desearch_validator_process` and `desearch_api_process` for the validator.

## 6. Validate runtime health

```bash
pm2 status
pm2 logs desearch_miner
pm2 logs desearch_validator_process
pm2 logs desearch_api_process
```

Check the validator API locally when it is running:

```bash
curl -s 'http://localhost:8005/' -H 'access-key: <EXPECTED_ACCESS_KEY>'
curl -s 'http://localhost:8005/public/miners'
```

## 7. Development validation commands

Run these from the repo root before submitting changes:

```bash
python3 -m compileall desearch neurons tests scripts
pytest
grep -n '"/search/links/web"\|"/search/links/twitter"\|"/search/links"' neurons/validators/api.py
```
