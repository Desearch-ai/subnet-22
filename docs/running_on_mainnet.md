# Running on the mainneting Network
This tutorial shows how to use the bittensor mainnetwork to create a subnetwork and connect your mechanism to it. It is highly recommended that you run mainnet first before mainnet. 
## Steps

1. Clone and Install Bittensor and the Bittensor Subnet datura.
This clones and installs the datura if you dont already have it (if you do, skip this step)
```bash
cd .. # back out of the subtensor repo
git clone https://github.com/Desearch-ai/subnet-22.git # Clone the Desearch repo
cd desearch # Enter the  Desearch  repo
python -m pip install -e . # Install the  Desearch  package
```

2. Create wallets for your subnet owner, for your validator and for your miner.
This creates local coldkey and hotkey pairs for your 3 identities. The owner will create and control the subnet and must have at least 100 mainnet net TAO on it before it can run step 3. The validator and miner will run the respective validator/miner scripts and be registered to the subnetwork created by the owner.
```bash
# Create a coldkey for your owner wallet.
btcli new_coldkey --wallet.name owner

# Create a coldkey and hotkey for your miner wallet.
btcli new_coldkey --wallet.name miner
btcli new_hotkey --wallet.name miner --wallet.hotkey default

# Create a coldkey and hotkey for your validator wallet.
btcli new_coldkey --wallet.name validator
btcli new_hotkey --wallet.name validator --wallet.hotkey default
```

3. Getting the price of subnetwork creation
Creating subnetworks on the mainnet is competitive and the cost it determined by the rate at which new network are being registered onto the chain. By default you must have at least 100 mainnet TAO on your owner wallet to create a subnetwork. However the exact amount will fluctuate based on demand. The code below shows how to get the current price of creating a subnetwork.
```bash
btcli get_subnet_cost --subtensor.network finney
>> Subnet burn cost: τ100.000000000
```

4. (Optional) Getting faucet tokens
If you dont have enough to create a subnet you can pull mainnet faucet tokens by solving periodic POW challenges. The code below shows how to get faucet tokens.
This step may take a while to complete depending on your system. Once you have enough TAO to purchase your slot, continue to the next step.
```bash
btcli run_faucet --wallet.name owner --subtensor.network finney
>> Balance: τ0.000000000 ➡ τ100.000000000
>> Balance: τ100.000000000 ➡ τ200.000000000
...
```

3. Purchasing a slot
Using the TAO from the previous step you can register your subnet to the chain. This will create a new subnet on the chain and give you the owner permissions to it. The code below shows how to purchase a slot. 
*Note: Slots cost TAO, you wont get this TAO back, it is instead recycled back into the mechanism to be later mined.*
```bash
# Run the register subnetwork command on the locally running chain.
btcli register_subnet --subtensor.network finney 
# Enter the owner wallet name which gives permissions to the coldkey to later define running hyper parameters.
>> Enter wallet name (default): owner # Enter your owner wallet name
>> Enter password to unlock key: # Enter your wallet password.
>> Register subnet? [y/n]: <y/n> # Select yes (y)
>> ⠇ 📡 Registering subnet...
✅ Registered subnetwork with netuid: 1 # Your subnet netuid will show here, save this for later.
```

10. Register your validator and miner keys to the networks.
This registers your validator and miner keys to the network giving them the first 2 slots on the network.
```bash
# Register your miner key to the network.
btcli register --wallet.name miner --wallet.hotkey default  --subtensor.network finney
>> Enter netuid [1] (1): # Enter netuid 1 to specify the network you just created.
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered

# Register your validator key to the network.
btcli register --wallet.name validator --wallet.hotkey default --subtensor.network finney
>> Enter netuid [1] (1): # Enter netuid 1 to specify the network you just created.
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered
```

11. Check that your keys have been registered.
This returns information about your registered keys.
```bash
# Check that your validator key has been registered.
btcli overview --wallet.name validator --subtensor.network finney
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  0      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0         

# Check that your miner has been registered.
btcli overview --wallet.name miner --subtensor.network finney
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  1      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0   
```

10. Edit the default `NETUID=1` and `CHAIN_ENDPOINT=ws://127.0.0.1:9946` arguments in `datura/__init__.py` to match your created subnetwork.
Or run the miner and validator directly with the netuid and chain_endpoint arguments.
```bash
# Run the miner with the netuid and chain_endpoint arguments.
python neurons/miner.py --netuid 1 --subtensor.network finney --wallet.name miner --wallet.hotkey default
>> 2023-08-08 16:58:11.223 |       INFO       | Running miner for subnet: 1 on network: ws://127.0.0.1:9946 with config: ...

# Run the validator with the netuid and chain_endpoint arguments.
python neurons/validator.py --netuid 1 --subtensor.network finney --wallet.name validator --wallet.hotkey default
>> 2023-08-08 16:58:11.223 |       INFO       | Running validator for subnet: 1 on network: ws://127.0.0.1:9946 with config: ...
```

7. Stopping Your Nodes:
If you want to stop your nodes, you can do so by pressing CTRL + C in the terminal where the nodes are running.
