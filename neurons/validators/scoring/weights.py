import asyncio

import bittensor as bt
import numpy as np
import wandb
from bittensor.utils.weight_utils import process_weights

import desearch

EMISSION_CONTROL_HOTKEY = "5CUu1QhvrfyMDBELUPJLt4c7uJFbi7TKqDHkS1Zz41oD4dyP"


def init_wandb(self):
    try:
        if self.config.wandb_on:
            run_name = f"validator-{self.uid}-{desearch.__version__}"
            self.config.uid = self.uid
            self.config.hotkey = self.wallet.hotkey.ss58_address
            self.config.run_name = run_name
            self.config.version = desearch.__version__
            self.config.type = "validator"

            # Initialize the wandb run for the single project
            run = wandb.init(
                name=run_name,
                project=desearch.PROJECT_NAME,
                entity=desearch.ENTITY,
                config=self.config,
                dir=self.config.full_path,
                reinit="finish_previous",
            )

            # Sign the run to ensure it's from the correct hotkey
            signature = self.wallet.hotkey.sign(run.id.encode()).hex()
            self.config.signature = signature
            wandb.config.update(self.config, allow_val_change=True)

            bt.logging.success(
                f"Started wandb run for project '{desearch.PROJECT_NAME}'"
            )
    except Exception as e:
        bt.logging.error(f"Error in init_wandb: {e}")
        raise


async def set_weights_subtensor(
    subtensor: bt.AsyncSubtensor, wallet: bt.Wallet, netuid, uids, weights, version_key
):
    try:
        success, message = await subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            version_key=version_key,
        )

        # Send the success status back to the main process
        return success, message
    except Exception as e:
        bt.logging.error(f"Failed to set weights on chain with exception: {e}")
        return False, message


async def set_weights_with_retry(self, processed_weight_uids, processed_weights):
    max_retries = 9  # Maximum number of retries
    retry_delay = 45  # Delay between retries in seconds

    success = False

    bt.logging.info("Starting to set weights...")

    for attempt in range(max_retries):
        success, message = await set_weights_subtensor(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            version_key=desearch.__weights_version__,
        )

        if success:
            bt.logging.success(f"Set weights completed with message: '{message}'")

            break
        else:
            bt.logging.info(
                f"Set weights failed with message: '{message}', retrying in {retry_delay} seconds..."
            )

            await asyncio.sleep(retry_delay)

    if success:
        bt.logging.success(f"Successfully set weights after {attempt + 1} attempts.")
    else:
        bt.logging.error(f"Failed to set weights after {attempt + 1} attempts.")

    return success


def find_target_uid(self, hotkey):
    for neuron in self.metagraph.neurons:
        if neuron.hotkey == hotkey:
            emission_control_uid = neuron.uid

            return emission_control_uid


def burn_weights(self) -> np.ndarray | None:
    target_uid = find_target_uid(self, EMISSION_CONTROL_HOTKEY)

    if target_uid is None:
        return None

    weights = np.zeros(len(self.metagraph.uids), dtype=np.float32)
    weights[target_uid] = 1.0

    return weights


async def process_weights_with_retry(self, weights):
    max_retries = 5  # Define the maximum number of retries
    retry_delay = 30  # Define the delay between retries in seconds

    netuid = self.config.netuid

    for attempt in range(max_retries):
        try:
            # process_weights_for_netuid uses sync subtensor calls for retrieving min and max values, we can directly call process_weight
            # https://github.com/opentensor/bittensor/blob/master/bittensor/utils/weight_utils.py#L253
            min_allowed_weights = await self.subtensor.min_allowed_weights(
                netuid=netuid
            )
            max_weight_limit = await self.subtensor.max_weight_limit(netuid=netuid)

            (
                processed_weight_uids,
                processed_weights,
            ) = process_weights(
                uids=self.metagraph.uids,
                weights=weights,
                num_neurons=int(self.metagraph.n),
                min_allowed_weights=min_allowed_weights,
                max_weight_limit=max_weight_limit,
            )

            weights_dict = {
                str(uid.item()): weight.item()
                for uid, weight in zip(processed_weight_uids, processed_weights)
            }

            return weights_dict, processed_weight_uids, processed_weights
        except Exception as e:
            bt.logging.error(f"Error in process_weights (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                bt.logging.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                return {}, None, None


async def set_weights(self):
    raw_weights = burn_weights(self)
    # Never fall back to zeros: process_weights would spread them evenly over every UID.
    if raw_weights is None:
        bt.logging.error(
            f"Burn hotkey {EMISSION_CONTROL_HOTKEY} is not in the metagraph, skipping weight setting."
        )
        return False

    # Process the raw weights to final_weights via subtensor limitations.
    (
        weights_dict,
        processed_weight_uids,
        processed_weights,
    ) = await process_weights_with_retry(self, raw_weights)

    if processed_weight_uids is None:
        return

    # Log the weights dictionary
    bt.logging.info(f"Attempting to set weights action for {weights_dict}")

    bt.logging.info(
        f"Attempting to set weights details begins: ================ for {len(processed_weight_uids)} UIDs"
    )
    uids_weights = [
        f"UID - {uid.item()} = Weight - {weight.item()}"
        for uid, weight in zip(processed_weight_uids, processed_weights)
    ]
    for i in range(0, len(uids_weights), 4):
        bt.logging.info(" | ".join(uids_weights[i : i + 4]))
    bt.logging.info("Attempting to set weights details ends: ================")

    # Call the new method to handle the process with retry logic
    success = await set_weights_with_retry(
        self, processed_weight_uids, processed_weights
    )

    return success
