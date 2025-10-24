# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright d© 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Utils for weights setting on chain.

import wandb
import torch
import bittensor as bt
import desearch
import time
import torch

ENABLE_EMISSION_CONTROL = True
EMISSION_CONTROL_HOTKEY = "5CUu1QhvrfyMDBELUPJLt4c7uJFbi7TKqDHkS1Zz41oD4dyP"
EMISSION_CONTROL_PERC = 0.8


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
                reinit="default",
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


def set_weights_subtensor(netuid, uids, weights, config, version_key):
    try:
        wallet = bt.wallet(config=config)
        subtensor = bt.subtensor(config=config)
        success, message = subtensor.set_weights(
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
        bt.logging.error(f"Failed to set weights on chain with exception: { e }")
        return False, message


def set_weights_with_retry(self, processed_weight_uids, processed_weights):
    max_retries = 9  # Maximum number of retries
    retry_delay = 45  # Delay between retries in seconds

    success = False

    bt.logging.info("Initiating weight setting process on Bittensor network.")
    for attempt in range(max_retries):
        success, message = set_weights_subtensor(
            self.config.netuid,
            processed_weight_uids,
            processed_weights,
            self.config,
            desearch.__weights_version__,
        )

        if success:
            bt.logging.success(
                f"Set Weights Completed set weights action successfully. Message: '{message}'"
            )

            break
        else:
            bt.logging.info(
                f"Set Weights Attempt failed with message: '{message}', retrying in {retry_delay} seconds..."
            )

            time.sleep(retry_delay)

    if success:
        bt.logging.success(
            f"Final Result: Successfully set weights after {attempt + 1} attempts."
        )
    else:
        bt.logging.error(
            f"Final Result: Failed to set weights after {attempt + 1} attempts."
        )

    return success


def find_target_uid(self, hotkey):
    for neuron in self.metagraph.neurons:
        if neuron.hotkey == hotkey:
            emission_control_uid = neuron.uid

            return emission_control_uid


def burn_weights(self, weights):
    target_uid = find_target_uid(self, EMISSION_CONTROL_HOTKEY)

    if not target_uid:
        bt.logging.info(f"target hotkey {EMISSION_CONTROL_HOTKEY} is not found")
        return weights

    total_score = torch.sum(weights)

    new_target_score = EMISSION_CONTROL_PERC * total_score
    remaining_weight = (1 - EMISSION_CONTROL_PERC) * total_score
    total_other_scores = total_score - weights[target_uid]

    if total_other_scores == 0:
        bt.logging.warning("All scores are zero except target UID, cannot scale.")
        return weights

    new_scores = torch.zeros_like(weights, dtype=float)
    uids = self.metagraph.uids

    for i, (uid, weight) in enumerate(zip(uids, weights)):
        if uid == target_uid:
            new_scores[i] = new_target_score
        else:
            new_scores[i] = (weight / total_other_scores) * remaining_weight

    return new_scores


def process_weights(self, raw_weights):
    max_retries = 5  # Define the maximum number of retries
    retry_delay = 30  # Define the delay between retries in seconds

    weights = raw_weights

    if ENABLE_EMISSION_CONTROL:
        weights = burn_weights(self, weights)

    for attempt in range(max_retries):
        try:
            subtensor = bt.subtensor(config=self.config)

            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids.to("cpu"),
                weights=weights.to("cpu"),
                netuid=self.config.netuid,
                subtensor=subtensor,
                metagraph=self.metagraph,
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
                time.sleep(retry_delay)
            else:
                return {}, None, None


def get_weights(self):
    if torch.all(self.moving_averaged_scores == 0):
        bt.logging.info(
            "All moving averaged scores are zero. Skipping weight retrieval."
        )
        return {}

    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)

    weights_dict, _, _ = process_weights(self, raw_weights)

    return weights_dict


def set_weights(self):
    if torch.all(self.moving_averaged_scores == 0):
        bt.logging.info("All moving averaged scores are zero, skipping weight setting.")
        return
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)
    bt.logging.trace("raw_weights", raw_weights)
    bt.logging.trace("top10 values", raw_weights.sort()[0])
    bt.logging.trace("top10 uids", raw_weights.sort()[1])

    # Process the raw weights to final_weights via subtensor limitations.
    weights_dict, processed_weight_uids, processed_weights = process_weights(
        self, raw_weights
    )

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
    bt.logging.info(f"Attempting to set weights details ends: ================")

    # Call the new method to handle the process with retry logic
    success = set_weights_with_retry(self, processed_weight_uids, processed_weights)
    return success
