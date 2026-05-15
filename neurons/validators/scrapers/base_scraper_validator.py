import time
from datetime import datetime, timezone
from typing import List

import bittensor as bt
import numpy as np
import wandb

from neurons.validators.base_validator import AbstractNeuron
from neurons.validators.clients.miner_response_logger import (
    build_log_entry,
    build_reward_payload,
    submit_logs_best_effort,
)
from neurons.validators.reward.reward import log_reward_aggregates
from neurons.validators.scoring import capacity


class BaseScraperValidator:
    # Subclasses must set these
    search_type: str = ""
    wandb_modality: str = ""
    wandb_reward_keys: List[str] = []

    def __init__(
        self,
        neuron: AbstractNeuron,
        reward_weights: np.ndarray,
        reward_functions: list,
        penalty_functions: list,
    ):
        self.neuron = neuron

        self.reward_weights = np.asarray(reward_weights, dtype=np.float32)

        if self.reward_weights.sum() != 1:
            message = (
                f"Reward function weights do not sum to 1 (Current sum: {self.reward_weights.sum()}.)"
                f"Check your reward config file at `reward/config.py` or ensure that all your cli reward flags sum to 1."
            )
            bt.logging.error(message)
            raise Exception(message)

        self.reward_functions = reward_functions
        self.penalty_functions = penalty_functions

    def compute_reward_weights_matrix(self, responses) -> np.ndarray:
        """Returns an (N, K) array where row i has reward-function weights for
        response i. Default broadcasts the scraper's fixed weights to all
        responses. Override in subclasses that need per-response weighting
        (e.g. tool-varying AI search).
        """
        n = len(responses)
        return np.broadcast_to(
            self.reward_weights, (n, self.reward_weights.shape[0])
        ).copy()

    async def compute_cheap_scores(self, responses, uids) -> np.ndarray:
        """Apply only non-deep reward + penalty functions and return per-response
        scores in [0, 1]. Cheap reward weights are renormalized per row to sum
        to 1 so cheap scores are comparable to full scores."""
        if not responses:
            return np.zeros(0, dtype=np.float32)

        weights_matrix = self.compute_reward_weights_matrix(responses)
        cheap_cols = np.array(
            [not fn.is_deep for fn in self.reward_functions], dtype=bool
        )
        cheap_weight_sum = weights_matrix[:, cheap_cols].sum(axis=1)
        cheap_weight_sum = np.where(cheap_weight_sum > 0, cheap_weight_sum, 1.0)

        rewards = np.zeros(len(responses), dtype=np.float32)

        for i, reward_fn in enumerate(self.reward_functions):
            if reward_fn.is_deep:
                continue
            reward_i, _, _, _ = await reward_fn.apply(responses, uids)
            weight = weights_matrix[:, i] / cheap_weight_sum
            rewards += weight * np.asarray(reward_i, dtype=np.float32)

        for penalty_fn in self.penalty_functions:
            if penalty_fn.is_deep:
                continue
            _, _, applied = await penalty_fn.apply_penalties(responses, uids, None)
            rewards *= np.asarray(applied, dtype=np.float32)

        return rewards

    async def _dendrite_call(self, axon, synapse, uid: int):
        """Send a non-streaming synapse to a miner axon via dendrite. Tracks
        per-call success so consecutive failures flag the miner unreachable."""
        dendrite = next(self.neuron.dendrites)
        success = False

        try:
            response = await dendrite.call(
                target_axon=axon,
                synapse=synapse,
                timeout=synapse.max_execution_time + 5,
                deserialize=False,
            )
            status = getattr(getattr(response, "dendrite", None), "status_code", None)
            success = status == 200
        except Exception as e:
            bt.logging.error(
                f"[{self.search_type}] dendrite call failed uid={uid}: {e}"
            )
            response = synapse

        await capacity.note_call_result(uid, self.search_type, success)
        return response

    async def _save_organic_for_scoring(self, uid: int, response) -> None:
        """Persist an organic response in ScoringStore under the current UTC hour."""
        store = getattr(self.neuron, "scoring_store", None)
        if store is None or uid is None or response is None:
            return
        hour_bucket = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        try:
            await store.save_organic(hour_bucket, uid, self.search_type, response)
        except Exception as e:
            bt.logging.warning(f"[Organic] save_organic failed uid={uid}: {e}")

    def get_penalty_additional_params(self, val_score_responses_list):
        """Override in subclasses that need to pass additional params to penalties (e.g. val_scores)."""
        return None

    def build_wandb_data(self, uids, rewards, responses, all_rewards):
        """Build W&B logging data. Override for custom reward key mapping."""
        wandb_data = {
            "modality": self.wandb_modality,
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }
        for key in self.wandb_reward_keys:
            wandb_data[key] = {}
        return wandb_data

    def populate_wandb_uid_data(self, wandb_data, uid, reward, response, reward_values):
        """Populate per-UID wandb data. Override for custom prompt/reward extraction."""
        wandb_data["scores"][uid] = reward
        if hasattr(response, "query"):
            wandb_data["prompts"][uid] = response.query
        elif hasattr(response, "id"):
            wandb_data["prompts"][uid] = response.id
        elif hasattr(response, "urls"):
            wandb_data["prompts"][uid] = response.urls
        for key, value in zip(self.wandb_reward_keys, reward_values):
            wandb_data[key][uid] = value

    async def compute_rewards_and_penalties(
        self,
        event,
        prompts: List[str],
        responses,
        uids,
        start_time,
        result_type=None,
        scoring_epoch_start=None,
    ):
        try:
            if not len(uids):
                bt.logging.warning("No UIDs provided for logging event.")
                return

            bt.logging.info("Computing rewards and penalties")

            rewards = np.zeros(len(responses), dtype=np.float32)

            all_rewards = []
            all_original_rewards = []
            val_score_responses_list = []

            weights_matrix = self.compute_reward_weights_matrix(responses)

            for i, reward_fn_i in enumerate(self.reward_functions):
                start_time = time.time()
                (
                    reward_i,
                    reward_event,
                    val_score_responses,
                    original_rewards,
                ) = await reward_fn_i.apply(responses, uids)

                all_rewards.append(reward_i)
                all_original_rewards.append(original_rewards)
                val_score_responses_list.append(val_score_responses)

                rewards += weights_matrix[:, i] * np.asarray(reward_i, dtype=np.float32)

                if not self.neuron.config.neuron.disable_log_rewards:
                    event = {**event, **reward_event}

                execution_time = time.time() - start_time
                bt.logging.trace(str(reward_fn_i.name), reward_i.tolist())
                bt.logging.info(
                    f"Applied reward function: {reward_fn_i.name} in {execution_time / 60:.2f} minutes"
                )

            penalty_additional_params = self.get_penalty_additional_params(
                val_score_responses_list
            )

            for penalty_fn_i in self.penalty_functions:
                (
                    raw_penalty_i,
                    adjusted_penalty_i,
                    applied_penalty_i,
                ) = await penalty_fn_i.apply_penalties(
                    responses, uids, penalty_additional_params
                )

                rewards *= np.asarray(applied_penalty_i, dtype=np.float32)

                if not self.neuron.config.neuron.disable_log_rewards:
                    event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
                    event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
                    event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()

            self.log_event(prompts, event, start_time, uids, rewards)

            scores = np.zeros(len(self.neuron.metagraph.hotkeys), dtype=np.float32)
            uid_scores_dict = {}
            wandb_data = self.build_wandb_data(uids, rewards, responses, all_rewards)

            log_reward_aggregates(
                name=f"{self.search_type} total",
                uids=uids,
                scores=rewards.tolist(),
            )

            # Build per-uid reward values for wandb
            reward_values_per_uid = (
                list(
                    zip(
                        *[
                            r.tolist() if hasattr(r, "tolist") else r
                            for r in all_rewards
                        ]
                    )
                )
                if all_rewards
                else [() for _ in uids]
            )

            for uid_tensor, reward, response, reward_values in zip(
                uids, rewards.tolist(), responses, reward_values_per_uid
            ):
                uid = uid_tensor.item()
                uid_scores_dict[uid] = reward
                scores[uid] = reward
                self.populate_wandb_uid_data(
                    wandb_data, uid, reward, response, reward_values
                )

            if self.neuron.config.wandb_on:
                wandb.log(wandb_data)

            scoring_logs = []
            response_count = len(responses)

            for index, (uid_tensor, response, reward) in enumerate(
                zip(uids, responses, rewards.tolist())
            ):
                uid = uid_tensor.item()
                reward_payload = build_reward_payload(
                    search_type=self.search_type,
                    response_count=response_count,
                    index=index,
                    uid=uid,
                    total_reward=reward,
                    all_rewards=all_rewards,
                    all_original_rewards=all_original_rewards,
                    validator_scores=val_score_responses_list,
                    event=event,
                )
                scoring_logs.append(
                    build_log_entry(
                        owner=self.neuron,
                        search_type=self.search_type,
                        query_kind="scoring",
                        response=response,
                        miner_uid=uid,
                        total_reward=reward,
                        reward_payload=reward_payload,
                        scoring_epoch_start=scoring_epoch_start,
                    )
                )

            submit_logs_best_effort(self.neuron, scoring_logs)

            return rewards, uids, val_score_responses_list, event, all_original_rewards
        except Exception as e:
            bt.logging.error(f"Error in compute_rewards_and_penalties: {e}")
            raise e

    def log_event(self, prompts: List[str], event, start_time, uids, rewards):
        event.update(
            {
                "step_length": time.time() - start_time,
                "prompts": prompts,
                "uids": uids.tolist(),
                "rewards": rewards.tolist(),
            }
        )

        bt.logging.debug("Run Task event:", event)
