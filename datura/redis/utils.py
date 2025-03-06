import torch
from datura.redis.redis_client import redis_client
import jsonpickle

REDIS_MOVING_AVERAGED_SCORES_KEY = "moving_averaged_scores"


def load_moving_averaged_scores(metagraph, config):
    scores = redis_client.get(REDIS_MOVING_AVERAGED_SCORES_KEY)

    if scores:
        return jsonpickle.decode(scores)

    return torch.zeros((metagraph.n)).to(config.neuron.device)


def save_moving_averaged_scores(scores):
    redis_client.set(REDIS_MOVING_AVERAGED_SCORES_KEY, jsonpickle.encode(scores))
