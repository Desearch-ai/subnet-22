import unittest
from types import SimpleNamespace

import numpy as np

from neurons.validators.reward.content_relevance import ContentRelevanceRewardModel
from neurons.validators.reward.reward import BaseRewardEvent


class _StubModel:
    def __init__(self, reward, label):
        self.reward = reward
        self.label = label
        self.seen = None

    async def get_rewards(self, responses, uids):
        self.seen = (responses, list(uids))
        events = [BaseRewardEvent(reward=self.reward) for _ in responses]
        labels = [{"u": self.label} for _ in responses]
        return events, labels


class ContentRelevanceDispatchTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_by_tool(self):
        model = ContentRelevanceRewardModel.__new__(ContentRelevanceRewardModel)
        model.twitter = _StubModel(0.9, "TW")
        model.web = _StubModel(0.4, "WEB")

        responses = [
            SimpleNamespace(tools=["Web Search"]),
            SimpleNamespace(tools=["Twitter Search"]),
            SimpleNamespace(tools=["Web Search"]),
        ]
        uids = np.array([10, 11, 12])

        events, labels = await model.get_rewards(responses, uids)

        self.assertEqual([e.reward for e in events], [0.4, 0.9, 0.4])
        self.assertEqual([entry["u"] for entry in labels], ["WEB", "TW", "WEB"])
        self.assertEqual(len(model.twitter.seen[0]), 1)
        self.assertEqual(model.twitter.seen[1], [11])
        self.assertEqual(len(model.web.seen[0]), 2)
        self.assertEqual(model.web.seen[1], [10, 12])


class WandbToolRoutingTestCase(unittest.TestCase):
    def test_content_reward_logged_under_tool_name(self):
        from neurons.validators.scrapers.advanced_scraper_validator import (
            AdvancedScraperValidator,
        )

        v = AdvancedScraperValidator.__new__(AdvancedScraperValidator)
        keys = [
            "scores",
            "responses",
            "prompts",
            "twitter_reward",
            "search_reward",
            "summary_reward",
        ]
        wandb_data = {k: {} for k in keys}

        x = SimpleNamespace(tools=["Twitter Search"], completion="c", prompt="p")
        web = SimpleNamespace(tools=["Web Search"], completion="c", prompt="p")
        v.populate_wandb_uid_data(wandb_data, 1, 0.7, x, [0.9, 0.8, 1.0])
        v.populate_wandb_uid_data(wandb_data, 2, 0.5, web, [0.4, 0.6, 1.0])

        self.assertEqual(wandb_data["twitter_reward"], {1: 0.9, 2: 0.0})
        self.assertEqual(wandb_data["search_reward"], {1: 0.0, 2: 0.4})
        self.assertEqual(wandb_data["summary_reward"], {1: 0.8, 2: 0.6})


class UidsNormalizationTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_plain_list_uids(self):
        model = ContentRelevanceRewardModel.__new__(ContentRelevanceRewardModel)
        model.twitter = _StubModel(0.9, "TW")
        model.web = _StubModel(0.4, "WEB")

        responses = [
            SimpleNamespace(tools=["Twitter Search"]),
            SimpleNamespace(tools=["Web Search"]),
        ]

        events, _ = await model.get_rewards(responses, [10, 11])

        self.assertEqual([e.reward for e in events], [0.9, 0.4])
        self.assertEqual(model.twitter.seen[1], [10])
        self.assertEqual(model.web.seen[1], [11])


if __name__ == "__main__":
    unittest.main()
