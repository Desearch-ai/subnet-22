import unittest

from neurons.validators.reward.twitter_content_relevance import (
    TwitterContentRelevanceModel,
)


class ContentRelevanceScoreClampTestCase(unittest.TestCase):
    def setUp(self):
        self.model = TwitterContentRelevanceModel.__new__(TwitterContentRelevanceModel)

    def test_relevance_passes_through_without_link_bonus(self):
        low_relevance = 1 / 3

        self.assertEqual(self.model.clamp_relevance_score(low_relevance), low_relevance)
        self.assertEqual(self.model.clamp_relevance_score(0.75), 0.75)

    def test_relevance_is_clamped_to_reward_range(self):
        self.assertEqual(self.model.clamp_relevance_score(1.25), 1)
        self.assertEqual(self.model.clamp_relevance_score(-0.25), 0)


if __name__ == "__main__":
    unittest.main()
