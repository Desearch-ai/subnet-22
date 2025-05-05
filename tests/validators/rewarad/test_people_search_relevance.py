import unittest
from neurons.validators.reward.people_search_relevance import PeopleSearchRelevanceModel
from datura.protocol import PeopleSearchResult, PeopleSearchSynapse
from tests_data.profiles.profile1 import profile1
from tests_data.profiles.profile2 import profile2


class PeopleSearchRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = PeopleSearchRelevanceModel(self.device, self.scoring_type)

    async def test_get_rewards(self):
        rewards, grouped_score = await self.model.get_rewards(
            [
                PeopleSearchSynapse(
                    query="AI startup founders in London with a PhD in machine learning",
                    criteria=[
                        "Founder role at an AI startup",
                        "Located in London",
                        "Holds a PhD in machine learning",
                    ],
                    results=[profile1, profile2],
                ),
                PeopleSearchSynapse(
                    query="Sports player based in London",
                    criteria=["Sports player role", "Based in London"],
                    results=[profile1, profile2],
                ),
                PeopleSearchSynapse(
                    query="Sports player based in US",
                    criteria=["Sports player role", "Based in US"],
                    results=[profile1, profile2],
                ),
            ],
            [
                1,
                2,
                3,
            ],
        )

        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0].reward, 1)
        self.assertEqual(rewards[1].reward, 0.5)
        self.assertEqual(rewards[2].reward, 0)
        self.assertEqual(grouped_score, {1: 1, 2: 0.5, 3: 0})

    # async def test_process_profiles(self):
    #     synapse = PeopleSearchSynapse(
    #         query="AI startup founders in London with a PhD in machine learning",
    #         results=[profile1, profile2],
    #     )

    #     await self.model.process_profiles([synapse])

    #     self.assertEqual(len(synapse.validator_results), 2)

    #     links = [link["link"] for link in synapse.results]
    #     self.assertTrue(all(item.link in links for item in synapse.validator_results))

    async def test_check_criteria(self):
        score = await self.model.check_criteria(
            PeopleSearchSynapse(
                query="AI startup founders in London with a PhD in machine learning",
                criteria=[
                    "Founder role at an AI startup",
                    "Located in London",
                    "Holds a PhD in machine learning",
                ],
            ),
            profile1,
        )
        self.assertEqual(score, 1)

        score = await self.model.check_criteria(
            PeopleSearchSynapse(
                query="Sports player based in London",
                criteria=["Sports player role", "Based in London"],
            ),
            profile1,
        )
        self.assertEqual(score, 0.5)

        score = await self.model.check_criteria(
            PeopleSearchSynapse(
                query="Sports player based in US",
                criteria=["Sports player role", "Based in US"],
            ),
            profile2,
        )
        self.assertEqual(score, 0.0)

    async def test_check_response(self):
        score = await self.model.check_response(
            PeopleSearchSynapse(
                query="AI startup founders in London with a PhD in machine learning",
                criteria=[
                    "Founder role at an AI startup",
                    "Located in London",
                    "Holds a PhD in machine learning",
                ],
                results=[{**profile1, "first_name": "wrong name"}],
            ),
        )
        self.assertEqual(score, 0)

        score = await self.model.check_response(
            PeopleSearchSynapse(
                query="AI startup founders in London with a PhD in machine learning",
                criteria=[
                    "Founder role at an AI startup",
                    "Located in London",
                    "Holds a PhD in machine learning",
                ],
                results=[{**profile1, "experiences": [profile1["experiences"][0]]}],
            ),
        )
        self.assertEqual(score, 0)

    def test_compare_lists(self):
        self.assertTrue(
            self.model.compare_lists(
                [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                [{"c": 3, "d": 4}, {"b": 2, "a": 1}],
            )
        )

        self.assertFalse(
            self.model.compare_lists(
                [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                [{"c": 3, "d": 4}, {"b": 2, "a": 2}],
            )
        )

        self.assertFalse(
            self.model.compare_lists(
                [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                [{"c": 3, "d": 4}, {"b": 2}],
            )
        )

        self.assertFalse(
            self.model.compare_lists(
                [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
                [{"c": 3, "d": 4}],
            )
        )


if __name__ == "__main__":
    unittest.main()
