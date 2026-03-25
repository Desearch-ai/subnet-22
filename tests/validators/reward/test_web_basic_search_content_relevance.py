import unittest
from neurons.validators.reward.web_basic_search_content_relevance import (
    WebBasicSearchContentRelevanceModel,
)
from desearch.protocol import WebSearchSynapse, WebSearchValidatorResult
from tests_data.links.links import link1, link2, link3, link4, link5


class WebBasicSearchContentRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = WebBasicSearchContentRelevanceModel(self.device, self.scoring_type)

    async def test_get_rewards(self):
        rewards, grouped_score = await self.model.get_rewards(
            [
                WebSearchSynapse(query="python", results=[link1, link2]),
                WebSearchSynapse(query="python", results=[link2, link3]),
                WebSearchSynapse(query="python", results=[link1, link4, link5]),
            ],
            [
                1,
                2,
                3,
            ],
        )

        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0].reward, 0.5)
        self.assertEqual(rewards[1].reward, 1)
        self.assertEqual(rewards[2].reward, 0)
        self.assertEqual(grouped_score, {1: 0.5, 2: 1, 3: 0})

    async def test_process_links(self):
        synapse = WebSearchSynapse(
            query="python", results=[link1, link2, link3, link4, link5]
        )

        await self.model.process_links([synapse])

        self.assertEqual(len(synapse.validator_links), 2)

        links = [link["link"] for link in synapse.results]
        self.assertTrue(all(item.link in links for item in synapse.validator_links))

    async def test_check_respones_random_link(self):
        synapse = WebSearchSynapse(
            query="python",
            results=[link1, link2, link3, link4, link5],
            validator_links=[
                WebSearchValidatorResult(**link1, html_text=link1["snippet"]),
                WebSearchValidatorResult(**link2, html_content=link2["snippet"]),
            ],
        )

        result = self.model.check_response_random_link(synapse)
        self.assertEqual(result, 1)

        for falseCase in [
            {
                "case": WebSearchSynapse(
                    query="python",
                    results=[link1, link2, link3, link4, link5],
                    validator_links=[],
                ),
                "description": "should return 0 : no validator links",
            },
            {
                "case": WebSearchSynapse(
                    query="python",
                    results=[link1, link2, link3, link4, link5],
                    validator_links=[
                        WebSearchValidatorResult(
                            **link1,
                        ),
                        WebSearchValidatorResult(
                            **link2,
                        ),
                    ],
                ),
                "description": "should return 0 : missing snippet in html content",
            },
            {
                "case": WebSearchSynapse(
                    query="python",
                    results=[link1, link2, link3, link4, link5],
                    validator_links=[
                        {**link1, "html_text": link1["snippet"], "title": "wrong title"}
                    ],
                ),
                "description": "should return 0 : incorrect title",
            },
            {
                "case": WebSearchSynapse(
                    query="python",
                    results=[link1, link2, link3, link4, link5],
                    validator_links=[
                        {**link1, "html_text": link1["snippet"], "link": "wrong link"}
                    ],
                ),
                "description": "should return 0 : incorrect link",
            },
            {
                "case": WebSearchSynapse(
                    query="blockchain",
                    results=[link1, link2, link3, link4, link5],
                    validator_links=[{**link1, "html_text": link1["snippet"]}],
                ),
                "description": "should return 0 : mismatching query",
            },
        ]:
            self.assertEqual(
                self.model.check_response_random_link(falseCase["case"]),
                0,
                falseCase["description"],
            )


if __name__ == "__main__":
    unittest.main()
