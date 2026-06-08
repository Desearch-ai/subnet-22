import re
import unittest

from desearch.protocol import WebSearchSynapse, WebSearchValidatorResult
from neurons.validators.reward.web_basic_search_content_relevance import (
    WEB_LINK_SCRAPE_AMOUNT,
    WebBasicSearchContentRelevanceModel,
)
from neurons.validators.utils.web_query_operators import parse_web_query
from tests_data.links.links import link1, link2, link3, link4, link5

ALL_LINKS = (link1, link2, link3, link4, link5)


def _validator_link(base, **overrides):
    """A validator scrape whose page HTML contains the snippet (passes authenticity)."""
    data = {**base, "html_text": base["snippet"], "html_content": base["snippet"]}
    data.update(overrides)
    return WebSearchValidatorResult(**data)


def _excerpt_page(snippet: str) -> str:
    """Page where the snippet appears as a contiguous excerpt within more text."""
    return f"Welcome to our site. {snippet} Read more below in the full article."


def _scrambled_page(snippet: str) -> str:
    """Snippet's words present but adjacency broken — not an excerpt (e.g. JS page)."""
    words = [w for w in re.split(r"\W+", snippet) if w]
    return "site nav " + " and ".join(words) + " footer links"


class FakeRewardLLM:
    """Returns a fixed verdict per URL and records the conversations it scored."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.scored = []

    async def llm_processing(self, messages):
        result = {}
        for message in messages:
            ((url, conversation),) = message.items()
            self.scored.append((url, conversation))
            result[url] = self.verdicts.get(url, "IRRELEVANT")
        return result


class CheckResponseRandomLinkTestCase(unittest.IsolatedAsyncioTestCase):
    """Gate logic in isolation: relevance is supplied directly, no LLM/network."""

    def setUp(self):
        self.model = WebBasicSearchContentRelevanceModel(
            scoring_type=None, neuron=None, llm_reward=None
        )

    def _synapse(self, validator_links, query="python"):
        return WebSearchSynapse(
            query=query, results=list(ALL_LINKS), validator_links=validator_links
        )

    def test_relevance_score_passes_through(self):
        synapse = self._synapse([_validator_link(link1), _validator_link(link2)])
        score = self.model.check_response_random_link(
            synapse, {link1["link"]: 1.0, link2["link"]: 1.0}
        )
        self.assertEqual(score, 1.0)

    def test_offtopic_link_scores_zero(self):
        # The customer's bug: a real page matched on one common word is now rejected.
        synapse = self._synapse([_validator_link(link1)])
        score = self.model.check_response_random_link(synapse, {link1["link"]: 0.0})
        self.assertEqual(score, 0.0)

    def test_mixed_relevance_averages(self):
        synapse = self._synapse([_validator_link(link1), _validator_link(link2)])
        score = self.model.check_response_random_link(
            synapse, {link1["link"]: 1.0, link2["link"]: 0.0}
        )
        self.assertEqual(score, 0.5)

    def test_missing_verdict_is_skipped_not_zeroed(self):
        synapse = self._synapse([_validator_link(link1), _validator_link(link2)])
        # link2 has no verdict (e.g. scoring outage) -> skipped, not counted as 0.
        score = self.model.check_response_random_link(synapse, {link1["link"]: 1.0})
        self.assertEqual(score, 1.0)

    def test_authenticity_gates_zero_regardless_of_relevance(self):
        relevant = {link1["link"]: 1.0, "wrong link": 1.0}

        cases = {
            "no validator links": [],
            "title mismatch": [_validator_link(link1, title="totally different title")],
            "link mismatch": [_validator_link(link1, link="wrong link")],
        }

        for description, validator_links in cases.items():
            score = self.model.check_response_random_link(
                self._synapse(validator_links), relevant
            )
            self.assertEqual(score, 0.0, description)

    def test_duplicate_miner_link_zeroes_response(self):
        synapse = WebSearchSynapse(
            query="python",
            results=[link1, link1],
            validator_links=[_validator_link(link1)],
        )
        score = self.model.check_response_random_link(synapse, {link1["link"]: 1.0})
        self.assertEqual(score, 0.0)


class SnippetAuthenticityTestCase(unittest.IsolatedAsyncioTestCase):
    """Tolerant snippet check: content-first with meta-description fallback + anti-gaming."""

    def setUp(self):
        self.model = WebBasicSearchContentRelevanceModel(
            scoring_type=None, neuron=None, llm_reward=None
        )

    def _synapse(self, validator_links, query="python"):
        return WebSearchSynapse(
            query=query, results=list(ALL_LINKS), validator_links=validator_links
        )

    def test_excerpt_snippet_passes_via_page_content(self):
        page = _excerpt_page(link1["snippet"])
        vlink = _validator_link(link1, html_text=page, html_content=page, snippet="")
        score = self.model.check_response_random_link(
            self._synapse([vlink]), {link1["link"]: 1.0}
        )
        self.assertEqual(score, 1.0)

    def test_unverifiable_snippet_scores_zero(self):
        page = _scrambled_page(link1["snippet"])
        vlink = _validator_link(link1, html_text=page, html_content=page, snippet=page)
        score = self.model.check_response_random_link(
            self._synapse([vlink]), {link1["link"]: 1.0}
        )
        self.assertEqual(score, 0.0)

    def test_filler_snippet_is_not_substantive(self):
        for filler in (
            "the the the the the the the the the the the",
            "python python python python python python python python",
        ):
            self.assertFalse(self.model._is_substantive_snippet(filler))

    def test_snippet_matches_via_meta_description_when_absent_from_body(self):
        # Body is navigation chrome; the snippet lives only in the meta description.
        vlink = _validator_link(
            link1,
            html_text="home about contact login menu",
            html_content="home about contact login menu",
            snippet=link1["snippet"],
        )
        score = self.model.check_response_random_link(
            self._synapse([vlink]), {link1["link"]: 1.0}
        )
        self.assertEqual(score, 1.0)

    def test_trivially_short_snippet_scores_zero(self):
        # Anti-gaming: a one-word snippet matches anything, so it earns no credit.
        gamed = {**link1, "snippet": "python"}
        vlink = _validator_link(
            gamed,
            html_text="python " * 50,
            html_content="python " * 50,
            snippet="python",
        )
        synapse = WebSearchSynapse(
            query="python", results=[gamed], validator_links=[vlink]
        )
        score = self.model.check_response_random_link(synapse, {link1["link"]: 1.0})
        self.assertEqual(score, 0.0)

    def test_empty_snippet_scores_zero(self):
        empty = {**link1, "snippet": ""}
        vlink = WebSearchValidatorResult(
            title=link1["title"],
            link=link1["link"],
            snippet="",
            html_text="anything at all here",
            html_content="anything at all here",
        )
        synapse = WebSearchSynapse(
            query="python", results=[empty], validator_links=[vlink]
        )
        score = self.model.check_response_random_link(synapse, {link1["link"]: 1.0})
        self.assertEqual(score, 0.0)

    def test_youtube_style_result_is_not_a_free_pass(self):
        # A YouTube page's html_text IS its title+description, so the snippet always
        # matches authenticity — but an off-topic video is still zeroed by relevance.
        yt = {
            "title": "What hard work looks like",
            "snippet": "A motivational video about discipline, effort and grinding every single day.",
            "link": "https://www.youtube.com/watch?v=abc123",
            "date": None,
        }
        body = f"{yt['title']} {yt['snippet']}"
        vlink = WebSearchValidatorResult(
            title=yt["title"],
            link=yt["link"],
            snippet=yt["snippet"],
            html_text=body,
            html_content=body,
        )
        synapse = WebSearchSynapse(
            query="how does docker work", results=[yt], validator_links=[vlink]
        )

        verified = self.model._snippet_verified(yt["snippet"], vlink)
        offtopic = self.model.check_response_random_link(synapse, {yt["link"]: 0.0})
        ontopic = self.model.check_response_random_link(synapse, {yt["link"]: 1.0})

        self.assertTrue(verified)
        self.assertEqual(offtopic, 0.0)
        self.assertEqual(ontopic, 1.0)


class SiteOperatorTestCase(unittest.IsolatedAsyncioTestCase):
    """`site:` gate on result host + stripping from the LLM relevance text."""

    def setUp(self):
        self.model = WebBasicSearchContentRelevanceModel(
            scoring_type=None, neuron=None, llm_reward=None
        )

    def test_in_domain_result_passes(self):
        # link1 host is www.python.org, a subdomain match for site:python.org.
        synapse = WebSearchSynapse(
            query="python site:python.org",
            results=[link1],
            validator_links=[_validator_link(link1)],
        )
        score = self.model.check_response_random_link(synapse, {link1["link"]: 1.0})
        self.assertEqual(score, 1.0)

    def test_out_of_domain_result_scores_zero(self):
        # link2 host is www.w3schools.com — not under python.org.
        synapse = WebSearchSynapse(
            query="python site:python.org",
            results=[link2],
            validator_links=[_validator_link(link2)],
        )
        score = self.model.check_response_random_link(synapse, {link2["link"]: 1.0})
        self.assertEqual(score, 0.0)

    async def test_site_operator_stripped_from_relevance_text(self):
        captured = {}

        class CapturingLLM:
            async def llm_processing(self, messages):
                for message in messages:
                    ((url, conversation),) = message.items()
                    captured[url] = conversation[1]["content"]
                return {url: "RELEVANT" for url in captured}

        self.model.reward_llm = CapturingLLM()
        synapse = WebSearchSynapse(
            query="machine learning site:python.org", results=[link1]
        )
        synapse.validator_links = [_validator_link(link1)]

        await self.model.llm_process_validator_links(synapse)

        user_content = captured[link1["link"]]
        self.assertIn("machine learning", user_content)
        self.assertNotIn("site:python.org", user_content)

    def test_parse_only_handles_site_operator(self):
        # Only site: is supported today; other tokens stay in the query text.
        operators = parse_web_query("filetype:pdf python site:docs.python.org")
        self.assertEqual(operators.sites, ["docs.python.org"])
        self.assertIn("filetype:pdf", operators.text)
        self.assertNotIn("site:", operators.text)


class WebSearchRelevancePromptTestCase(unittest.TestCase):
    def setUp(self):
        from neurons.validators.utils.prompts import WebSearchRelevancePrompt

        self.prompt = WebSearchRelevancePrompt()

    def test_relevant_scores_one(self):
        self.assertEqual(self.prompt.extract_score("RELEVANT"), 1.0)
        self.assertEqual(self.prompt.extract_score("Verdict: relevant"), 1.0)

    def test_irrelevant_scores_zero_despite_containing_relevant(self):
        # "IRRELEVANT" contains "RELEVANT" — must not be read as a pass.
        self.assertEqual(self.prompt.extract_score("IRRELEVANT"), 0.0)
        self.assertEqual(self.prompt.extract_score("This is not relevant"), 0.0)

    def test_empty_or_garbled_scores_zero(self):
        self.assertEqual(self.prompt.extract_score(""), 0.0)
        self.assertEqual(self.prompt.extract_score("unsure"), 0.0)


class WebBasicGetRewardsTestCase(unittest.IsolatedAsyncioTestCase):
    """End-to-end get_rewards with the page scrape and LLM stubbed (no network)."""

    def _model(self, verdicts):
        model = WebBasicSearchContentRelevanceModel(
            scoring_type=None, neuron=None, llm_reward=FakeRewardLLM(verdicts)
        )
        by_url = {link["link"]: link for link in ALL_LINKS}

        async def fake_scrape_links(urls):
            metadata = []
            for url in urls:
                link = by_url[url]
                metadata.append(
                    {
                        "title": link["title"],
                        "snippet": link["snippet"],
                        "link": link["link"],
                        "html_text": link["snippet"],
                        "html_content": link["snippet"],
                    }
                )
            return metadata, []

        model.scrape_links = fake_scrape_links
        return model

    async def test_process_links_populates_validator_links(self):
        model = self._model({})
        synapse = WebSearchSynapse(query="python", results=list(ALL_LINKS))

        await model.process_links([synapse])

        self.assertEqual(len(synapse.validator_links), WEB_LINK_SCRAPE_AMOUNT)
        result_urls = [r["link"] for r in synapse.results]
        self.assertTrue(all(v.link in result_urls for v in synapse.validator_links))

    async def test_relevant_links_score_full(self):
        model = self._model({link["link"]: "RELEVANT" for link in ALL_LINKS})
        rewards, grouped = await model.get_rewards(
            [
                WebSearchSynapse(query="python", results=[link1, link2]),
                WebSearchSynapse(query="python", results=[link3, link4, link5]),
            ],
            [1, 2],
        )
        self.assertEqual([r.reward for r in rewards], [1.0, 1.0])
        self.assertEqual(grouped, {1: 1.0, 2: 1.0})

    async def test_offtopic_links_score_zero_end_to_end(self):
        model = self._model({link["link"]: "IRRELEVANT" for link in ALL_LINKS})
        rewards, grouped = await model.get_rewards(
            [WebSearchSynapse(query="python", results=[link1, link2])], [7]
        )
        self.assertEqual(rewards[0].reward, 0.0)
        self.assertEqual(grouped, {7: 0.0})

    async def test_llm_judges_query_against_validator_scraped_text(self):
        model = self._model({link1["link"]: "RELEVANT", link2["link"]: "RELEVANT"})
        synapse = WebSearchSynapse(
            query="how does docker work", results=[link1, link2]
        )

        await model.get_rewards([synapse], [1])

        self.assertTrue(model.reward_llm.scored)
        scored_url, conversation = model.reward_llm.scored[0]
        user_content = conversation[1]["content"]
        self.assertIn("how does docker work", user_content)
        scraped_title = {link["link"]: link["title"] for link in ALL_LINKS}[scored_url]
        self.assertIn(scraped_title, user_content)


if __name__ == "__main__":
    unittest.main()
