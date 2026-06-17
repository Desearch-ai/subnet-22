import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from desearch import utils
from desearch.protocol import ScoringModel


class CallScoringLLMRoutingTest(unittest.IsolatedAsyncioTestCase):
    @patch.object(utils, "call_chutes", new_callable=AsyncMock)
    @patch.object(utils, "call_openai", new_callable=AsyncMock)
    async def test_openai_model_routes_to_openai(self, mock_openai, mock_chutes):
        mock_openai.return_value = "Verdict: HIGH"
        out = await utils.call_scoring_llm([{"role": "user", "content": "x"}], ScoringModel.OPENAI_GPT4_1_NANO)
        self.assertEqual(out, "Verdict: HIGH")
        mock_openai.assert_awaited_once()
        mock_chutes.assert_not_called()

    @patch.object(utils, "call_chutes", new_callable=AsyncMock)
    @patch.object(utils, "call_openai", new_callable=AsyncMock)
    async def test_chutes_model_routes_to_chutes(self, mock_openai, mock_chutes):
        mock_chutes.return_value = "Verdict: OFFTOPIC"
        out = await utils.call_scoring_llm([{"role": "user", "content": "x"}], ScoringModel.QWEN3_32B)
        self.assertEqual(out, "Verdict: OFFTOPIC")
        mock_chutes.assert_awaited_once()
        mock_openai.assert_not_called()

    @patch.object(utils, "call_chutes", new_callable=AsyncMock)
    @patch.object(utils, "call_openai", new_callable=AsyncMock)
    async def test_falls_back_to_openai_when_chutes_unavailable(self, mock_openai, mock_chutes):
        mock_chutes.return_value = None  # provider outage
        mock_openai.return_value = "Verdict: MEDIUM"
        out = await utils.call_scoring_llm([{"role": "user", "content": "x"}], ScoringModel.QWEN3_32B)
        self.assertEqual(out, "Verdict: MEDIUM")
        mock_chutes.assert_awaited_once()
        mock_openai.assert_awaited_once()


class ChutesApiKeyTest(unittest.TestCase):
    def test_reads_token_then_key(self):
        with patch.dict("os.environ", {"CHUTES_API_TOKEN": "tok", "CHUTES_API_KEY": "key"}, clear=True):
            self.assertEqual(utils.get_chutes_api_key(), "tok")
        with patch.dict("os.environ", {"CHUTES_API_KEY": "key"}, clear=True):
            self.assertEqual(utils.get_chutes_api_key(), "key")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(utils.get_chutes_api_key(), "")


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return "err"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Captures the JSON payload sent to Chutes."""
    captured = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, headers=None, json=None, timeout=None):
        _FakeSession.captured = json
        return _FakeResp(200, {"choices": [{"message": {"content": "Verdict: HIGH"}}]})


class CallChutesPayloadTest(unittest.IsolatedAsyncioTestCase):
    @patch.dict("os.environ", {"CHUTES_API_TOKEN": "tok"}, clear=True)
    @patch("desearch.utils.aiohttp.ClientSession", _FakeSession)
    async def test_payload_disables_thinking_and_uses_model_value(self):
        out = await utils.call_chutes(
            [{"role": "user", "content": "x"}], temperature=0.0001, model=ScoringModel.QWEN3_32B
        )
        self.assertEqual(out, "Verdict: HIGH")
        payload = _FakeSession.captured
        self.assertEqual(payload["model"], "Qwen/Qwen3-32B-TEE")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("response_format", payload)  # omitted when None


if __name__ == "__main__":
    unittest.main()
