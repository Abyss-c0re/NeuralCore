"""Unit tests for neuralcore.clients.client -- LLMClient."""
import sys
import json
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

TOKENIZER_PATH = str(PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json")


def _reset():
    import neuralcore.utils.config as cfg_mod
    cfg_mod.loader = None
    import neuralcore.clients.factory as cf_mod
    cf_mod._factory = None
    from neuralcore.utils.text_tokenizer import TextTokenizer
    TextTokenizer._instance = None
    TextTokenizer._initialized = False


@pytest.mark.asyncio(loop_scope="session")
class TestLLMClient:
    """Test the LLMClient against the mock server."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_client(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.clients.client import LLMClient
        self.client = LLMClient(
            base_url=mock_server.base_url,
            model="mock-model",
            name="test",
            tokenizer=TOKENIZER_PATH,
            api_key="test-key",
        )

    async def test_non_streaming_chat(self):
        result = await self.client.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_non_streaming_chat_with_string(self):
        result = await self.client.chat("Tell me something")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_stream_chat(self):
        queue = await self.client.stream_chat(
            messages=[{"role": "user", "content": "Stream test"}],
        )
        chunks = []
        while True:
            item = await queue.get()
            if item is None:
                break
            chunks.append(item)
        full = "".join(chunks)
        assert len(full) > 0

    async def test_call_tools(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "Echo input",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"]
                }
            }
        }]
        result = await self.client.call_tools(
            messages=[{"role": "user", "content": "Read this file for me"}],
            tools=tools,
        )
        assert result is not None
        assert len(result) > 0
        assert result[0].function.name == "echo_tool"

    async def test_ask(self):
        result = await self.client.ask("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_stop_stream(self):
        queue = await self.client.stream_chat(
            messages=[{"role": "user", "content": "Long message to stream"}],
        )
        # Stop immediately
        stopped = self.client.stop_stream()
        assert stopped is True
        # Drain to avoid leaks
        while True:
            item = await queue.get()
            if item is None:
                break

    async def test_fetch_embedding(self):
        result = await self.client.fetch_embedding("Test embedding text")
        assert result is not None
        assert len(result) > 0

    async def test_chat_error_handling(self):
        """Test that a bad base_url returns an error string."""
        from neuralcore.clients.client import LLMClient
        bad_client = LLMClient(
            base_url="http://127.0.0.1:1/v1",
            model="nonexistent",
            name="bad",
            tokenizer=TOKENIZER_PATH,
            api_key="none",
        )
        result = await bad_client.chat("Hello")
        assert "[error]" in result.lower()