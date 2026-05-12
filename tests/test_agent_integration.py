"""Integration tests for the Agent using mock LLM server."""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

TOKENIZER_PATH = str(PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json")


def _reset_all():
    import neuralcore.utils.config as cfg_mod

    cfg_mod.loader = None
    import neuralcore.clients.factory as cf_mod

    cf_mod._factory = None
    from neuralcore.utils.text_tokenizer import TextTokenizer

    TextTokenizer._instance = None
    TextTokenizer._initialized = False
    from neuralcore.actions.registry import registry

    registry.sets.clear()
    registry.all_actions.clear()
    registry._index.clear()


def _create_agent(mock_server):
    """Helper to create a fully wired agent."""
    _reset_all()
    from neuralcore.utils.config import ConfigLoader
    import neuralcore.utils.config as cfg_mod

    loader = ConfigLoader(
        cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
        app_root=PROJECT_ROOT,
    )
    cfg_mod.loader = loader

    from neuralcore.utils.logger import Logger

    Logger._logger = None
    Logger._config = None
    Logger.get_logger()

    from neuralcore.clients.factory import ClientFactory
    import neuralcore.clients.factory as cf_mod

    cf_mod._factory = None
    factory = ClientFactory(loader)
    factory.build()
    cf_mod._factory = factory

    # Force reload of test_toolset to re-register decorators
    import importlib

    if "test_toolset" in sys.modules:
        importlib.reload(sys.modules["test_toolset"])
    loader.load_tool_sets(sets_to_load=["TestTools"])
    agent = loader.create_agent(agent_id="test_agent")
    agent.attach_tools()
    return agent


@pytest.mark.asyncio(loop_scope="session")
class TestAgentCreation:
    async def test_agent_creation(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.client is not None

    async def test_agent_state_initialization(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.state.status == "idle"
        assert agent.state.loop_count == 0

    async def test_agent_config(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.max_iterations == 10
        assert agent.temperature == 0.7

    async def test_agent_has_tools_in_registry(self, mock_server):
        agent = _create_agent(mock_server)
        # Tools are registered in the global registry
        from neuralcore.actions.registry import registry

        assert len(registry.all_actions) > 0


@pytest.mark.asyncio(loop_scope="session")
class TestAgentChat:
    async def test_simple_chat(self, mock_server):
        agent = _create_agent(mock_server)
        result = await agent.client.chat("Hello from test")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_streaming_chat(self, mock_server):
        agent = _create_agent(mock_server)
        queue = await agent.client.stream_chat("Stream test")
        chunks = []
        while True:
            item = await queue.get()
            if item is None:
                break
            chunks.append(item)
        assert len("".join(chunks)) > 0

    async def test_context_manager_exists(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.context_manager is not None

    async def test_task_manager_exists(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.task_manager is not None

    async def test_action_manager_exists(self, mock_server):
        agent = _create_agent(mock_server)
        assert agent.action_manager is not None


@pytest.mark.asyncio(loop_scope="session")
class TestAgentMessaging:
    async def test_add_message(self, mock_server):
        agent = _create_agent(mock_server)
        await agent.add_message("user", "Test message")
        # Message should be in context manager

    async def test_post_message(self, mock_server):
        agent = _create_agent(mock_server)
        await agent.post_message("User question")
        assert not agent.message_queue.empty()

    async def test_get_detailed_status(self, mock_server):
        agent = _create_agent(mock_server)
        status = agent.get_detailed_status()
        assert "agent_id" in status
        assert "status" in status
        assert status["agent_id"] == "test_agent"

    async def test_get_full_state_dict(self, mock_server):
        agent = _create_agent(mock_server)
        state = agent.get_full_state_dict()
        assert "agent_id" in state
        assert "state" in state
        assert "loaded_tools" in state


@pytest.mark.asyncio(loop_scope="session")
class TestAgentToolExecution:
    async def test_tool_call_via_client(self, mock_server):
        agent = _create_agent(mock_server)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo_tool",
                    "description": "Echo input",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            }
        ]
        result = await agent.client.call_tools(
            messages=[{"role": "user", "content": "Read the test file"}],
            tools=tools,
        )
        assert result is not None
