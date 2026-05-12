"""
Shared pytest fixtures for NeuralCore tests.

Provides:
- mock_server: running MockLLMServer instance (session-scoped)
- config_loader: ConfigLoader pointed at test config
- agent: fully wired Agent instance
- event_loop: shared asyncio event loop
"""
import sys
import os
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

os.chdir(PROJECT_ROOT)

from mock_llm_server import MockLLMServer


# Use a single event loop for the entire test session
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def mock_server():
    """Start the mock LLM server for the entire test session."""
    server = MockLLMServer(host="127.0.0.1", port=9111)
    await server.start()
    yield server
    await server.stop()


def _reset_singletons():
    """Reset all NeuralCore singletons so each test config load is clean."""
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


@pytest.fixture
def config_loader(mock_server):
    """Create a ConfigLoader pointing at the test config."""
    _reset_singletons()
    from neuralcore.utils.config import ConfigLoader
    loader = ConfigLoader(
        cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
        app_root=PROJECT_ROOT,
    )
    return loader


@pytest.fixture
def logger_setup(config_loader):
    """Initialize logger from test config."""
    from neuralcore.utils.logger import Logger
    Logger._logger = None
    Logger._config = None
    return Logger.get_logger()


@pytest_asyncio.fixture
async def agent(config_loader, logger_setup):
    """Create a fully wired Agent instance using the test config."""
    # Build clients via factory (uses the singleton ConfigLoader)
    import neuralcore.utils.config as cfg_mod
    cfg_mod.loader = config_loader

    from neuralcore.clients.factory import ClientFactory
    import neuralcore.clients.factory as cf_mod
    cf_mod._factory = None
    factory = ClientFactory(config_loader)
    clients = factory.build()
    cf_mod._factory = factory

    # Load test tools
    config_loader.load_tool_sets(sets_to_load=["TestTools"])

    # Create agent
    ag = config_loader.create_agent(agent_id="test_agent")
    ag.attach_tools()
    yield ag