"""Unit tests for neuralcore.utils.config -- ConfigLoader."""
import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestConfigLoader:
    """Test ConfigLoader parsing, getters, and resolution."""

    def _reset(self):
        import neuralcore.utils.config as cfg_mod
        cfg_mod.loader = None
        import neuralcore.clients.factory as cf_mod
        cf_mod._factory = None
        from neuralcore.utils.text_tokenizer import TextTokenizer
        TextTokenizer._instance = None
        TextTokenizer._initialized = False

    def test_load_from_file(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        assert isinstance(loader.config, dict)
        assert "clients" in loader.config
        assert "agents" in loader.config

    def test_load_from_dict(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        cfg = {
            "clients": {"main": {"type": "chat", "model": "test", "base_url": "http://localhost:9111/v1", "tokenizer": "data/tokenizer/tokenizer.json"}},
            "agents": {"a1": {"id": "a1", "name": "Test", "client": "main"}},
        }
        loader = ConfigLoader(app_root=PROJECT_ROOT)
        parsed = loader.parse_config(cfg)
        assert parsed["clients"]["main"]["model"] == "test"

    def test_get_client_config(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cc = loader.get_client_config("main")
        assert cc["model"] == "mock-model"
        assert cc["base_url"] == "http://127.0.0.1:9111/v1"

    def test_get_agent_config(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        ac = loader.get_agent_config("test_agent")
        assert ac["name"] == "Test Agent"
        assert ac["client"] == "main"

    def test_get_system_prompt(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        sp = loader.get_system_prompt()
        assert isinstance(sp, str)
        assert len(sp) > 0

    def test_get_logging_config(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        lc = loader.get_logging_config()
        assert "logging_enabled" in lc
        assert "log_level" in lc

    def test_get_tool_sets(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        ts = loader.get_tool_sets()
        assert "TestTools" in ts

    def test_missing_client_returns_empty(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        assert loader.get_client_config("nonexistent") == {}

    def test_resolve_secret_default(self):
        self._reset()
        from neuralcore.utils.config import ConfigLoader
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        # main client has api_key: "test-key" in test config
        secret = loader.resolve_secret("main")
        assert secret == "test-key"