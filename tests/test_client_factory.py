"""Unit tests for neuralcore.clients.factory -- ClientFactory."""
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _reset():
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


class TestClientFactory:
    def test_build_clients(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cfg_mod.loader = loader

        from neuralcore.clients.factory import ClientFactory
        factory = ClientFactory(loader)
        clients = factory.build()
        assert "main" in clients

    def test_main_client_type(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cfg_mod.loader = loader

        from neuralcore.clients.factory import ClientFactory
        from neuralcore.clients.client import LLMClient
        factory = ClientFactory(loader)
        clients = factory.build()
        assert isinstance(clients["main"], LLMClient)

    def test_client_model(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cfg_mod.loader = loader

        from neuralcore.clients.factory import ClientFactory
        factory = ClientFactory(loader)
        clients = factory.build()
        assert clients["main"].model == "mock-model"

    def test_client_base_url(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cfg_mod.loader = loader

        from neuralcore.clients.factory import ClientFactory
        factory = ClientFactory(loader)
        clients = factory.build()
        assert "9111" in clients["main"].base_url

    def test_get_clients_singleton(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        loader = ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        cfg_mod.loader = loader

        from neuralcore.clients.factory import get_clients
        c1 = get_clients()
        c2 = get_clients()
        assert c1 is c2

    def test_missing_client_raises(self, mock_server):
        _reset()
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as cfg_mod
        # Config with nonexistent client ref
        cfg = {
            "clients": {"main": {
                "type": "chat",
                "model": "test",
                "base_url": "http://127.0.0.1:9111/v1",
                "tokenizer": "data/tokenizer/tokenizer.json",
            }},
            "agents": {"a1": {"id": "a1", "name": "Test", "client": "nonexistent"}},
        }
        loader = ConfigLoader(app_root=PROJECT_ROOT)
        loader.config = loader.parse_config(cfg)
        cfg_mod.loader = loader

        from neuralcore.clients.factory import ClientFactory
        factory = ClientFactory(loader)
        factory.build()

        with pytest.raises(ValueError, match="not found"):
            loader.create_agent(agent_id="a1")