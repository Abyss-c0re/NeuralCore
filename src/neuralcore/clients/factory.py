import inspect
from typing import Dict, Any

from neuralcore.clients.client import LLMClient
from neuralcore.actions.actions import Action, ActionSet

from neuralcore.utils.config import ConfigLoader, get_loader
from neuralcore.actions.manager import registry


class ClientFactory:
    """
    Builds LLM clients from ConfigLoader with support for:
    - multiple client types (chat, embeddings, etc.)
    - dynamic extra_body injection
    - optional thinking mode
    - tool registration
    """

    COMMON_FIELDS = ["presence_penalty", "top_k"]

    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        self.clients: Dict[str, Any] = {}

    def build(self) -> Dict[str, Any]:
        """Create all clients defined in the config"""
        clients_cfg = self.loader.config.get("clients", {})

        for name in clients_cfg:
            self.clients[name] = self._create(name)

        self.register_tool_clients()

        return self.clients

    def _create(self, client_name: str):
        cfg = self.loader.get_client_config(client_name)
        client_type = cfg.get("type", "chat")  # default = chat

        if client_type == "chat":
            return self._create_chat_client(client_name, cfg)

        elif client_type == "embeddings":
            return self._create_embedding_client(client_name, cfg)

        else:
            raise ValueError(
                f"Unknown client type '{client_type}' for client '{client_name}'"
            )

    # ----------------------
    # Chat client
    # ----------------------
    def _create_chat_client(self, client_name: str, cfg: dict) -> LLMClient:
        api_key = self.loader.resolve_secret(client_name)
        model = cfg["model"]

        extra_body = {}

        # Common fields
        for field in self.COMMON_FIELDS:
            if field in cfg:
                extra_body[field] = cfg[field]

        # Thinking support
        if cfg.get("enable_thinking") is not None:
            extra_body.setdefault("chat_template_kwargs", {})
            extra_body["chat_template_kwargs"]["enable_thinking"] = cfg[
                "enable_thinking"
            ]

        # Merge user extra_body
        if user_extra := cfg.get("extra_body"):
            if not isinstance(user_extra, dict):
                raise ValueError(f"extra_body must be a dict for client {client_name}")
            extra_body.update(user_extra)

        return LLMClient(
            base_url=cfg.get("base_url", "http://localhost:1212/v1"),
            model=model,
            name=client_name,
            api_key=api_key,
            extra_body=extra_body or None,
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 4096),
            system_prompt=cfg.get("system_prompt") or self.loader.get_system_prompt(),
        )

    # ----------------------
    # Embedding client (placeholder-friendly)
    # ----------------------
    def _create_embedding_client(self, client_name: str, cfg: dict):
        """
        Embedding clients may not use full chat interface.
        You can replace this with a dedicated EmbeddingClient later.
        For now, we reuse LLMClient but restrict usage.
        """
        api_key = self.loader.resolve_secret(client_name)

        return LLMClient(
            base_url=cfg.get("base_url", "http://localhost:1212/v1"),
            model=cfg["model"],
            name=client_name,
            api_key=api_key,
            extra_body=None,
            temperature=0.0,
            max_tokens=cfg.get("max_tokens", 2048),
        )

    # ----------------------
    # Tool registration
    def register_tool_clients(self):
        """
        Register clients marked with `register_as_tool: true` as tools.
        Supports method-level overrides from config.
        """
        clients_cfg = self.loader.config.get("clients", {})

        for name, cfg in clients_cfg.items():
            if not cfg.get("register_as_tool"):
                continue

            client = self.clients.get(name)
            if not client:
                continue

            if cfg.get("type", "chat") != "chat":
                continue

            toolset_name = cfg.get("tool_name", name)
            action_set = ActionSet(name=toolset_name)
            action_set.description = cfg.get("description", f"Client {name}")

            methods_cfg = cfg.get("methods")

            # 🔹 If no methods specified → fallback defaults
            if not methods_cfg:
                methods_cfg = [
                    {"target": "ask"},
                    {"target": "chat"},
                ]

            for mcfg in methods_cfg:
                method_name = mcfg["target"]
                method = getattr(client, method_name, None)

                if not method:
                    continue

                sig = inspect.signature(method)

                parameters = {}
                required = []

                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue

                    parameters[pname] = {
                        "type": "string",
                        "description": f"{pname} parameter",
                    }

                    if param.default is inspect.Parameter.empty:
                        required.append(pname)
                    else:
                        parameters[pname]["default"] = param.default

                # 🔥 SAFE EXECUTOR (fixes your kwargs bug)
                async def executor_wrapper(_method=method, _sig=sig, **kwargs):
                    filtered = {k: v for k, v in kwargs.items() if k in _sig.parameters}
                    result = _method(**filtered)
                    if inspect.iscoroutine(result):
                        result = await result
                    return result

                action = Action(
                    name=mcfg.get("name", method_name),
                    description=mcfg.get(
                        "description", f"{method_name} via client '{name}'"
                    ),
                    parameters=parameters,
                    required=required,
                    executor=executor_wrapper,
                    tags=["client", name, method_name],
                    action_type="tool",
                )

                action_set.add(action)

            registry.register_set(toolset_name, action_set)


# --------------------------
# Singleton instance
# --------------------------
_factory: ClientFactory | None = None


def get_client_factory() -> ClientFactory:
    global _factory
    if _factory is None:
        loader = get_loader()
        _factory = ClientFactory(loader)
        _factory.build()
    return _factory


def get_clients() -> Dict[str, Any]:
    """Shortcut to access built clients directly"""
    return get_client_factory().clients
