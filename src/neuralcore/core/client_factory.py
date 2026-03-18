from neuralcore.core.client import LLMClient
from neuralcore.utils.llm_tools import InternalTools
from neuralcore.utils.config import ConfigLoader
from neuralcore.utils.text_tokenizer import TextTokenizer


class ClientFactory:
    """
    Builds LLM clients from ConfigLoader and can auto-register them as tools.
    Fully flexible: supports dynamic extra_body injection per client,
    optional thinking mode, and tool registration.
    """

    COMMON_FIELDS = ["presence_penalty", "top_k"]

    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        self.clients: dict[str, LLMClient] = {}

    def build(self) -> dict[str, LLMClient]:
        """Create all clients defined in the config and store in self.clients"""
        clients_cfg = self.loader.config.get("clients", {})
        for name in clients_cfg:
            self.clients[name] = self._create(name)
        return self.clients

    def _create(self, client_name: str) -> LLMClient:
        cfg = self.loader.get_client_config(client_name)
        api_key = self.loader.resolve_secret(client_name)
        model = cfg["model"]

        extra_body = {}

        # Merge common fields dynamically
        for field in self.COMMON_FIELDS:
            if field in cfg:
                extra_body[field] = cfg[field]

        # Thinking / chat_template_kwargs
        if cfg.get("enable_thinking") is not None:
            extra_body.setdefault("chat_template_kwargs", {})
            extra_body["chat_template_kwargs"]["enable_thinking"] = cfg[
                "enable_thinking"
            ]

        # Merge user-provided extra_body
        if user_extra := cfg.get("extra_body"):
            if not isinstance(user_extra, dict):
                raise ValueError(f"extra_body must be a dict for client {client_name}")
            extra_body.update(user_extra)

        # --- tokenizer handling ---
        tokenizer_cfg = cfg.get("tokenizer")
        tokenizer_obj = None
        if tokenizer_cfg:
            if isinstance(tokenizer_cfg, str):
                tokenizer_obj = TextTokenizer(tokenizer_cfg)
            else:
                tokenizer_obj = tokenizer_cfg

        # ✅ NEW: pass missing params
        temperature = cfg.get("temperature", 0.7)
        max_tokens = cfg.get("max_tokens", 4096)

        return LLMClient(
            base_url=cfg.get("base_url", "http://localhost:1212/v1"),
            model=model,
            tokenizer=tokenizer_obj,
            api_key=api_key,
            extra_body=extra_body or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def register_tool_clients(self, registry, main_client: LLMClient):
        """
        Register clients marked with `register_as_tool: true` as InternalTools.
        """
        clients_cfg = self.loader.config.get("clients", {})
        for name, cfg in clients_cfg.items():
            if not cfg.get("register_as_tool"):
                continue

            client = self.clients[name]

            tools = InternalTools(
                client=client,
                description=cfg.get("description", name),
                methods=[client.ask, client.stream_chat, client.chat],
            )

            tool_name = cfg.get("tool_name", name)
            registry.register_set(tool_name, tools.as_action_set(tool_name))
