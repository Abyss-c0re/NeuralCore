import os
import sys
import yaml
import importlib
import copy
from typing import Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    UNIVERSAL CONFIG PARSER for NeuralCore
    - Accepts: file (Path/str), raw YAML string, or dict (live generated/edited configs).
    """

    DEFAULT_API_KEY = "not-needed"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, cli_path: str | None = None, app_root: Path | None = None):
        self.app_root = app_root or Path.cwd()
        # Universal parse on init — keeps old behavior but now supports in-memory too
        self.config: dict[str, Any] = self.parse_config(cli_path)

        self.agent_factory = None  # created on first use
        self.workflow_factory = None

        print(
            f"[DEBUG] ConfigLoader initialized with {len(self.config)} top-level keys"
        )

    # ===================================================================
    # UNIVERSAL PARSER
    # ===================================================================
    def parse_config(self, source: str | Path | dict | None = None) -> dict[str, Any]:
        """
        One method to rule them all.
        - str/Path → file (classic)
        - dict → in-memory
        - raw YAML str → direct from VR canvas
        Returns normalized config dict.
        """
        if isinstance(source, (str, Path)):
            # Safe conversion for _resolve_config_path
            path_input = source if isinstance(source, str) else str(source)
            path = self._resolve_config_path(path_input)
            raw = self._load_yaml(path)
            print(f"[INFO] Loaded config from file: {path}")
        elif isinstance(source, dict):
            print("[INFO] Using live in-memory dict")
            raw = copy.deepcopy(source)
        elif isinstance(source, str) and source.strip().startswith(("---", "{")):
            print("[INFO] Parsing raw YAML/JSON string from editor")
            raw = yaml.safe_load(source) or {}
        else:
            # Classic fallback (keeps __init__ and old calls 100% compatible)
            path = self._resolve_config_path(None)
            raw = self._load_yaml(path)

        if not isinstance(raw, dict):
            print("[WARNING] Config root is not a dict — using empty")
            return {}

        # Simple secret resolution (inline, no extra methods)
        for client_name, client_cfg in raw.get("clients", {}).items():
            if isinstance(client_cfg, dict):
                if env_key := client_cfg.pop("api_key_env", None):
                    val = os.getenv(env_key)
                    client_cfg["api_key"] = val if val else self.DEFAULT_API_KEY
                    print(f"[DEBUG] Resolved secret for client '{client_name}'")

        return raw

    # ===================================================================
    # UPDATED GETTERS — safe for live configs
    # ===================================================================
    def get_client_config(self, client_name: str) -> dict:
        return self.config.get("clients", {}).get(client_name, {})

    def resolve_secret(self, client_name: str) -> str:
        cfg = self.get_client_config(client_name)
        return cfg.get("api_key", self.DEFAULT_API_KEY)

    def get_system_prompt(self) -> str:
        return self.config.get(
            "system_prompt",
            "You are terminal chat assistant, use provided tools if needed",
        )

    def get_agent_config(self, agent_name: str) -> dict:
        return self.config.get("agents", {}).get(agent_name, {})

    def get_app_config(self) -> dict:
        return self.config.get("app", {})

    def get_logging_config(self) -> dict:
        app_cfg = self.get_app_config()
        log_dir_default = Path.home() / ".neuralcore"
        log_dir_default.mkdir(parents=True, exist_ok=True)
        log_file = app_cfg.get("log_file", log_dir_default / "neuralcore.log")
        if isinstance(log_file, str):
            log_file = Path(os.path.expanduser(log_file))
        return {
            "logging_enabled": app_cfg.get("logging_enabled", True),
            "log_level": app_cfg.get("log_level", "info"),
            "log_to_file": app_cfg.get("log_to_file", True),
            "log_to_ui": app_cfg.get("log_to_ui", False),
            "log_file": log_file,
        }

    def get_tool_sets(self) -> dict:
        return self.config.get("tools", {})

    def get_workflow_sets(self, client_name: str | None = None) -> dict:
        if client_name:
            client_cfg = self.get_client_config(client_name)
            return client_cfg.get("workflow_sets", {})
        return self.config.get("workflows", {})

    def get_loop_config(self, loop_name: str) -> dict:
        """Get loop configuration from YAML, supporting live overrides."""
        return self.config.get("loops", {}).get(loop_name, {})

    # ===================================================================
    # LOADERS — now support live override dicts
    # ===================================================================
    def load_tool_sets(
        self, sets_to_load: list[str] | None = None, config_override: dict | None = None
    ):

        sets_cfg = (config_override or self.config).get("tools", {})

        print(f"[DEBUG] Loading tool sets: {sets_to_load or 'ALL'}")

        for set_name, cfg in sets_cfg.items():
            if sets_to_load and set_name not in sets_to_load:
                continue

            # ── 1. Determine folder (exactly your original logic) ─────
            folder = cfg.get("folder") if isinstance(cfg, dict) else None
            if folder:
                folder_path = Path(folder).expanduser().resolve()
            else:
                folder_path = self.app_root / "tools" / set_name
                if not folder_path.exists() or not folder_path.is_dir():
                    folder_path = self.app_root / "tools"

            if not folder_path.exists() or not folder_path.is_dir():
                print(
                    f"[WARNING] Tool folder '{folder_path}' does not exist for set '{set_name}'"
                )
                continue

            # ── 2. Modules (exactly your original) ───────────────────
            modules_to_load = cfg.get("modules", []) if isinstance(cfg, dict) else []

            sys.path.insert(0, str(folder_path))
            imported_any = False

            if modules_to_load:
                for mod_name in modules_to_load:
                    try:
                        importlib.import_module(mod_name)
                        print(f"[INFO] Imported '{mod_name}' for set '{set_name}'")
                        imported_any = True
                    except Exception as e:
                        print(
                            f"[ERROR] Failed to import {mod_name} for set '{set_name}': {e}"
                        )
            else:
                for py_file in folder_path.glob("*.py"):
                    if py_file.name == "__init__.py":
                        continue
                    try:
                        importlib.import_module(py_file.stem)
                        print(f"[INFO] Imported '{py_file.name}' for set '{set_name}'")
                        imported_any = True
                    except Exception as e:
                        print(f"[ERROR] Failed to import {py_file.name}: {e}")

            sys.path.pop(0)

            # Registry check (your original)
            try:
                from neuralcore.actions.registry import registry

                if imported_any and getattr(registry, "sets", {}).get(set_name):
                    print(f"[INFO] Tool set '{set_name}' registered successfully")
            except ImportError:
                pass

    # ===================================================================
    # HIGH-LEVEL CREATORS FOR EXTERNAL MANAGMENT
    # ===================================================================
    def create_agent(
        self,
        config_source: str | Path | dict | None = None,
        agent_id: str | None = None,
    ):
        """
        Main entry point for creating agents.
        """
        if config_source is not None:
            self.config = self.parse_config(config_source)

        if not agent_id:
            agents = self.config.get("agents", {})
            agent_id = next(iter(agents.keys()), None)
            if not agent_id:
                raise ValueError("No agents found in config")

        agent_cfg = self.get_agent_config(agent_id)
        if not agent_cfg:
            raise ValueError(f"No agent config found for '{agent_id}'")

        if self.agent_factory is None:
            from neuralcore.agents.factory import AgentFactory

            self.agent_factory = AgentFactory(self)

        return self.agent_factory.create_agent(
            agent_id=agent_id,
            config=agent_cfg,
            app_root=self.app_root,
        )

    def create_workflow_engine(self, agent):
        """Create WorkflowEngine using factory (clean separation)."""
        if self.workflow_factory is None:
            from neuralcore.workflows.factory import WorkflowFactory

            self.workflow_factory = WorkflowFactory()

        # Register workflows from config into factory
        for name, wf_data in self.config.get("workflows", {}).items():
            self.workflow_factory.register_workflow(
                name=name,
                description=wf_data.get("description", f"Workflow {name}"),
                steps=wf_data.get("steps", []),
                hidden_toolsets=wf_data.get("hidden_toolsets"),
            )

        from neuralcore.workflows.engine import WorkflowEngine

        return WorkflowEngine(agent=agent, workflow_registry=self.workflow_factory)

    # ===================================================================
    # LEGACY METHODS — now powered by universal parser
    # ===================================================================
    def _resolve_config_path(
        self, cli_path: str | Path | None = None
    ) -> Optional[Path]:
        # Normalize Path to str
        if isinstance(cli_path, Path):
            cli_path = str(cli_path)

        # 1️⃣ CLI argument / explicit path
        if cli_path:
            path = Path(cli_path).expanduser().resolve()
            print(f"[DEBUG] Using config path: {path}")
            return path

        # 2️⃣ ENV variable
        if env_path := os.getenv("NEURALCORE_CONFIG"):
            path = Path(env_path).expanduser().resolve()
            print(f"[DEBUG] Using config from NEURALCORE_CONFIG: {path}")
            return path

        # 3️⃣ Project root
        local_path = (self.app_root / "config.yaml").resolve()
        if local_path.exists():
            print(f"[DEBUG] Using project config: {local_path}")
            return local_path

        # 4️⃣ Global fallback
        global_path = Path.home() / ".neuralcore" / "config.yaml"
        if global_path.exists():
            print(f"[DEBUG] Using global config: {global_path}")
            return global_path

        print("[WARNING] No config file found")
        return None

    def _load_yaml(self, path: Optional[Path]) -> dict[str, Any]:
        if path is None or not path.exists():
            print(f"[DEBUG] No YAML found at path: {path}")
            return {}

        print(f"[DEBUG] Loading YAML from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f) or {}

        if isinstance(data, dict):
            return data
        else:
            print(f"[WARNING] Expected dict from YAML, got {type(data).__name__}")
            return {}


# --------------------------
# Singleton
# --------------------------
loader: Optional[ConfigLoader] = None


def get_loader(
    cli_path: str | None = None, app_root: Path | None = None
) -> ConfigLoader:
    global loader
    if loader is None:
        loader = ConfigLoader(cli_path, app_root)
    return loader
