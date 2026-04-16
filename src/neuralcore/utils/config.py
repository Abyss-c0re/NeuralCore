import os
import sys
import yaml
import importlib
import copy
from typing import Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    UNIVERSAL CONFIG PARSER for NeuralCore — the single brain for NeuralHub + NeuralLabs.
    - Accepts: file (Path/str), raw YAML string, or dict (live generated/edited configs).
    - Removes ALL duplication with Agent._load_agent_config and WorkflowEngine.
    - Zero breakage on existing CLI/singleton/file flows.
    - Prepares core for external management + live visualization/editing.
    """

    DEFAULT_API_KEY = "not-needed"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, cli_path: str | None = None, app_root: Path | None = None):
        self.app_root = app_root or Path.cwd()
        # Universal parse on init — keeps old behavior but now supports in-memory too
        self.config: dict[str, Any] = self.parse_config(cli_path)

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
        - dict → in-memory (NeuralHub deploy, NeuralLabs live edit)
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
            print("[INFO] Using live in-memory dict (NeuralHub/NeuralLabs mode)")
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
        """
        Now accepts optional config_override for live editing in NeuralLabs.
        """
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

    def load_workflow_sets(self, engine, config_override: dict | None = None):
        """
        Now supports live workflow dict injection — perfect for VR canvas edits.
        """
        global_workflows = (config_override or self.config).get("workflows", {})
        agent_cfg = getattr(engine.agent, "config", {})
        agent_workflow_cfg = agent_cfg.get("workflow")

        resolved = {}
        primary_workflow_name = None

        if isinstance(agent_workflow_cfg, str):
            workflow_name = agent_workflow_cfg.strip()
            primary_workflow_name = workflow_name
            if workflow_name in global_workflows:
                resolved[workflow_name] = global_workflows[workflow_name]
            else:
                resolved[workflow_name] = {"name": workflow_name}
        elif isinstance(agent_workflow_cfg, dict):
            # backward compat
            for wf_key, wf_ref in agent_workflow_cfg.items():
                if isinstance(wf_ref, dict) and "name" in wf_ref:
                    wf_name = wf_ref["name"]
                    resolved[wf_key] = global_workflows.get(wf_name, wf_ref)
                else:
                    resolved[wf_key] = wf_ref
            if resolved:
                primary_workflow_name = next(iter(resolved.keys()))
        else:
            primary_workflow_name = "default"
            resolved["default"] = global_workflows.get("default", {})

        engine.agent.config["workflow"] = resolved

        if primary_workflow_name and primary_workflow_name in global_workflows:
            wf_data = global_workflows[primary_workflow_name]
            engine.current_workflow_name = primary_workflow_name
            engine.workflow_description = wf_data.get("description", "No description")
            steps = wf_data.get("steps", getattr(engine, "DEFAULT_WORKFLOW", []))
            engine.workflow_steps = engine._resolve_steps(steps)
            print(
                f"[DEBUG] Using workflow '{primary_workflow_name}' for agent '{engine.agent.name}'"
            )
        else:
            engine.current_workflow_name = "default"
            engine.workflow_description = "Default workflow"
            engine.workflow_steps = engine._resolve_steps(
                getattr(engine, "DEFAULT_WORKFLOW", [])
            )
            print(
                f"[DEBUG] Fallback to default workflow for agent '{engine.agent.name}'"
            )

        print("[DEBUG] Single-config mode: workflow sets loaded")

    # ===================================================================
    # HIGH-LEVEL CREATORS FOR EXTERNAL MANAGMENT
    # ===================================================================
    def create_agent(
        self,
        config_source: str | Path | dict | None = None,
        agent_id: str | None = None,
    ):
        """
        NeuralHub's favorite method: spin up an agent from ANY config source.
        """
        if config_source is not None:
            self.config = self.parse_config(config_source)

        if not agent_id:
            agents = self.config.get("agents", {})
            agent_id = next(iter(agents.keys()), None)
            if not agent_id:
                raise ValueError("No agents found in config")

        return self.load_agent_from_config(agent_id)

    def parse_agent_config_for_labs(self, agent_id_or_dict: str | dict) -> dict:
        """
        Returns config enriched for NeuralLabs visualization/editing.
        Call this from VR canvas → instant node graph.
        """
        if isinstance(agent_id_or_dict, dict):
            cfg = copy.deepcopy(agent_id_or_dict)
        else:
            cfg = self.get_agent_config(agent_id_or_dict)

        cfg["_neural_labs"] = {
            "type": "agent",
            "can_have_sub_agents": cfg.get("max_sub_agents", 6) > 0,
            "workflow_count": len(self.config.get("workflows", {})),
            "tool_sets": list(self.get_tool_sets().keys()),
        }
        return cfg

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

    def load_agent_from_config(self, agent_id: str):
        app_root = self.app_root
        agent_cfg = self.get_agent_config(agent_id)
        if not agent_cfg:
            raise ValueError(f"No agent config found for '{agent_id}'")

        from neuralcore.clients.factory import get_clients

        clients = get_clients()
        client_name = agent_cfg.get("client")
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        client = clients[client_name]

        from neuralcore.agents.core import Agent

        agent = Agent(
            agent_id=agent_cfg.get("id", agent_id),
            loader=self,
            app_root=app_root,
        )
        agent.client = client
        agent.name = agent_cfg.get("name", f"Agent-{agent_id}")
        agent.description = agent_cfg.get("description", "")
        agent.max_iterations = agent_cfg.get("max_iterations", 25)
        agent.max_reflections = agent_cfg.get("max_reflections", 4)
        agent.temperature = agent_cfg.get("temperature", 0.3)
        agent.max_tokens = agent_cfg.get("max_tokens", 12048)
        agent.system_prompt = agent_cfg.get(
            "system_prompt", getattr(client, "system_prompt", "")
        )

        tools_to_load = agent_cfg.get("tool_sets", [])
        if tools_to_load:
            self.load_tool_sets(sets_to_load=tools_to_load)

        print(f"[INFO] Agent '{agent.name}' created from config")
        return agent


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
