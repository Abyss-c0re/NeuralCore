import os
import sys
import yaml
import importlib
from pathlib import Path


class ConfigLoader:
    """
    Loads config from YAML + ENV hybrid with secret resolution.
    Provides defaults if nothing is specified.
    Also automatically loads tool sets and workflow sets.
    """

    DEFAULT_API_KEY = "not-needed"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, cli_path: str | None = None, app_root: Path | None = None):
        self.app_root = app_root or Path.cwd()
        self.config_path = self._resolve_config_path(cli_path)
        self.config = self._load_yaml(self.config_path)

        # Automatically load tools and workflows

    # --------------------------
    # Config resolution
    # --------------------------
    def _resolve_config_path(self, cli_path: str | None) -> Path | None:
        # 1️⃣ CLI argument
        if cli_path:
            path = Path(cli_path).expanduser().resolve()
            print(f"[DEBUG] Using CLI config path: {path}")
            return path

        # 2️⃣ ENV variable
        if env_path := os.getenv("NEURALCORE_CONFIG"):
            path = Path(env_path).expanduser().resolve()
            print(f"[DEBUG] Using config from NEURALCORE_CONFIG: {path}")
            return path

        # 3️⃣ Project root (use app_root!)
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

    def _load_yaml(self, path: Path | None) -> dict:
        if path is None or not path.exists():
            print(f"[DEBUG] No YAML found at path: {path}")
            return {}
        print(f"[DEBUG] Loading YAML from: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data

    # --------------------------
    # Generic config getters
    # --------------------------
    def get_client_config(self, client_name: str) -> dict:
        return self.config.get("clients", {}).get(client_name, {})

    def resolve_secret(self, client_name: str) -> str:
        cfg = self.get_client_config(client_name)
        if env_key := cfg.get("api_key_env"):
            if val := os.getenv(env_key):
                return val
        if val := cfg.get("api_key"):
            return val
        return self.DEFAULT_API_KEY

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

    # --------------------------
    # Tool & Workflow getters
    # --------------------------
    def get_tool_sets(self) -> dict:
        return self.config.get("tools", {})

    # --------------------------
    # Loader for tool sets
    # --------------------------
    def load_tool_sets(self, sets_to_load: list[str] | None = None):
        """
        Load all tool sets from config, supporting:
        - external folders (`folder` in config)
        - default internal folders (app_root/tools/<set_name>/)
        """
        app_root = self.app_root
        sets_cfg = self.config.get("tools", {})

        print(f"[DEBUG] App root: {app_root}")
        print(f"[DEBUG] Sets to load: {sets_to_load or 'ALL'}")
        print(f"[DEBUG] Found sets in config: {list(sets_cfg.keys())}")

        for set_name, cfg in sets_cfg.items():
            if sets_to_load and set_name not in sets_to_load:
                print(f"[DEBUG] Skipping set '{set_name}' (not requested)")
                continue

            print(f"[DEBUG] Loading tool set '{set_name}'")

            folder = cfg.get("folder")
            # Determine folder path
            if folder:
                folder_path = Path(folder).expanduser().resolve()
                print(
                    f"[DEBUG] Using external folder for set '{set_name}': {folder_path}"
                )
                if not folder_path.exists() or not folder_path.is_dir():
                    print(f"[WARNING] Tool folder '{folder_path}' does not exist")
                    continue
            else:
                folder_path = app_root / "tools" / set_name
                if not folder_path.exists() or not folder_path.is_dir():
                    fallback_folder = app_root / "tools"
                    if fallback_folder.exists() and fallback_folder.is_dir():
                        folder_path = fallback_folder
                        print(
                            f"[DEBUG] Fallback folder for set '{set_name}': {folder_path}"
                        )
                    else:
                        print(f"[WARNING] No tools folder found for set '{set_name}'")
                        continue
                else:
                    print(
                        f"[DEBUG] Using internal folder for set '{set_name}': {folder_path}"
                    )

            sys.path.insert(0, str(folder_path))
            imported_any = False
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

            # Registry check
            try:
                from neuralcore.actions.manager import registry

                if imported_any:
                    if getattr(registry, "sets", {}).get(set_name):
                        print(f"[INFO] Tool set '{set_name}' registered successfully")
                    else:
                        print(
                            f"[WARNING] Tool set '{set_name}' imported but not found in registry"
                        )
                else:
                    print(f"[WARNING] No .py files imported for set '{set_name}'")
            except ImportError:
                if not imported_any:
                    print(f"[WARNING] No .py files imported for set '{set_name}'")

    def load_workflow_sets(self, engine):
        """
        Load workflow sets for the agent's workflow engine.

        Supports:
        - Simple string: workflow: deploy_chat
        - Old dict style (backward compatibility)
        - Global workflows defined at top level
        - Python modules and YAML workflows from folders
        """

        agent_cfg = getattr(engine.agent, "config", {})
        agent_workflow_cfg = agent_cfg.get("workflow")
        global_workflows = getattr(self, "config", {}).get("workflows", {})

        # ── 1. Resolve the workflow (support string OR dict) ─────────────────────
        resolved = {}
        primary_workflow_name = None

        if isinstance(agent_workflow_cfg, str):
            # New clean format: workflow: deploy_chat
            workflow_name = agent_workflow_cfg.strip()
            primary_workflow_name = workflow_name

            if workflow_name in global_workflows:
                resolved[workflow_name] = global_workflows[workflow_name]
                print(f"[DEBUG] Using global workflow '{workflow_name}' for agent")
            else:
                resolved[workflow_name] = {"name": workflow_name}
                print(
                    f"[WARNING] Workflow '{workflow_name}' not found in global workflows"
                )

        elif isinstance(agent_workflow_cfg, dict):
            # Old dict format (backward compatibility)
            for wf_key, wf_ref in agent_workflow_cfg.items():
                if isinstance(wf_ref, dict) and "name" in wf_ref:
                    wf_name = wf_ref["name"]
                    if wf_name in global_workflows:
                        resolved[wf_key] = global_workflows[wf_name]
                    else:
                        resolved[wf_key] = wf_ref
                else:
                    resolved[wf_key] = wf_ref
            if resolved:
                primary_workflow_name = next(iter(resolved.keys()))

        else:
            # Fallback
            primary_workflow_name = "default"
            resolved["default"] = global_workflows.get("default", {})

        # Store resolved workflows back into agent config
        engine.agent.config["workflow"] = resolved

        # ── 2. Apply the primary workflow to the engine ─────────────────────────
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
            # Ultimate fallback
            engine.current_workflow_name = "default"
            engine.workflow_description = "Default workflow"
            engine.workflow_steps = engine._resolve_steps(
                getattr(engine, "DEFAULT_WORKFLOW", [])
            )
            print(
                f"[DEBUG] No valid workflow found, falling back to default for agent '{engine.agent.name}'"
            )

        # ── 3. Optionally load workflow sets from folders (Python + YAML) ───────
        sets_cfg = getattr(engine, "workflow_sets_config", {})
        app_root = getattr(engine, "app_root", Path.cwd())

        for set_name, cfg in sets_cfg.items():
            folder = cfg.get("folder")
            folder_path = (
                Path(folder).expanduser().resolve()
                if folder
                else app_root / "workflows" / set_name
            )
            if not folder_path.exists() or not folder_path.is_dir():
                folder_path = app_root / "workflows"

            if not folder_path.exists() or not folder_path.is_dir():
                print(
                    f"[WARNING] Workflow folder not found for set '{set_name}': {folder_path}"
                )
                continue

            sys.path.insert(0, str(folder_path))
            imported_any = False

            # Load Python workflow modules
            for py_file in folder_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                try:
                    importlib.import_module(py_file.stem)
                    imported_any = True
                    print(
                        f"[INFO] Imported Python workflow '{py_file.name}' as set '{set_name}'"
                    )

                    try:
                        from neuralcore.workflows.registry import condition

                        condition.register_to_engine(engine)
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to register conditions from '{py_file.name}': {e}"
                        )

                except Exception as e:
                    print(f"[ERROR] Failed to import workflow {py_file.name}: {e}")

            # Load YAML workflow files
            for yaml_file in list(folder_path.glob("*.yml")) + list(
                folder_path.glob("*.yaml")
            ):
                try:
                    with open(yaml_file, "r") as f:
                        wf_data = yaml.safe_load(f)
                    engine.registered_workflows[set_name] = wf_data
                    imported_any = True
                    print(
                        f"[INFO] Loaded workflow YAML '{yaml_file.name}' as set '{set_name}'"
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to load workflow YAML {yaml_file.name}: {e}")

            sys.path.pop(0)

            if not imported_any:
                print(f"[WARNING] No workflows loaded for set '{set_name}'")

    # --------------------------
    # Loader for workflow sets (YAML-based)

    def get_workflow_sets(self, client_name: str | None = None) -> dict:
        """
        Return dict of workflow sets for a client.
        Priority:
          1. Client config workflows (client.workflow)
          2. Default workflows in app_root/workflows/
        """
        if client_name:
            client_cfg = self.get_client_config(client_name)
            return client_cfg.get("workflow_sets", {})
        return self.config.get("workflows", {})

    # Inside ConfigLoader
    def load_agent_from_config(self, agent_id: str):

        app_root = self.app_root

        # --- Agent config ---
        agent_cfg = self.get_agent_config(agent_id)
        if not agent_cfg:
            raise ValueError(f"No agent config found for '{agent_id}'")

        # --- Client config ---
        from neuralcore.core.client_factory import get_clients

        clients = get_clients()
        client_name = agent_cfg.get("client")
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        client = clients[client_name]

        from neuralcore.agents.core import Agent

        # --- Instantiate Agent ---
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

        # --- Load tools ---
        tools_to_load = agent_cfg.get("tool_sets", [])
        if tools_to_load:
            self.load_tool_sets(sets_to_load=tools_to_load)

        return agent


# --------------------------
# Singleton instance
# --------------------------
loader: ConfigLoader | None = None


def get_loader(
    cli_path: str | None = None, app_root: Path | None = None
) -> ConfigLoader:
    global loader
    if loader is None:
        loader = ConfigLoader(cli_path, app_root)
    return loader
