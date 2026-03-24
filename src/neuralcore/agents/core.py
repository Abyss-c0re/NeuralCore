import yaml
import asyncio
from pathlib import Path

from neuralcore.utils.logger import Logger
from neuralcore.actions.manager import (
    AgentActionHelper,
    ToolBrowser,
    DynamicActionManager,
    registry,
)

from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.cognition.memory import ContextManager
from neuralcore.core.client_factory import get_clients


from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = Logger.get_logger()


class Agent:
    def __init__(
        self, agent_id: str, loader, app_root: Path, config_file: Optional[Path] = None
    ):
        self.agent_id = agent_id
        self.loader = loader
        self.app_root = app_root
        self.config = self._load_agent_config(agent_id, config_file)
        clients = get_clients()
        self.client = clients[self.config.get("client", "main")]

        self.name = self.config.get("name", f"Agent-{agent_id}")
        self.description = self.config.get("description", "")
        self.max_iterations = self.config.get("max_iterations", 25)
        self.max_reflections = self.config.get("max_reflections", 2)
        self.temperature = self.config.get("temperature", 0.75)
        self.max_tokens = self.config.get("max_tokens", 12048)
        self.system_prompt: str = self.config.get(
            "system_prompt",
            self.client.system_prompt if hasattr(self.client, "system_prompt") else "",
        )

        self.registry = registry  # Global tool registry

        self.manager = DynamicActionManager(registry)
        self.context_manager = ContextManager()

        self.agent_tools = AgentActionHelper(
            self
        )  # Register tools that need access to the agent
        self.workflow = WorkflowEngine(self)
        ToolBrowser(registry, self.manager)
        logger.debug(" ToolBrowser registered")

    def attach_tools(self):
        """Call after instantiating Agent to load tool sets."""
        tool_sets = self.config.get("tool_sets", [])
        if tool_sets:
            self.loader.load_tool_sets(sets_to_load=tool_sets)
        # Explicitly register all actions in the registry
        for action_name in self.registry.all_actions:
            self.manager.load_tools([action_name])

    def _load_agent_config(self, agent_id: str, config_file: Optional[Path]) -> dict:
        print(f"[DEBUG] Loading config for agent_id='{agent_id}'")
        if config_file:
            print(f"[DEBUG] Using config file: {config_file}")
            if config_file.exists():
                with open(config_file, "r") as f:
                    full_cfg = yaml.safe_load(f)
                print(
                    f"[DEBUG] Config file loaded, keys at top level: {list(full_cfg.keys())}"
                )
                cfg = full_cfg.get("agents", {}).get(agent_id, {})
            else:
                print(f"[WARNING] Config file does not exist: {config_file}")
                cfg = {}
        else:
            print(f"[DEBUG] Using loader.config")
            loader_config_agents = getattr(self.loader, "config", {}).get("agents", {})
            print(
                f"[DEBUG] Agents in loader.config: {list(loader_config_agents.keys())}"
            )
            cfg = loader_config_agents.get(agent_id, {})

        if not cfg:
            print(f"[ERROR] Agent '{agent_id}' not found in config")
            raise ValueError(f"Agent '{agent_id}' not found")
        print(f"[DEBUG] Loaded config for agent '{agent_id}': {cfg.keys()}")
        return cfg

    def _load_agent_tools(self):
        tool_sets = self.config.get("tool_sets", [])
        self.loader.load_tool_sets(sets_to_load=tool_sets)
        for action_name in self.registry.all_actions:
            self.manager.load_tools([action_name])

    def _reset_state(self):
        self.task = ""
        self.goal = ""
        self.tool_results: List[Dict] = []
        self.executed_signatures: set[tuple] = set()
        self.steps: List[str] = []
        self._stop_event: Optional[asyncio.Event] = None

    # ---------------- PUBLIC API (signature unchanged) ----------------

    async def run(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Run the agent workflow with defaults safely pulled from config.
        Type-safe: ensures all values passed to workflow.run are correct types.
        """
        # Use config defaults if None
        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        temperature = (
            float(temperature) if temperature is not None else float(self.temperature)
        )
        max_tokens = int(max_tokens) if max_tokens is not None else int(self.max_tokens)

        async for event, payload in self.workflow.run(
            user_prompt, system_prompt, temperature, max_tokens, stop_event
        ):
            yield event, payload
