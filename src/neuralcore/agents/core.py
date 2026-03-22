import yaml
import asyncio
from pathlib import Path

from neuralcore.utils.logger import Logger
from neuralcore.actions.manager import registry
from neuralcore.agents.engine import WorkflowEngine
from neuralcore.cognition.memory import ContextManager
from neuralcore.core.client_factory import get_clients
from neuralcore.utils.tool_loader import load_tool_sets
from neuralcore.workflows.default_flow import AgentFlow, workflow

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

        client_name = self.config.get("client", "main")
        clients = get_clients()
        self.clients = clients
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found")
        self.client = clients[client_name]

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
        self.registry = registry
        self.manager = registry.manager
        self.context_manager = ContextManager()
        self.workflow = WorkflowEngine(self)
        workflow.register_to(self, AgentFlow(self))

        self._load_agent_tools()

        self._reset_state()

    def _load_agent_config(self, agent_id: str, config_file: Optional[Path]) -> dict:
        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                full_cfg = yaml.safe_load(f)
            cfg = full_cfg.get("agents", {}).get(agent_id, {})
        else:
            cfg = getattr(self.loader, "config", {}).get("agents", {}).get(agent_id, {})
        if not cfg:
            raise ValueError(f"Agent '{agent_id}' not found")
        return cfg

    def _load_agent_tools(self):
        tool_sets = self.config.get("tool_sets", [])
        load_tool_sets(self.loader, app_root=self.app_root, sets_to_load=tool_sets)
        for action_name in self.registry.all_actions:
            self.registry.manager.load_tools([action_name])

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
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1212,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        async for event, payload in self.workflow.run(
            user_prompt, system_prompt, temperature, max_tokens, stop_event
        ):
            yield event, payload
