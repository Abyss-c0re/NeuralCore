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

        # NEW: Persistent queue where user and system (external code) can post messages.
        # Messages are treated as user prompts and redirected into the workflow.
        # Queue items can be:
        #   - str                     → treated as user prompt
        #   - dict with "content" key → treated as user prompt (role ignored for now)
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()

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

    # ---------------- PUBLIC API ----------------

    async def post_message(self, message: str | Dict[str, Any]) -> None:
        """
        Post a message as USER (default behavior).
        """
        if isinstance(message, str):
            item = {"role": "user", "content": message}
        elif isinstance(message, dict):
            item = {"role": "user", **message}  # ensure role is user unless overridden
            if "role" not in item:
                item["role"] = "user"
        else:
            item = {"role": "user", "content": str(message)}

        await self.message_queue.put(item)
        logger.debug(
            f"Agent '{self.name}' ← user message posted: {str(message)[:80]}..."
        )

    async def post_system_message(self, message: str | Dict[str, Any]) -> None:
        """
        Post a message as SYSTEM.
        This is useful for injecting instructions, context updates, tool results,
        or internal system events into the workflow.
        """
        if isinstance(message, str):
            item = {"role": "system", "content": message}
        elif isinstance(message, dict):
            item = {"role": "system", **message}
            if "role" not in item:
                item["role"] = "system"
        else:
            item = {"role": "system", "content": str(message)}

        await self.message_queue.put(item)
        logger.debug(
            f"Agent '{self.name}' ← system message posted: {str(message)[:80]}..."
        )

    async def post_control(self, control: str | Dict[str, Any]) -> None:
        """
        Post a control message/event to dynamically control the workflow
        while it is running.

        Supported controls (exactly as documented in Workflow.run):
            {"event": "go_to", "name": "think"}                    # or "index": 3, "offset": 2
            {"event": "switch_workflow", "name": "research"}
            {"event": "break"}
            {"event": "finish_iteration"}
            {"event": "restart_iteration"}
            {"event": "insert_steps", "steps": ["new_step1", ...], "after": "think"}
            {"event": "insert_steps", "steps": ["final"], "at_end": True}
            {"event": "cancelled"} / {"event": "finish"} / {"event": "needs_confirmation"}

        Controls are intercepted by Workflow._drain_control() during step execution.
        If posted outside an active run they are safely ignored (with debug log).
        """
        if isinstance(control, str):
            item = {"event": control}
        elif isinstance(control, dict):
            item = dict(control)  # shallow copy
            # Ensure it is recognisable as control even if caller omitted the key
            if not any(k in item for k in ("event", "action", "control")):
                item["event"] = "custom_control"
        else:
            item = {"event": "custom_control", "payload": str(control)}

        await self.message_queue.put(item)
        logger.debug(f"Agent '{self.name}' ← control posted: {str(control)[:80]}...")

    async def run(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Run the agent with queue support + full control-message compatibility.
        """
        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        temperature = (
            float(temperature) if temperature is not None else float(self.temperature)
        )
        max_tokens = int(max_tokens) if max_tokens is not None else int(self.max_tokens)

        # 1. Optional initial user prompt (backward compatibility)
        if user_prompt is not None and str(user_prompt).strip():
            async for event, payload in self.workflow.run(
                user_prompt, system_prompt, temperature, max_tokens, stop_event
            ):
                yield event, payload

        # 2. Continuous message queue processor
        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.debug(f"Agent '{self.name}' stopped via stop_event")
                    break

                # Get next message from queue
                msg = await self.message_queue.get()

                # === NEW: CONTROL MESSAGE SUPPORT (fixes compatibility) ===
                # Controls must be left for Workflow._drain_control() when a run is active.
                # If they arrive outside an active run we safely ignore them.
                if isinstance(msg, dict):
                    event = msg.get("event") or msg.get("action") or msg.get("control")
                    if event in {
                        "switch_workflow",
                        "go_to",
                        "break",
                        "finish_iteration",
                        "restart_iteration",
                        "insert_steps",
                        "needs_confirmation",
                        "cancelled",
                        "finish",
                    }:
                        logger.debug(
                            f"Agent '{self.name}' ← control ignored (outside active run): {event}"
                        )
                        self.message_queue.task_done()
                        continue

                # Normalize message (user / system)
                if isinstance(msg, str):
                    role = "user"
                    content = msg.strip()
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content") or str(msg)
                    content = str(content).strip()
                else:
                    role = "user"
                    content = str(msg).strip()

                if not content:
                    self.message_queue.task_done()
                    continue

                # === Redirect to workflow based on role ===
                if role == "system":
                    # For system messages, we pass them as the system_prompt override
                    current_system = content
                    user_content = ""  # no user prompt for pure system injection
                else:
                    # Normal user message
                    current_system = system_prompt
                    user_content = content

                # Run the workflow with appropriate prompt
                async for event, payload in self.workflow.run(
                    user_content
                    if role != "system"
                    else "",  # empty user prompt for system-only
                    current_system,
                    temperature,
                    max_tokens,
                    stop_event,
                ):
                    yield event, payload

                self.message_queue.task_done()

        except asyncio.CancelledError:
            logger.debug(f"Agent '{self.name}' run cancelled")
            raise
        except Exception as e:
            logger.error(f"Agent '{self.name}' queue error: {e}", exc_info=True)
            raise
