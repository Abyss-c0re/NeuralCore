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
        self.agent_id: str = agent_id
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
        self.context_manager = ContextManager(self.max_tokens)

        self.agent_tools = AgentActionHelper(
            self
        )  # Register tools that need access to the agent
        self.workflow = WorkflowEngine(self)
        ToolBrowser(registry, self.manager)
        logger.debug(" ToolBrowser registered")
        self.sub_agent: bool = False
        self.dispatcher: str = "user"
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0
        self.sub_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> metadata
        self._sub_task_counter: int = 0

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
        """Post a message as USER."""
        if isinstance(message, str):
            item = {"role": "user", "content": message}
        elif isinstance(message, dict):
            item = {"role": "user", **message}
            if "role" not in item or item["role"] != "user":
                item["role"] = "user"
        else:
            item = {"role": "user", "content": str(message)}

        await self.message_queue.put(item)
        self._input_counter += 1
        self._input_event.set()  # still useful for other parts if any
        logger.debug(
            f"Agent '{self.name}' ← user message posted: {str(message)[:80]}..."
        )

    async def post_system_message(self, message: str | Dict[str, Any]) -> None:
        if isinstance(message, str):
            item = {"role": "system", "content": message}
        elif isinstance(message, dict):
            item = {"role": "system", **message}
        else:
            item = {"role": "system", "content": str(message)}

        await self.message_queue.put(item)
        logger.debug(
            f"Agent '{self.name}' ← system alert posted: {str(message)[:100]}..."
        )

        # Optional: signal that something important arrived (if you have other waiters)
        self._input_event.set()

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

        Controls are intercepted by Workflow._drain_control(self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0) during step execution.
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
        chat_mode: bool = False,  # ← NEW
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Run the agent.
        - chat_mode=True  → starts the deploy_chat_loop (real conversation)
        - chat_mode=False → original task-oriented behavior
        """
        system_prompt = system_prompt or self.system_prompt
        temperature = (
            float(temperature) if temperature is not None else float(self.temperature)
        )
        max_tokens = int(max_tokens) if max_tokens is not None else int(self.max_tokens)

        # Optional initial prompt (backward compatibility)
        if user_prompt and str(user_prompt).strip():
            async for event, payload in self.workflow.run(
                user_prompt, system_prompt, temperature, max_tokens, stop_event
            ):
                yield event, payload

        # === CHAT MODE ===
        if chat_mode:
            logger.info(f"Agent '{self.name}' → Starting CHAT mode (deploy_chat_loop)")

            # Put the first user message into the queue if provided (so first iteration doesn't wait)
            if user_prompt and str(user_prompt).strip():
                await self.message_queue.put({"role": "user", "content": user_prompt})
                self._input_event.set()

            # Start the chat workflow
            async for event, payload in self.workflow.run(
                user_prompt="",  # not used in chat mode
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_event=stop_event,
                workflow="deploy_chat",  # ← This selects your _wf_deploy_chat_loop
            ):
                yield event, payload
            return

        # === OLD NON-CHAT BEHAVIOR (kept for compatibility) ===
        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.debug(f"Agent '{self.name}' stopped via stop_event")
                    break

                msg = await self.message_queue.get()

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
                        logger.debug(f"Control ignored (outside active run): {event}")
                        self.message_queue.task_done()
                        continue

                # Normalize
                if isinstance(msg, str):
                    content = msg.strip()
                elif isinstance(msg, dict):
                    content = msg.get("content") or str(msg)
                    content = str(content).strip()
                else:
                    content = str(msg).strip()

                if not content:
                    self.message_queue.task_done()
                    continue

                async for event, payload in self.workflow.run(
                    content, system_prompt, temperature, max_tokens, stop_event
                ):
                    yield event, payload

                self.message_queue.task_done()

        except asyncio.CancelledError:
            logger.debug(f"Agent '{self.name}' run cancelled")
            raise
        except Exception as e:
            logger.error(f"Agent '{self.name}' queue error: {e}", exc_info=True)
            raise

            # ====================== SUB-TASK / DEPLOYMENT EXECUTION ======================

    async def start_complex_deployment(
        self, task_description: str, user_facing_name: Optional[str] = None
    ) -> str:
        """
        Starts a complex deployment in the background and returns the task_id immediately.
        """
        self._sub_task_counter += 1
        task_id = f"deploy_{self.agent_id}_{self._sub_task_counter:03d}"
        display_name = user_facing_name or task_description[:70]

        logger.info(f"[Agent {self.name}] Starting sub-task {task_id}: {display_name}")

        # Register immediately
        self.sub_tasks[task_id] = {
            "id": task_id,
            "display_name": display_name,
            "status": "pending",
            "started_at": asyncio.get_event_loop().time(),
            "description": task_description,
            "progress": 0,
            "task_obj": None,
            "result_summary": None,
            "error": None,
        }

        try:
            sub_agent = self._create_sub_agent()
            sub_agent.task = task_description
            sub_agent.goal = task_description

            # Create background task
            coro = self._run_sub_agent_internal(sub_agent, task_id, task_description)
            background_task = asyncio.create_task(coro, name=task_id)

            self.sub_tasks[task_id]["task_obj"] = background_task
            self.sub_tasks[task_id]["status"] = "running"

            return task_id  # ← Return task_id to the tool / chat

        except Exception as e:
            self.sub_tasks[task_id]["status"] = "failed"
            self.sub_tasks[task_id]["error"] = str(e)
            logger.error(f"Failed to start sub-task {task_id}", exc_info=True)
            return f"ERROR_{task_id}"

    async def _run_sub_agent_internal(
        self, sub_agent: "Agent", task_id: str, task_description: str
    ):
        """Internal coroutine that actually runs the sub-agent and updates status."""
        try:
            # Run the full default workflow
            events = []
            async for event, payload in sub_agent.workflow.run(
                user_prompt=task_description,
                system_prompt="You are a precise deployment executor. Complete the task thoroughly and report clear results.",
                workflow="default",
                temperature=0.3,
                max_tokens=10000,
            ):
                events.append((event, payload))
                # Optional: update progress here if you emit progress events

            # Success
            summary = await self._generate_deployment_summary(
                sub_agent, task_description
            )

            self.sub_tasks[task_id].update(
                {
                    "status": "completed",
                    "completed_at": asyncio.get_event_loop().time(),
                    "result": summary,
                    "progress": 100,
                }
            )

            # Alert the main chat
            alert = f"""[DEPLOYMENT COMPLETE] ✅

            **Task ID:** `{task_id}`
            **Name:** {self.sub_tasks[task_id]["display_name"]}

            {summary}

            Back to normal chat — how else can I help?"""

            await self.post_system_message(alert)

        except asyncio.CancelledError:
            self.sub_tasks[task_id]["status"] = "cancelled"
            logger.warning(f"Sub-task {task_id} was cancelled")
            raise
        except Exception as exc:
            self.sub_tasks[task_id].update(
                {
                    "status": "failed",
                    "completed_at": asyncio.get_event_loop().time(),
                    "error": str(exc),
                }
            )
            error_alert = f"[DEPLOYMENT FAILED] ❌ Task `{task_id}` failed: {exc}"
            await self.post_system_message(error_alert)
            logger.error(f"Sub-task {task_id} failed", exc_info=True)
        finally:
            # Optional: clean up old finished tasks after some time
            pass

    def _create_sub_agent(self) -> "Agent":
        """Create an isolated sub-agent for running complex tasks."""
        sub = Agent(
            agent_id=f"{self.agent_id}",
            loader=self.loader,
            app_root=self.app_root,
            # You can pass a minimal config or None
            config_file=None,
        )

        sub.sub_agent = True
        sub.dispatcher = self.agent_id
        # Copy important settings
        sub.max_iterations = self.max_iterations
        sub.max_reflections = self.max_reflections
        sub.temperature = 0.3
        sub.max_tokens = 10000

        sub.attach_tools()  # Load the same tools as main agent
        sub.context_manager = self.context_manager  # Fresh context

        return sub

    def get_sub_tasks(self) -> Dict[str, Dict]:
        """Return current status of all sub-agents (safe to call from tools or chat)."""
        now = asyncio.get_event_loop().time()
        return {
            tid: {
                **info,
                "runtime_seconds": round(now - info["started_at"], 1)
                if "started_at" in info
                else 0,
                "task_obj": None,  # don't expose the raw task object
            }
            for tid, info in self.sub_tasks.items()
        }

    async def get_sub_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific sub-task."""
        return self.sub_tasks.get(task_id)

    def cancel_sub_task(self, task_id: str) -> bool:
        """Cancel a running sub-task."""
        if task_id not in self.sub_tasks:
            return False
        task = self.sub_tasks[task_id].get("task_obj")
        if task and not task.done():
            task.cancel()
            self.sub_tasks[task_id]["status"] = "cancelling"
            return True
        return False

    def cleanup_finished_sub_tasks(self, older_than_seconds: int = 3600):
        """Remove old completed/failed tasks to keep memory clean."""
        now = asyncio.get_event_loop().time()
        to_remove = [
            tid
            for tid, info in self.sub_tasks.items()
            if info.get("status") in ("completed", "failed", "cancelled")
            and (now - info.get("completed_at", info["started_at"]))
            > older_than_seconds
        ]
        for tid in to_remove:
            self.sub_tasks.pop(tid, None)

    async def _generate_deployment_summary(self, sub_agent: "Agent", task: str) -> str:
        """Generate a natural, user-friendly summary from the sub-agent results."""
        tool_results_str = "\n".join(
            f"• {r.get('name', 'unknown')}: {str(r.get('result', ''))[:350]}"
            for r in getattr(sub_agent, "tool_results", [])[-12:]
        )

        prompt = f"""You are a helpful Deploy Agent. A complex background task has just finished.

        Task: {task}

        Key results from tools:
        {tool_results_str or "No detailed tool output available."}

        Write a friendly, concise summary (3-7 sentences) for the user.
        - Mention what was accomplished
        - Highlight any important outcomes or warnings
        - Use natural language and light emojis if appropriate
        - Keep it easy to read"""

        try:
            summary = await self.client.chat([{"role": "user", "content": prompt}])
            return summary.strip()
        except Exception:
            # Safe fallback
            return f"✅ The deployment task **{task}** has been completed successfully."
