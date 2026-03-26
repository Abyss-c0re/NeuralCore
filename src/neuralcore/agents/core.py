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
        self.max_sub_agents = self.config.get("max_sub_agents", 6)
        self.sub_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> metadata
        self._sub_task_counter: int = 0
        self.task_context: Optional["ContextManager.TaskContext"] = None
        self.assigned_tools: Optional[List[str]] = None

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
        chat_mode: bool = False,
        workflow: Optional[str] = None,  # ← allow explicit workflow override
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Unified run method.
        - chat_mode=True          → persistent deploy_chat_loop (UI/terminal chat)
        - chat_mode=False         → headless / script mode (one-shot or queue-driven)
        """
        system_prompt = system_prompt or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Reset state for a fresh run
        self._reset_state()
        if stop_event is None:
            stop_event = asyncio.Event()

        # === CHAT MODE (persistent loop) ===
        if chat_mode:
            logger.info(
                f"Agent '{self.name}' → Starting CHAT mode with workflow 'deploy_chat'"
            )

            if user_prompt and str(user_prompt).strip():
                await self.post_message(user_prompt)

            # Run the persistent chat workflow
            async for event, payload in self.workflow.run(
                user_prompt="",  # not used directly
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_event=stop_event,
                workflow="deploy_chat",  # forces deploy_chat_loop
            ):
                yield event, payload
            return

        # === HEADLESS / SCRIPT MODE ===
        logger.info(f"Agent '{self.name}' → Starting HEADLESS mode")

        # If a specific workflow is requested (e.g. "default" for complex tasks)
        target_workflow = workflow or "default"

        try:
            # Initial prompt handling
            if user_prompt and str(user_prompt).strip():
                await self.post_message(user_prompt)

            while True:
                if stop_event.is_set():
                    logger.debug("Headless run stopped via stop_event")
                    break

                # Wait for next message (user or control)
                try:
                    msg = await asyncio.wait_for(self.message_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # No new input → continue or break depending on your policy
                    continue

                # Handle control messages
                if isinstance(msg, dict) and "event" in msg:
                    event = msg.get("event")
                    logger.debug(f"Headless received control: {event}")

                    if event == "switch_workflow":
                        wf_name = msg.get("name", "default")
                        logger.info(f"Headless switching to workflow: {wf_name}")
                        try:
                            self.workflow.switch_workflow(wf_name)
                        except Exception as e:
                            logger.warning(f"Switch failed: {e}")
                        self.message_queue.task_done()
                        continue

                    elif event in ("finish", "cancelled", "break"):
                        logger.info(f"Headless received termination signal: {event}")
                        self.message_queue.task_done()
                        break

                    self.message_queue.task_done()
                    continue

                # Normal content message
                if isinstance(msg, dict):
                    content = msg.get("content") or str(msg)
                else:
                    content = str(msg).strip()

                if not content:
                    self.message_queue.task_done()
                    continue

                logger.debug(f"Headless processing prompt: {content[:150]}...")

                # Run the workflow (default or the one set via switch)
                async for event, payload in self.workflow.run(
                    user_prompt=content,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_event=stop_event,
                    workflow=target_workflow,  # respect current workflow
                ):
                    yield event, payload

                self.message_queue.task_done()

        except asyncio.CancelledError:
            logger.debug("Headless run cancelled")
            raise
        except Exception as e:
            logger.error(f"Headless run error: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"Agent '{self.name}' headless run finished")

            # ====================== SUB-TASK / DEPLOYMENT EXECUTION ======================

    async def start_complex_deployment(
        self,
        task_description: str,
        user_facing_name: Optional[str] = None,
        sub_profile: Optional[str] = None,
        assigned_tools: Optional[List[str]] = None,  # ← NEW: best-match tools
        custom_system_prompt: Optional[str] = None,  # ← NEW: per-task prompt
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> str:
        """Start a specialized sub-agent with optional tool restriction."""
        self._sub_task_counter += 1
        task_id = f"deploy_{self.agent_id}_{self._sub_task_counter:03d}"
        display_name = user_facing_name or task_description[:65]

        logger.info(
            f"[DEPLOY] Starting sub-task {task_id}: {display_name} | tools={len(assigned_tools) if assigned_tools else 'all'}"
        )

        self.sub_tasks[task_id] = {
            "id": task_id,
            "display_name": display_name,
            "status": "pending",
            "started_at": asyncio.get_event_loop().time(),
            "description": task_description,
            "assigned_tools": assigned_tools or [],  # ← store for monitoring
            "progress": 0,
            "task_obj": None,
            "result": None,
            "error": None,
        }

        try:
            sub_agent = self._create_sub_agent(
                task_name=display_name,
                profile=sub_profile,
                assigned_tools=assigned_tools,  # ← pass down
                custom_system_prompt=custom_system_prompt,
                temperature=temperature,
                max_iterations=max_iterations,
            )

            sub_agent.task = task_description
            sub_agent.goal = task_description

            coro = self._run_sub_agent_internal(sub_agent, task_id, task_description)
            background_task = asyncio.create_task(coro, name=task_id)

            self.sub_tasks[task_id]["task_obj"] = background_task
            self.sub_tasks[task_id]["status"] = "running"

            return task_id

        except Exception as e:
            logger.error(
                f"[DEPLOY] Failed to create sub-agent {task_id}", exc_info=True
            )
            self.sub_tasks[task_id]["status"] = "failed"
            self.sub_tasks[task_id]["error"] = str(e)
            return f"ERROR_{task_id}"

    async def _run_sub_agent_internal(
        self, sub_agent: "Agent", task_id: str, task_description: str
    ):
        """Run sub-agent with dedicated workflow and task context."""
        task_ctx = getattr(sub_agent, "task_context", None)

        try:
            assigned = getattr(sub_agent, "assigned_tools", None)
            tool_hint = (
                f"\n\nYou have been given these tools: {', '.join(assigned[:15])}{', ...' if assigned and len(assigned) > 15 else ''}"
                if assigned
                else ""
            )

            sub_system = f"""You are a precise, focused sub-agent.
            Task: {task_description}{tool_hint}

            Complete the task using the tools available to you.
            Record important findings using task_context.add_important_result when useful.
            Stay concise and goal-oriented. Do not call tools unnecessarily."""

            async for event, payload in sub_agent.workflow.run(
                user_prompt=task_description,
                system_prompt=sub_system,
                workflow="sub_agent_execute",  # ← important
                temperature=0.25,
                max_tokens=10000,
            ):
                if event == "tool_result" and task_ctx and isinstance(payload, dict):
                    result_str = str(payload.get("result") or "")
                    if result_str.strip():
                        await task_ctx.add_important_result(
                            title=f"{payload.get('name', 'Tool')} Result",
                            content=result_str,
                            source=payload.get("name", "tool"),
                            metadata=payload.get("args", {}),
                        )

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

            if task_ctx:
                await task_ctx.add_important_result(
                    title="Final Summary",
                    content=summary,
                    source="sub_agent",
                    metadata={"task_id": task_id},
                )

            alert = f"""[DEPLOYMENT COMPLETE] ✅

            **Task ID:** `{task_id}`
            **Name:** {self.sub_tasks[task_id]["display_name"]}

            {summary}"""

            await self.post_system_message(alert)

        except Exception as exc:
            self.sub_tasks[task_id].update(
                {
                    "status": "failed",
                    "completed_at": asyncio.get_event_loop().time(),
                    "error": str(exc),
                }
            )
            await self.post_system_message(
                f"[DEPLOYMENT FAILED] Task `{task_id}` failed: {exc}"
            )
            logger.error(f"Sub-task {task_id} failed", exc_info=True)

    def _create_sub_agent(
        self,
        task_name: str,
        profile: Optional[str] = None,
        assigned_tools: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> "Agent":
        """Create a focused sub-agent with proper config and restricted tools."""
        logger.info(
            f"[SUB-AGENT] Creating '{task_name}' (profile={profile or 'auto'}, "
            f"tools={len(assigned_tools) if assigned_tools else 'full'})"
        )

        try:
            # ==================== CONFIG / PROFILE SELECTION ====================
            loader_config = getattr(self.loader, "config", {}) or {}
            agents_section = loader_config.get("agents", {})

            if profile and profile in agents_section:
                template = agents_section[profile]
                chosen_profile = profile
            elif f"sub_{self.agent_id}" in agents_section:
                template = agents_section[f"sub_{self.agent_id}"]
                chosen_profile = f"sub_{self.agent_id}"
            elif "sub_default" in agents_section:
                template = agents_section["sub_default"]
                chosen_profile = "sub_default"
            else:
                template = {}
                chosen_profile = None

            # ==================== INSTANTIATE SUB-AGENT ====================
            sub = Agent.__new__(Agent)

            sub.agent_id = f"{self.agent_id}_sub_{self._sub_task_counter}"
            sub.loader = self.loader
            sub.app_root = self.app_root
            sub.config = dict(template)  # copy

            # Sub-agent flags
            sub.sub_agent = True
            sub.dispatcher = self.agent_id
            sub.assigned_tools = assigned_tools or []

            # Basic attributes
            sub.name = (
                f"{template.get('name', 'Sub-Agent')}-{self._sub_task_counter:03d}"
                if template
                else f"Sub-{self._sub_task_counter:03d}"
            )
            sub.description = template.get("description", "Focused sub-agent")

            # Client
            clients = get_clients()
            client_key = template.get("client", "reasoning")
            sub.client = clients.get(client_key, clients.get("main"))

            # Runtime params
            sub.temperature = temperature or template.get("temperature", 0.25)
            sub.max_iterations = max_iterations or template.get("max_iterations", 15)
            sub.max_reflections = template.get("max_reflections", 4)
            sub.max_tokens = template.get("max_tokens", 20000)
            sub.max_sub_agents = template.get("max_sub_agents", 4)

            # System prompt
            base_prompt = custom_system_prompt or template.get("system_prompt", "")
            sub.system_prompt = base_prompt or (
                "You are a precise sub-agent. Complete the assigned task efficiently "
                "using only the tools you have been given."
            )

            # ==================== CORE INFRASTRUCTURE ====================
            sub.registry = registry
            sub.manager = DynamicActionManager(registry)
            sub.agent_tools = AgentActionHelper(sub)
            sub.workflow = WorkflowEngine(sub)

            # Context sharing
            sub.context_manager = self.context_manager
            sub.task_context = self.context_manager.create_task_context(task_name)

            # ==================== TOOL LOADING (KEY DIFFERENCE) ====================
            sub.attach_tools(assigned_tools=assigned_tools)

            ToolBrowser(registry, sub.manager)

            logger.info(
                f"[SUB-AGENT] '{sub.name}' created successfully | "
                f"profile='{chosen_profile or 'fallback'}' | "
                f"tools loaded: {len(assigned_tools) if assigned_tools else 'full set'}"
            )
            return sub

        except Exception as e:
            logger.error(
                f"[SUB-AGENT] Creation failed for '{task_name}'", exc_info=True
            )
            raise

    def attach_tools(self, assigned_tools: Optional[List[str]] = None):
        """Load tools for this agent.

        - Main agents → follow config.tool_sets + load everything (your original behavior)
        - Sub-agents  → load ONLY the assigned best-match tools (specialization)
        """
        if getattr(self, "sub_agent", False) and assigned_tools:
            # === SUB-AGENT WITH RESTRICTED TOOLSET ===
            if assigned_tools:
                logger.info(
                    f"[SUB-AGENT] '{self.name}' loading {len(assigned_tools)} specialized tools: "
                    f"{assigned_tools[:10]}{'...' if len(assigned_tools) > 10 else ''}"
                )
                self.manager.load_tools(
                    assigned_tools
                )  # ← Uses your DynamicActionManager
            else:
                # Fallback: load minimal safe set for sub-agents
                logger.warning(
                    f"[SUB-AGENT] '{self.name}' has no assigned tools → loading core only"
                )
                core_tools = [
                    "GetContext",
                    "RequestComplexAction",
                    "GetDeploymentStatus",
                    "browse_tools",
                ]
                self.manager.load_tools(
                    [t for t in core_tools if t in self.registry.all_actions]
                )
        else:
            # === MAIN AGENT or unrestricted sub-agent ===
            tool_sets = self.config.get("tool_sets", [])
            if tool_sets:
                logger.debug(f"Loading tool sets for main agent: {tool_sets}")
                self.loader.load_tool_sets(sets_to_load=tool_sets)

            # Load all registered actions (your original behavior)
            for action_name in list(self.registry.all_actions.keys()):
                try:
                    self.manager.load_tools([action_name])
                except Exception as e:
                    logger.warning(f"Failed to load tool '{action_name}': {e}")

        # === ALWAYS ensure critical control tools are present ===
        critical_tools = [
            "GetContext",
            "RequestComplexAction",
            "GetDeploymentStatus",
            "browse_tools",
        ]
        for tool_name in critical_tools:
            if tool_name in self.registry.all_actions and not self.manager.is_loaded(
                tool_name
            ):
                self.manager.load_tools([tool_name])

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
