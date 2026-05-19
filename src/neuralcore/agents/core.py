import time
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union


from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.workflows.registry import workflow
from neuralcore.cognition.memory import ContextManager
from neuralcore.utils.logger import Logger

from neuralcore.tasks.manager import TaskManager
from neuralcore.tasks.task import Task
from neuralcore.agents.state import AgentState
from neuralcore.actions.registry import registry
from neuralcore.actions.manager import (
    ActionRegistry,
    ToolBrowser,
    DynamicActionManager,
)


logger = Logger.get_logger()


class Agent:
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        client,
        app_root: Path,
        system_prompt: str = "",
        max_iterations: int = 25,
        temperature: float = 0.75,
        max_tokens: int = 12048,
        config: Optional[dict] = None,
        loader=None,
        sub_agent: bool = False,
        parent: Optional["Agent"] = None,
        action_registry: ActionRegistry = registry,
        app_config: Optional[dict] = None,
        embeddings_config: Optional[dict] = None,
    ):
        self.agent_id: str = agent_id
        self.name = name
        self.description = description
        self.client = client
        self.loader = loader
        self.app_root = app_root
        self.system_prompt: str = system_prompt or getattr(client, "system_prompt", "")
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.registry = action_registry

        # Config dict kept for runtime reference (workflow/tool_sets lookup)
        self.config = dict(config) if isinstance(config, dict) else {}

        # Resolved config sections for infrastructure components
        self.app_config = app_config or {}
        self.cognition_config = self.app_config.get("cognition", {})
        self.kb_config = self.app_config.get("knowledge_base", {})
        self.embeddings_config = embeddings_config or {}

        # ====================== STATE - THE SOURCE OF TRUTH ======================
        self.state: AgentState = AgentState(agent_id=agent_id)

        self.sub_agent: bool = sub_agent
        self.parent = parent

        # ====================== INFRASTRUCTURE ======================

        self.context_manager = ContextManager(self)
        self.action_manager = DynamicActionManager(self.registry, self)
        self.task_manager = TaskManager(self)

        self._last_sync_ts = 0.0
        self.workflow = WorkflowEngine(self, workflow)
        ToolBrowser(self.registry, self.action_manager)

        # Async runtime only
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0
        self._background_task: Optional[asyncio.Task[None]] = None

        # Light operational containers

        self.assigned_tools: Optional[List[str]] = None

        self._reset_state()

    # ====================== STATE DELEGATION HELPERS ======================
    @property
    def current_task(self) -> str:
        return self.state.current_task

    @current_task.setter
    def current_task(self, value: str) -> None:
        self.state.current_task = value

    @property
    def current_role(self) -> str:
        return self.state.current_role

    @current_role.setter
    def current_role(self, value: str) -> None:
        self.state.current_role = value

    @property
    def current_workflow(self) -> str:
        return self.state.current_workflow

    @current_workflow.setter
    def current_workflow(self, value: str) -> None:
        self.state.current_workflow = value

    @property
    def status(self) -> str:
        return self.state.status

    @property
    def stop_event(self) -> asyncio.Event:
        """Always returns the client's current stop event (never None)."""
        return getattr(self.client, "_current_stop_event", None) or asyncio.Event()

    @status.setter
    def status(self, value: str) -> None:
        self.state.status = value

    async def start_background_services(self):
        """Start all background services after the agent is fully initialized.
        Call this once when the agent starts running.
        """
        logger.info(f"🚀 Starting background services for '{self.name}'...")

        try:
            # 1. KnowledgeBase background watcher (file monitoring + reindexing)
            if hasattr(self.context_manager, "knowledge_base") and hasattr(
                self.context_manager.knowledge_base, "start_background_watcher"
            ):
                await self.context_manager.knowledge_base.start_background_watcher()
                logger.info("   ✓ KnowledgeBase background watcher started")

            # 2. Load reranker model (important for retrieval quality)
            if hasattr(self.context_manager, "consolidator"):
                await self.context_manager.consolidator.load_reranker()
                logger.info("   ✓ KnowledgeConsolidator reranker loaded")

            # 3. Start persistent message queue listener
            await self.start_background_queue_listener()
            logger.info("   ✓ Background queue listener started")

            # 4. Optional: Pre-load tools if not already loaded (can be heavy)
            if not self.action_manager.loaded_tools:
                self.attach_tools()
                logger.info("   ✓ Default tools attached")

            logger.info(f"✅ All background services started for '{self.name}'")

        except Exception as e:
            logger.error(f"Failed to start background services: {e}", exc_info=True)

    def get_full_state_dict(self) -> Dict[str, Any]:
        """Public, generic snapshot of the entire agent for any transport layer."""
        base = {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.to_dict(),
            "loaded_tools": self.action_manager.loaded_tools,
            "loaded_toolsets": self.action_manager.loaded_toolsets,
            "context_summary": self.context_manager.get_context_summary(
                max_messages=12, max_chars=2000
            ),
            "timestamp": time.time(),
        }

        # Extra top-level waiting info for quick access
        base["waiting"] = getattr(self.state, "waiting", False)
        base["wait_type"] = getattr(self.state, "wait_type", None)
        base["wait_prompt"] = getattr(self.state, "wait_prompt", "")

        return base

    def get_detailed_status(self) -> Dict[str, Any]:
        """Lightweight version for heartbeats / dashboards / NeuralHub."""

        # Safe wait elapsed calculation
        wait_elapsed: Optional[float] = None
        wait_start = getattr(self.state, "wait_start_time", None)
        if wait_start is not None:
            wait_elapsed = round(time.time() - wait_start, 1)

        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.state.status,
            "phase": self.state.phase,
            "current_task": self.state.current_task,
            "goal": self.state.task,
            "duration": round(self.state.duration, 1),
            "loop_count": self.state.loop_count,
            "total_tool_calls": self.state.total_tool_calls,
            "error_count": self.state.error_count,
            "waiting": getattr(self.state, "waiting", False),
            "wait_type": getattr(self.state, "wait_type", None),
            "wait_elapsed": wait_elapsed,  # ← now type-safe
            "timestamp": time.time(),
        }

    # ====================== BASIC UTILS ======================
    def is_sub_agent(self) -> bool:
        return self.sub_agent

    def get_parent_agent(self) -> Optional["Agent"]:
        return getattr(self, "parent", None)

    def _reset_state(self) -> None:
        """Reset everything through the state object (SSOT)."""
        self.state.reset_for_new_task()
        self.message_queue = asyncio.Queue()
        self._input_event.clear()
        self._input_counter = 0
        self._sync_loaded_tools_to_state()

    def _sync_loaded_tools_to_state(self) -> None:
        """Keep AgentState in sync with the real DynamicActionManager."""
        if hasattr(self, "manager") and hasattr(self.action_manager, "loaded_tools"):
            self.state.update_loaded_tools(self.action_manager.loaded_tools)

    # ====================== CONFIG & TOOLS ======================
    def reload_config(self, new_config: dict | str | Path) -> None:
        try:
            if isinstance(new_config, dict):
                parsed = new_config
            elif self.loader is not None:
                parsed = self.loader.parse_config(new_config)
            else:
                logger.error("reload_config: no loader and config is not a dict")
                return

            agent_cfg = parsed.get("agents", {}).get(self.agent_id, {}) or parsed

            if isinstance(agent_cfg, dict) and agent_cfg:
                self.config = dict(agent_cfg)
                self.name = self.config.get("name", self.name)
                self.temperature = self.config.get("temperature", self.temperature)
                self.max_tokens = self.config.get("max_tokens", self.max_tokens)
                self.system_prompt = self.config.get(
                    "system_prompt", self.system_prompt
                )

                self.state.current_role = self.config.get(
                    "role", self.state.current_role
                )
                logger.info(f"Agent '{self.name}' reloaded config")
        except Exception as e:
            logger.error(f"reload_config failed: {e}")

    def attach_tools(self, assigned_tools: Optional[List[str]] = None) -> None:
        if self.sub_agent and assigned_tools:
            valid_tools = [t for t in assigned_tools if t in registry.all_actions]
            if valid_tools:
                self.action_manager.unload_all()
                self.action_manager.load_tools(valid_tools)
                self._sync_loaded_tools_to_state()
                logger.info(
                    f"[SUB-AGENT] '{self.name}' loaded {len(valid_tools)} tools"
                )
            else:
                core = ["FindTool", "GetContext"]
                self.action_manager.load_tools(
                    [t for t in core if t in registry.all_actions]
                )
        else:
            tool_sets = self.config.get("tool_sets", [])
            if tool_sets and self.loader:
                self.loader.load_tool_sets(sets_to_load=tool_sets)
            for action_name in list(registry.all_actions.keys()):
                try:
                    self.action_manager.load_tools([action_name])
                except Exception as e:
                    logger.warning(f"Failed to load tool '{action_name}': {e}")

    def _resolve_workflow(self, workflow_override: Optional[str] = None) -> str:
        if workflow_override:
            return workflow_override

        workflow_cfg = self.config.get("workflow")
        if isinstance(workflow_cfg, str):
            return workflow_cfg.strip()
        if isinstance(workflow_cfg, dict) and workflow_cfg:
            return next(iter(workflow_cfg.keys()))

        # Check registered workflows in the engine
        if hasattr(self, "workflow") and hasattr(self.workflow, "registered_workflows"):
            registered = self.workflow.registered_workflows
            if registered:
                return next(iter(registered.keys()))

        logger.warning("No workflow found, using fallback")
        return "deploy_chat"

    # ====================== MESSAGING ======================
    async def add_message(self, role: str, message: str) -> None:
        await self.context_manager.add_message(role, message)
        await self._auto_sync_state()

    async def post_message(self, message: str | Dict[str, Any]) -> None:
        if isinstance(message, str):
            item = {"role": "user", "content": message}
        else:
            item = {"role": "user", **message}

        await self.message_queue.put(item)
        self._input_counter += 1
        self._input_event.set()

        role_str = item.get("role", "user")
        content_str = item.get("content", "")
        if not isinstance(content_str, str):
            content_str = str(content_str)

        await self.add_message(role_str, content_str)
        logger.debug(f"Agent '{self.name}' ← user message posted")

    async def post_system_message(self, message: str | Dict[str, Any]) -> None:
        """System messages go to queue + ContextManager + AgentState (as before)."""
        if isinstance(message, str):
            item = {"role": "system", "content": message}
        else:
            item = {"role": "system", **message}

        await self.message_queue.put(item)

        # Still route through add_message → ContextManager + state sync
        role_str = item.get("role", "system")
        content_str = item.get("content", "")
        if not isinstance(content_str, str):
            content_str = str(content_str)

        await self.add_message(role_str, content_str)  # This triggers auto-sync
        logger.debug(f"Agent '{self.name}' ← system message posted")

    async def post_control(self, control: str | Dict[str, Any]) -> None:
        """Control messages:
        - Go to queue (for workflow loop)
        - Are logged to ContextManager as system messages (clean "event: xxx" format)
        """
        if isinstance(control, str):
            item = {"event": control}
        else:
            item = dict(control)
            if not any(k in item for k in ("event", "action", "control")):
                item["event"] = "custom_control"

        # 1. Always put raw control into queue for internal processing
        await self.message_queue.put(item)

        # 2. Convert to clean system message for ContextManager only
        event_name = item.get("event", "custom_control")
        clean_content = f"event: {event_name}"

        # Add extra useful info if available (without bloating)
        if "task_id" in item:
            clean_content += f" | task_id={item['task_id']}"
        if "status" in item:
            clean_content += f" | status={item['status']}"
        if "sub_agent_name" in item:
            clean_content += f" | sub_agent={item['sub_agent_name']}"

        await self.add_message("system", clean_content)

        logger.debug(
            f"Agent '{self.name}' ← control posted as system | event={event_name}"
        )

    async def wait_for_incoming_message(
        self,
        timeout: float | None = None,
        role: Optional[str] = None,  # filter by role ("user", "system", etc.)
        contains: Optional[str] = None,  # filter by substring in content
        return_content_only: bool = False,
    ) -> Optional[Union[dict, str]]:
        """Wait for an incoming message with optional filtering and output format.

        timeout:
            - float > 0   → wait that many seconds
            - None        → wait forever (ideal for persistent chat_tool_loop)
            - <= 0        → treated as None (infinite)

        role / contains:
            Same selective waiting as before.

        return_content_only:
            - False (default) → returns the full original message dict (as posted)
            - True            → returns only the cleaned content string
                              (never the old fake placeholder)

        Returns None on timeout/cancellation.
        """
        if timeout is None or timeout <= 0:
            effective_timeout: float | None = None
        else:
            effective_timeout = timeout

        try:
            while True:  # inner loop for filtering
                if effective_timeout is None:
                    await self._input_event.wait()
                else:
                    await asyncio.wait_for(
                        self._input_event.wait(), timeout=effective_timeout
                    )

                self._input_event.clear()

                if not self.message_queue.empty():
                    msg = self.message_queue.get_nowait()

                    # === SELECTIVE FILTERING ===
                    if role is not None:
                        if isinstance(msg, dict) and msg.get("role") != role:
                            continue

                    if contains is not None:
                        content_str = (
                            msg.get("content", "")
                            if isinstance(msg, dict)
                            else str(msg)
                        )
                        if contains not in content_str:
                            continue

                    if return_content_only:
                        # Return clean content string only
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            if not isinstance(content, str):
                                content = str(content)
                            return content.strip()
                        else:
                            return str(msg).strip()
                    else:
                        # Return full original message (dict or raw)
                        return msg

                logger.debug("wait_for_incoming_message: queue empty after wake")

        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            logger.debug("wait_for_incoming_message: cancelled")
            return None
        except Exception as e:
            logger.debug(f"wait_for_incoming_message error: {e}")
            return None

    async def _auto_sync_state(self) -> None:
        """Lightweight auto-sync with rate limiting + learning trigger."""
        now = time.time()
        if now - self._last_sync_ts < 0.3:
            return
        self._last_sync_ts = now

        await self.context_manager.sync_to_agent_state(self.state)

        # Direct call — no wrapper needed
        asyncio.create_task(self.context_manager.consolidator.extract_and_consolidate())

    async def on_background_event(self, event: str, payload: Any) -> None:
        """Generic hook for background/headless events.
        Default: logs the event (keeps NeuralCore generic).
        External runners in NeuralVoid can override this for WebSocket, etc.
        """
        logger.info(
            f"[BACKGROUND EVENT] {self.name} | event='{event}' | "
            f"payload={str(payload)[:400]}{'...' if len(str(payload)) > 400 else ''}"
        )

    async def get_agent_status(self) -> Dict[str, Any]:
        try:
            context_summary = self.context_manager.get_context_summary(
                max_messages=8, max_chars=1200
            )

            # Safe wait elapsed calculation
            wait_elapsed: Optional[float] = None
            wait_start = getattr(self.state, "wait_start_time", None)
            if wait_start is not None:
                wait_elapsed = round(time.time() - wait_start, 1)

            # Rich status prompt for LLM
            status_prompt = f"""Agent: {self.name} (ID: {self.agent_id})
            Role: {self.state.current_role}
            Current Task: {self.state.current_task or "Idle"}
            Status: {self.state.status}
            Phase: {self.state.phase}
            Duration: {self.state.duration:.1f}s
          

            Waiting: {
                "YES (" + str(getattr(self.state, "wait_type", "unknown")) + ")"
                if getattr(self.state, "waiting", False)
                else "No"
            }
            Wait Prompt: {getattr(self.state, "wait_prompt", "")[:150] or "None"}

            Context: {context_summary}

            Provide a clear 4-7 sentence status report."""

            summary = await self.client.chat(
                [{"role": "user", "content": status_prompt}],
                temperature=0.3,
                max_tokens=700,
            )

            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "role": self.state.current_role,
                "task": self.state.current_task,
                "goal": self.state.task,
                "status": self.state.status,
                "phase": self.state.phase,
                "duration": round(self.state.duration, 1),
                # === Waiting State - fully exposed  ===
                "waiting": getattr(self.state, "waiting", False),
                "wait_type": getattr(self.state, "wait_type", None),
                "wait_prompt": getattr(self.state, "wait_prompt", ""),
                "wait_target": getattr(self.state, "wait_target", None),
                "wait_timeout": getattr(self.state, "wait_timeout", None),
                "wait_completed": getattr(self.state, "wait_completed", False),
                "wait_elapsed": wait_elapsed,  # ← fixed here
                "background_running": bool(
                    self._background_task and not self._background_task.done()
                ),
                "context_summary": context_summary[:800] + "..."
                if len(context_summary) > 800
                else context_summary,
                "llm_summary": summary.strip(),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.warning(f"get_agent_status failed: {e}")
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "status": self.state.status,
                "error": str(e),
                "timestamp": time.time(),
            }

    # ====================== AGENT COOPERATION ======================

    async def request_agent(
        self,
        target_agent: "Agent",
        task: Task,
        timeout: Optional[float] = None,
        drain_context: bool = True,
    ) -> Task:
        """Request another agent to execute a task.

        Creates a completion event on the task so the caller can await it.
        Posts a ``delegated_task`` control message to *target_agent*.
        When *drain_context* is True (default), the requesting agent will
        automatically drain the target agent's context/tool output after
        the task completes.

        Returns the same Task object; call ``await_task_completion(task)``
        to block until the target agent finishes.
        """
        # Ensure the task has a completion event for synchronization
        task.get_completion_event()
        task.requesting_agent_id = self.agent_id
        task.assigned_agent = target_agent.agent_id
        task.metadata["drain_context"] = drain_context
        task.metadata["requesting_agent"] = self.agent_id

        # Track in our state
        self.state.active_sub_agents.append(target_agent.agent_id)
        self.state.sub_task_ids.append(task.task_id)

        # Post the task to the target agent's control queue
        await target_agent.post_control(
            {
                "event": "delegated_task",
                "task_id": task.task_id,
                "description": task.description,
                "expected_outcome": task.expected_outcome,
                "requesting_agent_id": self.agent_id,
            }
        )

        logger.info(
            f"[COOPERATION] Agent '{self.name}' requested '{target_agent.name}' "
            f"to execute task {task.task_id[:8]}: {task.description[:80]}"
        )

        return task

    async def await_task_completion(
        self,
        task: Task,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Wait for a delegated task to complete and return its result.

        Uses the task's completion event for efficient async waiting.
        Returns a dict with task_id, status, result, error, and result_payload.
        Returns None-result on timeout.
        """
        event = task.get_completion_event()

        try:
            if timeout and timeout > 0:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
        except asyncio.TimeoutError:
            logger.warning(
                f"[COOPERATION] Timeout waiting for task {task.task_id[:8]} "
                f"after {timeout}s"
            )
            return {
                "task_id": task.task_id,
                "status": "timeout",
                "result": None,
                "error": f"Timed out after {timeout}s",
                "result_payload": None,
            }

        logger.info(
            f"[COOPERATION] Task {task.task_id[:8]} completed with status "
            f"{task.status.value}"
        )

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "result_payload": task.result_payload,
        }

    def drain_agent_context(self, source_agent: "Agent") -> Dict[str, Any]:
        """Drain context and tool output from another agent's context manager.

        This is critical for sub-agents which have their own context manager.
        Extracts tool results, short-term memory items, and tool call history
        so the requesting agent can incorporate the data into its own context.

        Returns a dict containing:
        - tool_results: list of tool result dicts
        - context_items: list of serialized knowledge items
        - tool_call_history: list of tool call records
        - context_summary: text summary of the agent's context
        """
        cm = source_agent.context_manager
        state = source_agent.state

        # Extract tool results from state
        tool_results = list(state.tool_results)

        # Extract knowledge items from short-term memory
        context_items = []
        if hasattr(cm, "short_term_mem") and cm.short_term_mem:
            for key, item in cm.short_term_mem.items():
                context_items.append(
                    {
                        "key": key,
                        "content": getattr(item, "content", str(item)),
                        "source_type": getattr(item, "source_type", "unknown"),
                        "category_path": getattr(item, "category_path", ""),
                        "timestamp": getattr(item, "timestamp", None),
                    }
                )

        # Extract tool call history
        tool_call_history = []
        if hasattr(cm, "tool_call_history"):
            tool_call_history = list(cm.tool_call_history)

        # Text summary for quick context injection
        context_summary = cm.get_context_summary(max_messages=20, max_chars=4000)

        drained = {
            "agent_id": source_agent.agent_id,
            "agent_name": source_agent.name,
            "tool_results": tool_results,
            "context_items": context_items,
            "tool_call_history": tool_call_history,
            "context_summary": context_summary,
        }

        logger.info(
            f"[COOPERATION] Drained context from '{source_agent.name}': "
            f"{len(tool_results)} tool results, {len(context_items)} context items, "
            f"{len(tool_call_history)} tool history entries"
        )

        return drained

    async def inject_drained_context(self, drained_data: Dict[str, Any]) -> None:
        """Inject drained context from another agent into this agent's state.

        Takes the output of ``drain_agent_context`` and merges it into the
        current agent's tool results and context.
        """
        source_name = drained_data.get("agent_name", "unknown")
        tool_results = drained_data.get("tool_results", [])
        context_summary = drained_data.get("context_summary", "")

        # Merge tool results into our state
        for tr in tool_results:
            tagged = dict(tr)
            tagged["source_agent"] = source_name
            self.state.tool_results.append(tagged)

        # Add summary as a system message so it enters our context
        if context_summary:
            await self.add_message(
                "system",
                f"[Delegated result from agent '{source_name}']\n{context_summary}",
            )

        logger.info(
            f"[COOPERATION] Injected {len(tool_results)} tool results "
            f"from '{source_name}' into '{self.name}'"
        )

    async def handle_delegated_task(
        self, control_msg: Dict[str, Any], task: Optional[Task] = None
    ) -> None:
        """Handle a delegated_task control message from another agent.

        Runs the task through this agent's task_manager.execute_delegated().
        """
        task_id = control_msg.get("task_id", "")
        description = control_msg.get("description", "")
        expected_outcome = control_msg.get("expected_outcome", "")
        requesting_agent_id = control_msg.get("requesting_agent_id", "")

        if task is None:
            task = Task(
                task_id=task_id,
                description=description,
                expected_outcome=expected_outcome,
            )
            task.requesting_agent_id = requesting_agent_id

        logger.info(
            f"[COOPERATION] Agent '{self.name}' received delegated task "
            f"{task.task_id[:8]}: {description[:80]}"
        )

        await self.task_manager.execute_delegated(task)

    # ====================== CONFIRMATION HANDLERS ======================

    async def _handle_confirmation_event(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Agent-level confirmation handler – sets state and yields event."""
        self.state.needs_approval = True
        self.state.pending_approval_prompt = payload.get(
            "preview", payload.get("details", {}).get("preview", "")
        )

        # Store full details for later re-execution
        self.state.last_confirmation_request = {
            "tool_name": payload.get("tool_name"),
            "args": payload.get("details", {}).get("args", payload.get("args", {})),
            "action_name": payload.get("tool_name"),
            "timestamp": time.time(),
        }

        logger.info(
            f"[CONFIRMATION] Agent '{self.name}' requires approval for "
            f"{payload.get('tool_name')} | preview={self.state.pending_approval_prompt[:120]}..."
        )

        yield (
            "needs_confirmation",
            {
                "tool_name": payload.get("tool_name"),
                "preview": self.state.pending_approval_prompt,
                "args": self.state.last_confirmation_request["args"],
                "agent_id": self.agent_id,
            },
        )

    async def _process_confirmation_response(self, control_msg: Dict[str, Any]) -> None:
        """Re-execute the tool after human approval (called from control message)."""
        if (
            not self.state.needs_approval
            or self.state.last_confirmation_request is None
        ):
            logger.warning(
                "[CONFIRMATION] Response received but no active confirmation request"
            )
            return

        # Safe access
        req = self.state.last_confirmation_request
        tool_name = req.get("tool_name")
        args = req.get("args", {})

        if not tool_name:
            logger.error("[CONFIRMATION] Missing tool_name in confirmation request")
            self.state.needs_approval = False
            self.state.last_confirmation_request = None
            return

        approved = control_msg.get("approved", False)

        if not approved:
            logger.info(f"[CONFIRMATION] User denied {tool_name}")
            self.state.add_tool_result(
                tool_name, "User denied confirmation", success=False
            )
            self.state.needs_approval = False
            self.state.pending_approval_prompt = ""
            self.state.last_confirmation_request = None
            await self.post_control({"event": "confirmation_denied", "tool": tool_name})
            return

        logger.info(f"[CONFIRMATION] User approved {tool_name} – re-executing")

        action = self.action_manager.get_executor(tool_name)
        if action is None:
            logger.error(f"[CONFIRMATION] Executor for {tool_name} not found")
            self.state.needs_approval = False
            self.state.last_confirmation_request = None
            return

        try:
            # Magic flag that Action.__call__ respects
            result = await action(_confirmation_passed=True, **args)

            self.state.add_tool_result(tool_name, result, success=True)
            await self.post_control(
                {
                    "event": "confirmation_approved",
                    "tool": tool_name,
                    "result": str(result)[:500],
                }
            )
        except Exception as exc:
            logger.error(f"[CONFIRMATION] Re-execution failed: {exc}")
            self.state.add_tool_result(
                tool_name, f"Error after approval: {exc}", success=False
            )
        finally:
            self.state.needs_approval = False
            self.state.pending_approval_prompt = ""
            self.state.last_confirmation_request = None

    async def _generic_queue_consumer(self) -> None:
        """Pure notifier — does NOT consume messages from the queue.
        Lets _run_headless_loop / wait_for_incoming_message be the real consumer.
        This removes the redundancy we are cleaning up for TaskManager."""
        logger.info(f"[QUEUE LISTENER] Pure notifier started for '{self.name}'")
        try:
            while True:
                # Just wait for any wake-up signal (no consumption)
                await self._input_event.wait()
                self._input_event.clear()

                if not self.message_queue.empty():
                    await self.on_background_event(
                        "queue_message_received",
                        {
                            "content": "message available",
                            "agent_id": self.agent_id,
                        },
                    )
                    # [FIX] Re-signal if the queue still has items after our
                    # notification pass.  Without this, other awaiters of the
                    # same _input_event (like wait_for_incoming_message inside
                    # chat_tool_loop) would miss the notification and block
                    # forever, because this consumer clears the event first.
                    if not self.message_queue.empty():
                        self._input_event.set()
                # No task_done() needed because we never called get()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("[QUEUE LISTENER] Notifier stopped")

    async def start_background_queue_listener(self) -> asyncio.Task[None]:
        """Generic persistent background listener for message_queue.

        - Drains the queue and forwards events.
        - Returns the task so client can await/cancel it if needed.
        """
        if self._background_task is not None and not self._background_task.done():
            logger.debug(f"[QUEUE LISTENER] Already active for Agent '{self.name}'")
            return self._background_task

        self.state.status = "listening"

        # Create task with proper typing
        self._background_task = asyncio.create_task(
            self._generic_queue_consumer(), name=f"queue_listener_{self.agent_id}"
        )
        return self._background_task

    async def stop_background_listener(self) -> None:
        """Gracefully stop the generic queue listener."""
        if self._background_task is None or self._background_task.done():
            return

        self._background_task.cancel()
        try:
            await self._background_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Error while stopping listener: {e}")

        self._background_task = None
        logger.info(f"[QUEUE LISTENER] Stopped for Agent '{self.name}'")

    async def run_background(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        workflow: Optional[str] = None,
        **kwargs,
    ) -> asyncio.Task:
        if self._background_task and not self._background_task.done():
            return self._background_task

        self.state.current_task = user_prompt or "background processing"
        self.state.status = "running_background"

        async def _background_consumer():
            await self.start_background_services()
            try:
                async for event, payload in self.run(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    workflow=workflow,
                    chat_mode=False,
                    **kwargs,
                ):
                    await self.post_control(
                        {
                            "event": "background_event",
                            "type": event,
                            "payload": payload
                            if isinstance(payload, dict)
                            else {"content": str(payload)},
                        }
                    )
            except asyncio.CancelledError:
                logger.info(f"Background task for '{self.name}' cancelled")
                raise
            except Exception as exc:
                logger.error(f"Background error in '{self.name}'", exc_info=True)
                await self.post_control(
                    {"event": "background_error", "error": str(exc)}
                )
            finally:
                self.state.status = "idle"
                self.state.current_task = ""
                self._background_task = None
                await self.stop_background_listener()

        self._background_task = asyncio.create_task(
            _background_consumer(), name=f"bg_{self.agent_id}"
        )
        logger.info(f"Agent '{self.name}' started in BACKGROUND mode")
        return self._background_task

    async def execute_loop(
        self, loop_name: str, initial_state: Optional[dict] = None, **kwargs
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Delegate loop execution to the WorkflowEngine and yield events"""
        await self.start_background_services()
        async for event, payload in self.workflow.execute_loop(
            loop_name, initial_state, **kwargs
        ):
            yield event, payload

    # ======================  RUN METHODS ======================

    def _setup_for_run(
        self,
        user_prompt: Optional[str],
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        workflow: Optional[str],
        chat_mode: bool = False,
    ) -> tuple[str, float, int, str]:
        """Common setup shared by chat and headless modes."""
        system_prompt = system_prompt or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self._reset_state()

        if user_prompt:
            # Sync to both places
            self.state.task = user_prompt

        self.current_task = user_prompt or (
            "chat session" if chat_mode else "headless processing"
        )
        self._status = "running"

        workflow_name = self._resolve_workflow(workflow_override=workflow)
        return system_prompt, temperature, max_tokens, workflow_name

    async def _run_workflow_once(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        workflow_name: str,
        stop_event: asyncio.Event,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Execute one full workflow run and yield its events."""
        async for event, payload in self.workflow.run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_event=stop_event,
            workflow=workflow_name,
        ):
            yield event, payload

    async def _handle_control_message(self, msg: Dict[str, Any]) -> bool:
        event = msg.get("event")
        if event in ("finish", "cancelled", "break"):
            logger.info(f"Termination signal: {event}")
            return True
        if event == "switch_workflow":
            wf_name = msg.get("name", self.state.current_workflow)
            try:
                self.workflow.switch_workflow(wf_name)
                self.state.current_workflow = wf_name
            except Exception as e:
                logger.warning(f"Workflow switch failed: {e}")
        return False

    async def _run_headless_loop(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        workflow_name: str,
        stop_event: asyncio.Event,
    ) -> AsyncIterator[Tuple[str, Any]]:
        processed_initial = False

        while not stop_event.is_set():
            try:
                timeout = 1.0 if not processed_initial else 30.0
                msg = await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue

            if isinstance(msg, dict) and "event" in msg:
                if await self._handle_control_message(msg):
                    break
                self.message_queue.task_done()
                continue

            content = msg.get("content") if isinstance(msg, dict) else str(msg).strip()
            if not content:
                self.message_queue.task_done()
                continue

            processed_initial = True
            logger.info(f"[HEADLESS] Processing: {content[:120]}...")

            async for event, payload in self.workflow.run(
                user_prompt=content,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_event=stop_event,
                workflow=workflow_name,
            ):
                yield event, payload

            self.message_queue.task_done()

    # ========================= MAIN PUBLIC RUN METHOD =========================

    async def run(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
        chat_mode: bool = False,
        workflow: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Main agent execution loop with full built-in confirmation support."""
        stop_event = stop_event or asyncio.Event()
        await self.start_background_services()

        system_prompt = system_prompt or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self._reset_state()

        if user_prompt:
            self.state.task = user_prompt
            self.state.current_task = user_prompt

        self.state.status = "running"
        workflow_name = self._resolve_workflow(workflow_override=workflow)

        if not chat_mode:
            self.action_manager.reset_to_default_package(
                "headless_bootstrap", self.workflow
            )
            self.attach_tools()

        try:
            if chat_mode:
                logger.info(
                    f"Agent '{self.name}' → CHAT mode | workflow={workflow_name}"
                )
                if user_prompt:
                    await self.post_message(user_prompt)

                async for event, payload in self.workflow.run(
                    user_prompt="",
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_event=stop_event,
                    workflow=workflow_name,
                ):
                    if event == "needs_confirmation":
                        async for (
                            conf_event,
                            conf_payload,
                        ) in self._handle_confirmation_event(payload):
                            yield conf_event, conf_payload
                        continue
                    yield event, payload

            else:
                logger.info(
                    f"Agent '{self.name}' → HEADLESS mode | workflow={workflow_name}"
                )
                if user_prompt:
                    await self.post_message(user_prompt)

                async for event, payload in self._run_headless_loop(
                    system_prompt, temperature, max_tokens, workflow_name, stop_event
                ):
                    if event == "needs_confirmation":
                        async for (
                            conf_event,
                            conf_payload,
                        ) in self._handle_confirmation_event(payload):
                            yield conf_event, conf_payload
                        continue
                    yield event, payload

            # ====================== CONTROL MESSAGE LOOP ======================
            # (handles confirmation responses + existing controls)
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(self.message_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                if isinstance(msg, dict):
                    #  Confirmation response handling
                    if msg.get("event") == "confirmation_response":
                        await self._process_confirmation_response(msg)
                        self.message_queue.task_done()
                        continue

                    # Original control handling
                    if await self._handle_control_message(msg):
                        break

                self.message_queue.task_done()

        finally:
            self.state.status = "idle"
            logger.info(f"Agent '{self.name}' run finished")
