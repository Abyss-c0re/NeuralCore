import time
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from neuralcore.utils.prompt_builder import PromptBuilder


from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.workflows.registry import workflow
from neuralcore.cognition.memory import ContextManager
from neuralcore.clients.factory import get_clients
from neuralcore.utils.logger import Logger
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
        loader,
        app_root: Path,
        config_file: Optional[Path] = None,
        config_override: Optional[dict] = None,
        config: Optional[dict] = None,
        sub_agent: bool = False,
        action_registry: ActionRegistry = registry,
    ):
        self.agent_id: str = agent_id
        self.loader = loader
        self.app_root = app_root
        self.registry = action_registry

        # ====================== STATE - THE SOURCE OF TRUTH ======================
        self.state: AgentState = AgentState(agent_id=agent_id)

        # ====================== CONFIG HANDLING ======================
        self.sub_agent: bool = sub_agent

        if config is not None and isinstance(config, dict):
            self.config = dict(config)
        # 2. config_override (still supported for SubAgent and direct calls)
        elif config_override is not None:
            self.config = dict(config_override)
        # 3. Legacy fallback (old direct Agent(...) calls)
        else:
            if config_file:
                full_config = self.loader.parse_config(config_file)
                agent_cfg = full_config.get("agents", {}).get(agent_id, {})
                self.config = dict(agent_cfg) if isinstance(agent_cfg, dict) else {}
            else:
                self.config = self.loader.get_agent_config(agent_id)

        if not isinstance(self.config, dict) or not self.config:
            raise ValueError(f"Agent config for '{agent_id}' is empty or not found")

        clients = get_clients()
        client_name = self.config.get("client", "main")
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        self.client = clients[client_name]

        self.name = self.config.get("name", f"Agent-{agent_id}")
        self.description = self.config.get("description", "")
        self.max_iterations = self.config.get("max_iterations", 25)
        self.max_reflections = self.config.get("max_reflections", 2)
        self.temperature = self.config.get("temperature", 0.75)
        self.max_tokens = self.config.get("max_tokens", 12048)
        self.system_prompt: str = self.config.get(
            "system_prompt",
            getattr(self.client, "system_prompt", ""),
        )

        # ====================== INFRASTRUCTURE ======================

        self.context_manager = ContextManager(self)
        self.manager = DynamicActionManager(self.registry, self)

        self._last_sync_ts = 0.0
        self.workflow = WorkflowEngine(self, workflow)
        ToolBrowser(self.registry, self.manager)

        # Async runtime only
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0
        self._background_task: Optional[asyncio.Task[None]] = None

        # Light operational containers

        self.sub_tasks: Dict[str, Dict[str, Any]] = {}
        self._sub_task_events: Dict[str, asyncio.Event] = {}
        self._sub_task_counter: int = 0
        self.task_context: Optional["ContextManager.TaskContext"] = None
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
            if not self.manager.loaded_tools:
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
            "loaded_tools": self.manager.loaded_tools,
            "loaded_toolsets": self.manager.loaded_toolsets,
            "sub_tasks": self.get_sub_tasks(),
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
            "sub_tasks_count": len(self.sub_tasks),
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
        """Reset everything through the state object."""
        self.state.reset_for_new_task()
        self.message_queue = asyncio.Queue()
        self._input_event.clear()
        self._input_counter = 0
        self.sub_tasks.clear()
        self._sub_task_counter = 0
        self._sync_loaded_tools_to_state()

    def _sync_loaded_tools_to_state(self) -> None:
        """Keep AgentState in sync with the real DynamicActionManager."""
        if hasattr(self, "manager") and hasattr(self.manager, "loaded_tools"):
            self.state.update_loaded_tools(self.manager.loaded_tools)

    # ====================== CONFIG & TOOLS ======================
    def reload_config(self, new_config: dict | str | Path) -> None:
        try:
            parsed = self.loader.parse_config(new_config)
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
                self.manager.unload_all()
                self.manager.load_tools(valid_tools)
                self._sync_loaded_tools_to_state()
                logger.info(
                    f"[SUB-AGENT] '{self.name}' loaded {len(valid_tools)} tools"
                )
            else:
                core = ["FindTool", "GetContext"]
                self.manager.load_tools([t for t in core if t in registry.all_actions])
        else:
            tool_sets = self.config.get("tool_sets", [])
            if tool_sets:
                self.loader.load_tool_sets(sets_to_load=tool_sets)
            for action_name in list(registry.all_actions.keys()):
                try:
                    self.manager.load_tools([action_name])
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

        if hasattr(self.loader, "config"):
            global_workflows = self.loader.config.get("workflows", {})
            if global_workflows:
                return next(iter(global_workflows.keys()))

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

    async def wait_for_sub_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        raise_on_timeout: bool = False,
    ) -> Dict[str, Any]:
        """
        Wait until a sub-task reaches a terminal state.
        Uses asyncio.Event when available (clean & efficient).
        Falls back to polling for legacy tasks.
        """
        if task_id not in self.sub_tasks:
            return {"error": f"Sub-task {task_id} not found", "status": "not_found"}

        event = self._sub_task_events.get(task_id)

        if event is None:
            # Legacy fallback (polling)
            return await self._poll_sub_task_status(task_id, timeout)

        try:
            if timeout is not None:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()

            return self.sub_tasks.get(task_id, {})
        except asyncio.TimeoutError:
            if raise_on_timeout:
                raise TimeoutError(
                    f"Sub-task {task_id} did not complete within {timeout}s"
                )
            return {"status": "timeout", "task_id": task_id}

    async def _auto_sync_state(self) -> None:
        """Lightweight auto-sync with rate limiting + learning trigger."""
        now = time.time()
        if now - self._last_sync_ts < 0.3:
            return
        self._last_sync_ts = now

        await self.context_manager.sync_to_agent_state(self.state)

        # Direct call — no wrapper needed
        asyncio.create_task(self.context_manager.consolidator.extract_and_consolidate())

    async def _poll_sub_task_status(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        start = time.time()
        while True:
            info = self.sub_tasks.get(task_id, {})
            if info.get("status") in ("completed", "failed", "cancelled"):
                return info
            if timeout and (time.time() - start) > timeout:
                return {"status": "timeout", "task_id": task_id}
            await asyncio.sleep(0.1)

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
            Sub-tasks: {len(self.sub_tasks)}

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
                "sub_tasks_count": len(self.sub_tasks),
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

        action = self.manager.get_executor(tool_name)
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
        logger.info(
            f"[QUEUE LISTENER] Generic background queue listener STARTED for '{self.name}'"
        )
        try:
            while True:
                try:
                    # Wait for any item to appear
                    await self.message_queue.get()  # blocks until something is there

                    self._input_event.set()  # wake the main loop
                    await self.on_background_event(
                        "queue_message_received",
                        {
                            "content": "message available",
                            "agent_id": self.agent_id,
                        },
                    )
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass
        finally:
            logger.info("[QUEUE LISTENER] ... STOPPED")

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

    async def start_background_loop(
        self,
        loop_name: str,
        task_description: Optional[str] = None,
        initial_state: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Launch a named loop from WorkflowEngine as a true background task.
        Returns the task_id for monitoring/cancellation."""
        self._sub_task_counter += 1
        task_id = f"bg_loop_{self.agent_id}_{self._sub_task_counter:03d}"

        display_name = task_description or f"Background loop: {loop_name}"

        self.sub_tasks[task_id] = {
            "id": task_id,
            "display_name": display_name,
            "type": "background_loop",
            "loop_name": loop_name,
            "status": "running",
            "started_at": asyncio.get_event_loop().time(),
            "task_obj": None,
            "result": None,
            "error": None,
        }

        async def _background_loop_runner():
            try:
                async for event, payload in self.execute_loop(
                    loop_name=loop_name,
                    initial_state=initial_state,
                    **kwargs,
                ):
                    # Forward important events to parent
                    await self.post_control(
                        {
                            "event": "background_loop_event",
                            "task_id": task_id,
                            "loop_name": loop_name,
                            "type": event,
                            "payload": payload
                            if isinstance(payload, dict)
                            else {"data": str(payload)},
                        }
                    )

                    # Auto-update sub_task status on completion signals
                    if event in ("loop_completed", "loop_broken"):
                        self.sub_tasks[task_id]["status"] = (
                            "completed" if event == "loop_completed" else "broken"
                        )

            except asyncio.CancelledError:
                self.sub_tasks[task_id]["status"] = "cancelled"
                await self.post_control(
                    {"event": "background_loop_cancelled", "task_id": task_id}
                )
                raise
            except Exception as exc:
                logger.error(f"Background loop {loop_name} failed", exc_info=True)
                self.sub_tasks[task_id].update({"status": "failed", "error": str(exc)})
                await self.post_control(
                    {
                        "event": "background_loop_failed",
                        "task_id": task_id,
                        "error": str(exc),
                    }
                )
            finally:
                if self.sub_tasks[task_id]["status"] == "running":
                    self.sub_tasks[task_id]["status"] = "completed"

        task = asyncio.create_task(_background_loop_runner(), name=f"bg_loop_{task_id}")
        self.sub_tasks[task_id]["task_obj"] = task

        logger.info(
            f"Agent '{self.name}' started background loop '{loop_name}' → task_id={task_id}"
        )
        return task_id

    # ====================== REFACTORED RUN METHODS ======================

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
            self.manager.reset_to_default_package("headless_bootstrap", self.workflow)
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
                    # NEW: Confirmation response handling
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

    # ====================== IMPROVED COMPLEX DEPLOYMENT ======================

    async def start_complex_deployment(
        self,
        task_description: str,
        user_facing_name: Optional[str] = None,
        sub_profile: Optional[str] = None,
        assigned_tools: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        depends_on: Optional[str] = None,
    ) -> str:
        self._sub_task_counter += 1
        task_id = f"deploy_{self.agent_id}_{self._sub_task_counter:03d}"
        self._sub_task_events[task_id] = asyncio.Event()
        display_name = user_facing_name or task_description[:65]

        logger.info(
            f"[DEPLOY] Starting sub-task {task_id}: {display_name} | "
            f"depends_on={depends_on or 'None'} | tools={len(assigned_tools) if assigned_tools else 'all'}"
        )

        self.sub_tasks[task_id] = {
            "id": task_id,
            "display_name": display_name,
            "status": "pending",
            "started_at": asyncio.get_event_loop().time(),
            "description": task_description,
            "assigned_tools": assigned_tools or ["DynamicCore"],
            "progress": 0,
            "task_obj": None,
            "result": None,
            "error": None,
            "depends_on": depends_on,
            "dependents": [],
            "completed_at": None,
            "custom_system_prompt": custom_system_prompt,
        }

        if depends_on and depends_on in self.sub_tasks:
            self.sub_tasks[depends_on]["dependents"].append(task_id)

        try:
            sub_agent = SubAgent(
                parent=self,
                task_name=display_name,
                assigned_tools=assigned_tools,
                custom_system_prompt=custom_system_prompt,
                temperature=temperature or 0.25,
                max_iterations=max_iterations,
                profile=sub_profile,
                agent_id_override=f"{self.agent_id}_sub_{self._sub_task_counter}",
            )

            sub_agent.state.task = task_description
            sub_agent.current_task = task_description
            sub_agent.current_role = f"sub-agent:{display_name}"

            coro = self._run_sub_agent_internal(
                sub_agent, task_id, task_description, depends_on
            )
            background_task = asyncio.create_task(coro, name=task_id)

            self.sub_tasks[task_id]["task_obj"] = background_task
            self.sub_tasks[task_id]["status"] = "waiting" if depends_on else "running"

            return task_id

        except Exception as e:
            logger.error(
                f"[DEPLOY] Failed to create sub-agent {task_id}", exc_info=True
            )
            self.sub_tasks[task_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": asyncio.get_event_loop().time(),
                }
            )
            return f"ERROR_{task_id}"

    # ====================== IMPROVED SUB-AGENT RUNNER ======================

    async def _run_sub_agent_internal(
        self,
        sub_agent: "SubAgent",
        task_id: str,
        task_description: str,
        depends_on: Optional[str] = None,
    ):
        try:
            # === 1. WAIT FOR DEPENDENCY ===
            if depends_on:
                logger.info(f"Sub-task {task_id} waiting for dependency {depends_on}")
                self.sub_tasks[task_id]["status"] = "waiting"
                while True:
                    dep = self.sub_tasks.get(depends_on)
                    if dep and dep.get("status") in (
                        "completed",
                        "failed",
                        "cancelled",
                    ):
                        break
                    await asyncio.sleep(0.25)

                if dep and dep.get("result"):
                    await sub_agent.context_manager.add_external_content(
                        source_type="dependency_result",
                        content=f"Result from previous step ({depends_on}):\n{dep['result']}",
                        metadata={"dependency_task_id": depends_on},
                    )

            # === 2. SAFEGUARDED TOOL LOADING ===
            assigned = getattr(sub_agent, "assigned_tools", None) or []
            if assigned:
                # Filter only tools that actually exist
                valid_tools = [t for t in assigned if t in registry.all_actions]
                if valid_tools:
                    sub_agent.manager.unload_all()
                    sub_agent.manager.load_tools(valid_tools)
                    logger.info(
                        f"[SUB-AGENT] Loaded {len(valid_tools)} valid tools: {valid_tools}"
                    )
                else:
                    logger.warning(
                        f"[SUB-AGENT] No valid tools found in assigned list: {assigned}"
                    )

            # Always ensure core tools
            for core in ["GetContext", "GetDeploymentStatus"]:
                if core in registry.all_actions and not sub_agent.manager.is_loaded(
                    core
                ):
                    sub_agent.manager.load_tools([core])

            # === 3. SYSTEM PROMPT ===
            sub_system = getattr(sub_agent, "system_prompt", "")
            if not sub_system or "precise sub-agent" not in sub_system.lower():
                sub_system = PromptBuilder.sub_agent_system_prompt(
                    task_desc=task_description, assigned_tools=assigned
                )

            # === 4. EXECUTE ===
            async for event, payload in sub_agent.run(
                user_prompt=task_description,
                system_prompt=sub_system,
                temperature=getattr(sub_agent, "temperature", 0.25),
                max_tokens=10000,
                workflow="sub_agent_execute",
                chat_mode=False,
            ):
                if event in (
                    "tool_result",
                    "step_completed",
                    "iteration_finished",
                    "final_answer",
                ):
                    await self.post_control(
                        {
                            "event": "sub_agent_progress",
                            "task_id": task_id,
                            "type": event,
                            "payload": payload,
                            "sub_agent_name": sub_agent.name,
                            "origin": sub_agent.agent_id,
                        }
                    )

        except asyncio.CancelledError:
            self.sub_tasks[task_id].update(
                {"status": "cancelled", "completed_at": asyncio.get_event_loop().time()}
            )
            await self.post_control(
                {"event": "sub_task_failed", "task_id": task_id, "error": "cancelled"}
            )
            raise
        except Exception as exc:
            logger.error(f"Sub-agent {task_id} failed", exc_info=True)
            self.sub_tasks[task_id].update(
                {
                    "status": "failed",
                    "completed_at": asyncio.get_event_loop().time(),
                    "error": str(exc),
                }
            )
            await self.post_control(
                {"event": "sub_task_failed", "task_id": task_id, "error": str(exc)}
            )
        finally:
            summary = await self._generate_deployment_summary(
                sub_agent, task_description
            )
            self.sub_tasks[task_id].update(
                {
                    "completed_at": asyncio.get_event_loop().time(),
                    "result": summary,
                    "progress": 100,
                    "status": "failed"
                    if self.sub_tasks[task_id].get("status") == "failed"
                    else "completed",
                }
            )

            await self.post_control(
                {
                    "event": "sub_task_completed",
                    "task_id": task_id,
                    "summary": summary[:500],
                    "success": self.sub_tasks[task_id]["status"] == "completed",
                    "origin": sub_agent.agent_id,
                }
            )

            for dep_id in self.sub_tasks[task_id].get("dependents", []):
                await self.post_control(
                    {
                        "event": "dependency_satisfied",
                        "task_id": dep_id,
                        "depends_on": task_id,
                    }
                )

            await self.post_system_message(
                f"✅ Step completed: {self.sub_tasks[task_id].get('display_name', task_id)}\n{summary[:300]}{'...' if len(summary) > 300 else ''}"
            )

            sub_agent.context_manager.prune_sub_agent_noise()

            if task_id in self._sub_task_events:
                self._sub_task_events[task_id].set()

    def get_sub_tasks(self) -> Dict[str, Dict]:
        now = asyncio.get_event_loop().time()
        return {
            tid: {
                **info,
                "runtime_seconds": round(now - info["started_at"], 1)
                if "started_at" in info
                else 0,
                "task_obj": None,
            }
            for tid, info in self.sub_tasks.items()
        }

    async def get_sub_task_status(self, task_id: str) -> Optional[Dict]:
        return self.sub_tasks.get(task_id)

    def cancel_sub_task(self, task_id: str) -> bool:
        if task_id not in self.sub_tasks:
            return False
        task = self.sub_tasks[task_id].get("task_obj")
        if task and not task.done():
            task.cancel()
            self.sub_tasks[task_id]["status"] = "cancelling"
            return True
        return False

    def cleanup_finished_sub_tasks(self, older_than_seconds: int = 3600):
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

    def purge_sub_agents(self, only_completed: bool = True, force: bool = False) -> int:
        if not self.sub_tasks:
            return 0

        now = asyncio.get_event_loop().time()
        to_purge = []
        purged_count = 0

        for task_id, info in list(self.sub_tasks.items()):
            status = info.get("status", "unknown")
            age = now - info.get("started_at", 0)

            should_purge = False
            if only_completed:
                if status in ("completed", "failed", "cancelled"):
                    should_purge = True
            else:
                should_purge = True

            if status == "running" and age < 30 and not force:
                continue

            if should_purge:
                to_purge.append(task_id)

        for task_id in to_purge:
            info = self.sub_tasks[task_id]
            task_obj = info.get("task_obj")

            if task_obj and not task_obj.done() and force:
                try:
                    task_obj.cancel()
                    logger.info(f"Purged (cancelled) running sub-task: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel task {task_id}: {e}")

            self.sub_tasks.pop(task_id, None)

            if hasattr(self.context_manager, "_task_contexts"):
                self.context_manager._task_contexts.pop(task_id, None)

            purged_count += 1
            logger.debug(f"Purged sub-agent: {task_id} (status: {info.get('status')})")

        if purged_count > 0:
            logger.info(
                f"Purged {purged_count} sub-agent(s). Remaining: {len(self.sub_tasks)}"
            )

        return purged_count

    async def _generate_deployment_summary(self, sub_agent: "Agent", task: str) -> str:
        tool_results_str = "\n".join(
            f"• {r.get('name', 'unknown')}: {str(r.get('result', ''))[:350]}"
            for r in getattr(sub_agent, "tool_results", [])[-12:]
        )

        prompt = PromptBuilder.task_execution_summary_prompt(task, tool_results_str)

        try:
            summary = await self.client.chat([{"role": "user", "content": prompt}])
            return summary.strip()
        except Exception:
            return f"✅ The deployment task **{task}** has been completed successfully."

    async def wait_for_sub_agent_event(
        self,
        task_id: str,
        event: str = "sub_task_completed",
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a specific event from a particular sub-agent.
        Always returns a dict (or None). Strings are parsed if possible.
        """
        if task_id not in self.sub_tasks:
            logger.warning(f"[wait_for_sub_agent_event] task_id {task_id} not found")
            return None

        result = await self.wait_for_incoming_message(
            timeout=timeout,
            contains=f'"task_id": "{task_id}"',
            return_content_only=False,
        )

        if result is None:
            return None

        if isinstance(result, dict):
            return result

        # Try to parse string as JSON
        if isinstance(result, str):
            try:
                import json

                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # Fallback: wrap string in a dict
            return {"raw_message": result}

        return None


class SubAgent(Agent):
    def __init__(
        self,
        parent: "Agent",
        task_name: str,
        assigned_tools: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        profile: Optional[str] = None,
        agent_id_override: Optional[str] = None,
    ):
        loader_config = getattr(parent.loader, "config", {}) or {}
        agents_section = loader_config.get("agents", {})

        if profile and profile in agents_section:
            template = dict(agents_section[profile])
            chosen_profile = profile
        elif f"sub_{parent.agent_id}" in agents_section:
            template = dict(agents_section[f"sub_{parent.agent_id}"])
            chosen_profile = f"sub_{parent.agent_id}"
        elif "sub_default" in agents_section:
            template = dict(agents_section["sub_default"])
            chosen_profile = "sub_default"
        else:
            template = {}
            chosen_profile = "fallback"

        if agent_id_override is None:
            agent_id_override = f"{parent.agent_id}_sub_unknown"

        super().__init__(
            agent_id=agent_id_override,
            loader=parent.loader,
            app_root=parent.app_root,
            config_override=template,
            sub_agent=True,
        )

        self.message_queue = asyncio.Queue()
        self.dispatcher = parent.agent_id
        self.assigned_tools = assigned_tools or None
        self.task_context = parent.context_manager.create_task_context(task_name)

        self.temperature = temperature or self.config.get("temperature", 0.25)
        self.max_iterations = max_iterations or self.config.get("max_iterations", 15)
        self.max_tokens = self.config.get("max_tokens", 20000)
        self.max_reflections = self.config.get("max_reflections", 4)
        self.max_sub_agents = self.config.get("max_sub_agents", 4)

        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        elif not self.system_prompt.strip():
            self.system_prompt = (
                "You are a precise sub-agent. Complete the assigned task efficiently "
                "using only the tools you have been given."
            )

        self.attach_tools(assigned_tools=assigned_tools)

        if hasattr(parent, "workflow") and parent.workflow is not None:
            self.workflow.inherit_workflows_from_parent(parent.workflow)
        else:
            logger.warning(
                f"Parent agent has no workflow engine - sub-agent '{self.name}' may miss workflows"
            )

        logger.info(
            f"[SUB-AGENT] '{self.name}' created successfully | "
            f"profile='{chosen_profile}' | "
            f"tools loaded: {len(assigned_tools) if assigned_tools else 'full set'}"
        )
