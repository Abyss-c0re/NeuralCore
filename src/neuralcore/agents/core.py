import yaml
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from neuralcore.utils.logger import Logger
from neuralcore.actions.manager import (
    ActionRegistry,
    AgentActionHelper,
    ToolBrowser,
    DynamicActionManager,
    registry,
)
from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.workflows.registry import workflow
from neuralcore.cognition.memory import ContextManager
from neuralcore.clients.factory import get_clients
from neuralcore.actions.manager import tool
from neuralcore.agents.state import AgentState

logger = Logger.get_logger()


# ================================ TOOLS =======================================
# Enabling agent to use own methods as tools (search context. deploy sub agents)


@tool("ContextManager", name="GetContext", description="Search your own memory")
async def provide_context(agent, query: str):
    parent_agent = getattr(agent, "parent", None)
    target_agent = (
        parent_agent if (getattr(agent, "sub_agent", False) and parent_agent) else agent
    )
    context_manager = getattr(target_agent, "context_manager", None)

    if context_manager is None:
        raise AttributeError(
            f"Target agent '{target_agent}' (ID: {getattr(target_agent, 'agent_id', 'unknown')}) "
            f"missing required 'context_manager' attribute."
        )
    provide_method = getattr(context_manager, "provide_context", None)

    if provide_method is None:
        raise AttributeError(
            f"ContextManager of agent '{target_agent}' missing required 'provide_context' method."
        )
    try:
        result = await provide_method(query)
        return result
    except (TypeError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to await context_manager.provide_context on agent '{target_agent}'. "
            f"Error: {e}"
        ) from e


@tool(
    "DeployControls",
    name="DeploySubAgent",
    description="""Deploy a focused sub-agent to handle one micro-task.

    Parameters:
    - task_description: Clear, specific description of what this step should do.
    - assigned_tools: Optional list of tool names...
    - depends_on_task_id: Optional task ID this one depends on.
    - user_facing_name: Optional friendly name for logs/UI.
    """,
    hidden_for_subagents=True,
)
async def deploy_sub_agent(
    agent,
    task_description: str,
    assigned_tools: Optional[Union[List[str], str]] = None,
    depends_on_task_id: Optional[str] = None,
    user_facing_name: Optional[str] = None,
) -> str:
    """Safely clean assigned_tools and launch sub-agent with recursion protection."""

    # === NEURALCORE RECURSION GUARD ===
    if agent.is_sub_agent():
        parent = agent.get_parent_agent()
        if parent is not None:
            logger.warning(
                f"[DEPLOY GUARD] Sub-agent '{agent.name}' attempted to call DeploySubAgent. "
                f"Blocked to prevent recursion. Parent: {parent.name}"
            )
            return (
                "⛔ DeploySubAgent is not available to sub-agents to prevent infinite recursion. "
                "Complete the current micro-task using only the tools you were assigned."
            )

    # === Existing defensive cleaning of assigned_tools (unchanged) ===
    if assigned_tools is None:
        cleaned_tools = None
    elif isinstance(assigned_tools, str):
        s = assigned_tools.strip()
        if not s:
            cleaned_tools = None
        elif s.startswith("[") and s.endswith("]"):
            try:
                import json

                cleaned_tools = json.loads(s)
            except Exception:
                cleaned_tools = [s.strip("[]\"' ")]
        else:
            cleaned_tools = [s.strip("\"' ")]
    elif isinstance(assigned_tools, (list, tuple)):
        cleaned_tools = [str(t).strip() for t in assigned_tools if str(t).strip()]
    else:
        cleaned_tools = None

    if cleaned_tools:
        cleaned_tools = [t for t in cleaned_tools if isinstance(t, str) and t.strip()]
        if not cleaned_tools:
            cleaned_tools = None

    tool_list = cleaned_tools[:8] if cleaned_tools else []
    logger.info(
        f"[DeploySubAgent] task='{task_description[:80]}...' | "
        f"assigned_tools={tool_list} | depends_on={depends_on_task_id}"
    )

    # Proceed with deployment only for main agents
    task_id = await agent.start_complex_deployment(
        task_description=task_description,
        user_facing_name=user_facing_name,
        assigned_tools=cleaned_tools,
        depends_on=depends_on_task_id,
        temperature=0.25,
    )

    dep_info = f" (depends on {depends_on_task_id})" if depends_on_task_id else ""
    return f"✅ Launched sub-task `{task_id}`{dep_info}: {task_description[:80]}..."


@tool("DeployControls", name="GetDeploymentStatus", hidden_for_subagents=True)
async def get_deployment_status(agent, task_id: Optional[str] = None):
    # Auto-detect current task if no ID provided
    # if not task_id:
    #     if hasattr(self, "state") and getattr(self.state, "sub_task_ids", None):
    #         task_id = self.state.sub_task_ids[0] if self.state.sub_task_ids else None
    #     elif hasattr(self.state, "task_id_map") and self.state.task_id_map:
    #         task_id = list(self.state.task_id_map.values())[-1]

    # Lookup in sub_tasks
    status = agent.sub_tasks.get(task_id) if task_id else None

    # Fallback: task_id exists in task_id_map but not yet in sub_tasks
    if not status and task_id:
        return f"⚠ Task ID `{task_id}` registered but not yet active. Try again in a few seconds."

    if status:
        output = [
            f"**Task ID:** `{task_id}`",
            f"**Step:** {status.get('step_number', '?')}",
            f"**Name:** {status.get('display_name', 'Unnamed Task')}",
            f"**Status:** {status.get('status', 'unknown').upper()}",
            f"**Runtime:** {status.get('runtime_seconds', 0):.1f} seconds",
            f"**Progress:** {status.get('progress', 0)}%",
            f"**Description:** {status.get('description', '')[:280]}...",
        ]

        if status.get("assigned_tools"):
            output.append(
                f"**Assigned Tools:** {', '.join(status['assigned_tools'][:10])}..."
            )

        if status.get("status") in ("completed", "failed") and status.get("result"):
            output.append(f"\n**Result:**\n{status['result']}")
        if status.get("error"):
            output.append(f"\n**Error:** {status['error']}")

        return "\n".join(output)

    # Full overview if no specific task found
    tasks = agent.get_sub_tasks()
    if not tasks:
        return "No background deployments running at the moment."

    lines = ["# 📊 **Deployment Status Overview**"]
    total = len(tasks)
    running = sum(1 for t in tasks.values() if t.get("status") == "running")
    completed = sum(1 for t in tasks.values() if t.get("status") == "completed")
    failed = sum(
        1 for t in tasks.values() if t.get("status") in ("failed", "cancelled")
    )

    lines.append(
        f"**Progress:** {completed}/{total} steps completed | Running: {running} | Failed: {failed}"
    )
    if hasattr(agent, "task") and agent.state.task:
        lines.append(f"\n**Main Task:** {agent.state.task}")
    if hasattr(agent.workflow, "current_workflow"):
        lines.append(f"**Current Stage:** {agent.workflow.current_workflow_name}")

    lines.append("\n**Steps:**")
    for t in sorted(tasks.values(), key=lambda x: x.get("started_at", 0)):
        emoji = {
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "pending": "⏳",
        }.get(t.get("status", "").lower(), "•")
        line = f"{emoji} `{t['id']}` → **{t.get('status', 'unknown').upper()}** — {t.get('display_name', '')}"
        if t.get("runtime_seconds", 0) > 5:
            line += f" ({t['runtime_seconds']:.1f}s)"
        lines.append(line)

    if completed == total and total > 0:
        lines.append("\n✅ **All steps completed successfully.**")

    return "\n".join(lines)


class Agent:
    def __init__(
        self,
        agent_id: str,
        loader,
        app_root: Path,
        config_file: Optional[Path] = None,
        config_override: Optional[dict] = None,
        sub_agent: bool = False,
        action_registry: ActionRegistry = registry,
    ):
        self.agent_id: str = agent_id
        self.loader = loader
        self.app_root = app_root
        self.registry = action_registry
        self.state: AgentState = AgentState()

        # === CONFIG HANDLING — now uses universal parser (no duplication) ===
        self.sub_agent: bool = sub_agent
        if config_override is not None:
            logger.debug(
                f"Using config_override for agent '{agent_id}' (sub-agent mode)"
            )
            self.config = dict(config_override)
        else:
            if config_file:
                # parse full config, then extract the specific agent section
                full_config = self.loader.parse_config(config_file)
                agent_cfg = full_config.get("agents", {}).get(agent_id, {})
                self.config = dict(agent_cfg) if isinstance(agent_cfg, dict) else {}
            else:
                # classic loader path — get_agent_config already returns the dict
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
            self.client.system_prompt if hasattr(self.client, "system_prompt") else "",
        )

        self.manager = DynamicActionManager(self.registry, self)
        self.context_manager = ContextManager(self.max_tokens)

        self.agent_tools = AgentActionHelper(self)
        self.workflow = WorkflowEngine(self, workflow)
        ToolBrowser(self.registry, self.manager)
        logger.debug("ToolBrowser registered")

        # Sub-agent flags
        self.dispatcher: str = "user"
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0
        self.max_sub_agents = self.config.get("max_sub_agents", 6)
        self.sub_tasks: Dict[str, Dict[str, Any]] = {}
        self._sub_task_counter: int = 0
        self.task_context: Optional["ContextManager.TaskContext"] = None
        self.assigned_tools: Optional[List[str]] = None

        self.current_task: str = ""
        self.current_role: str = self.config.get("role", "general_assistant")
        self.default_workflow: str = self.config.get("default", "default")
        self._status: str = "idle"
        self._background_task: Optional[asyncio.Task] = None
        self._reset_state()

    def is_sub_agent(self) -> bool:
        return getattr(self, "sub_agent", False)

    def get_parent_agent(self) -> Optional["Agent"]:
        return getattr(self, "parent", None)

    def reload_config(self, new_config: dict | str | Path) -> None:
            """Allow NeuralLabs to push a new config dict/string/Path and re-wire the agent instantly."""
            try:
                parsed = self.loader.parse_config(new_config)
                # Support both full config and direct agent dict
                if isinstance(parsed.get("agents"), dict):
                    agent_cfg = parsed["agents"].get(self.agent_id, {})
                else:
                    agent_cfg = parsed  # direct agent dict passed

                if isinstance(agent_cfg, dict) and agent_cfg:
                    self.config = dict(agent_cfg)
                    # Re-apply important runtime attributes
                    self.name = self.config.get("name", self.name)
                    self.temperature = self.config.get("temperature", self.temperature)
                    self.max_tokens = self.config.get("max_tokens", self.max_tokens)
                    self.system_prompt = self.config.get("system_prompt", self.system_prompt)
                    logger.info(f"Agent '{self.name}' reloaded config from NeuralLabs")
                else:
                    logger.warning("reload_config received invalid or empty agent section")
            except Exception as e:
                logger.error(f"reload_config failed: {e}")

    def _load_agent_tools(self):
        tool_sets = self.config.get("tool_sets", [])
        self.loader.load_tool_sets(sets_to_load=tool_sets)
        for action_name in registry.all_actions:
            self.manager.load_tools([action_name])

    def _resolve_workflow(
        self, chat_mode: bool = False, workflow_override: Optional[str] = None
    ) -> str:
        """
        Return the workflow to use.
        Priority:
          1. Explicit override
          2. workflow key from agent config (string or dict form)
          3. First workflow defined in config.yaml
          4. Never fall back to literal string "default"
        """
        if workflow_override:
            return workflow_override

        # 1. Check explicit "workflow" key in agent config (this is what you set in config.yaml)
        workflow_cfg = self.config.get("workflow")
        if isinstance(workflow_cfg, str):
            return workflow_cfg.strip()
        if isinstance(workflow_cfg, dict) and workflow_cfg:
            return next(iter(workflow_cfg.keys()))

        # 2. Fallback: use first workflow from the full config (via loader)
        if hasattr(self.loader, "config"):
            global_workflows = self.loader.config.get("workflows", {})
            if global_workflows:
                return next(iter(global_workflows.keys()))

        # Ultimate safe fallback - should never reach here with proper config
        logger.warning(
            "No workflow found in agent config or global config. Using first available."
        )
        return "deploy_chat"  # or raise, but we keep it running

    def _reset_state(self):
        """Reset both AgentState and legacy agent attributes."""
        # Reset the rich state object (this is the source of truth)
        self.state.reset_for_new_task()
        self.executed_signatures: set[tuple] = set()
        self.steps: List[str] = []
        self._stop_event: Optional[asyncio.Event] = None
        self.current_task = ""
        self._status = "idle"

    # ---------------- PUBLIC API ----------------

    async def post_message(self, message: str | Dict[str, Any]) -> None:
        """Post a message as USER."""
        if isinstance(message, str):
            item = {"role": "user", "content": message}
        elif isinstance(message, dict):
            item = {"role": "user", **message}
            if "role" not in item or item["role"] != "user":
                item["role"] = "user"

        await self.message_queue.put(item)
        self._input_counter += 1
        self._input_event.set()
        logger.debug(
            f"Agent '{self.name}' ← user message posted: {str(message)[:80]}..."
        )

    async def post_system_message(self, message: str | Dict[str, Any]) -> None:
        if isinstance(message, str):
            item = {"role": "system", "content": message}
        elif isinstance(message, dict):
            item = {"role": "system", **message}

        await self.message_queue.put(item)
        logger.debug(
            f"Agent '{self.name}' ← system alert posted: {str(message)[:100]}..."
        )
        self._input_event.set()

    async def post_control(self, control: str | Dict[str, Any]) -> None:
        if isinstance(control, str):
            item = {"event": control}
        elif isinstance(control, dict):
            item = dict(control)
            if not any(k in item for k in ("event", "action", "control")):
                item["event"] = "custom_control"

        await self.message_queue.put(item)
        logger.debug(f"Agent '{self.name}' ← control posted: {str(control)[:80]}...")

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
        """Retrieve current agent status with rich LLM-generated summary.
        Uses ContextManager.get_context_summary() as requested."""
        try:
            # Use the rich summary already available in ContextManager
            context_summary = self.context_manager.get_context_summary(
                max_messages=8, max_chars=1200
            )

            status_prompt = f"""You are a status reporter for an AI agent.

            Agent: {self.name} (ID: {self.agent_id})
            Role: {self.current_role}
            Current Task: {self.current_task or "None / Idle"}
            Status: {self._status}
            Default Workflow: {self.default_workflow}
            Background Mode: {"Yes" if self._background_task and not self._background_task.done() else "No"}
            Active Sub-tasks: {len(self.sub_tasks)}

            === CONTEXT SUMMARY ===
            {context_summary}

            Provide a clear, friendly 4-7 sentence status report for the user. 
            Include what the agent is working on, any progress, and next expected steps."""

            summary = await self.client.chat(
                [{"role": "user", "content": status_prompt}],
                temperature=0.3,
                max_tokens=700,
            )

            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "role": self.current_role,
                "task": self.current_task,
                "status": self._status,
                "default_workflow": self.default_workflow,
                "background_running": bool(
                    self._background_task and not self._background_task.done()
                ),
                "sub_tasks_count": len(self.sub_tasks),
                "sub_tasks": self.get_sub_tasks(),
                "context_summary": context_summary[:800] + "..."
                if len(context_summary) > 800
                else context_summary,
                "llm_summary": summary.strip(),
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.warning(f"get_agent_status failed for {self.name}: {e}")
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "status": self._status,
                "task": self.current_task,
                "error": str(e),
                "llm_summary": "Status summary generation failed.",
            }

    async def run_background(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        workflow: Optional[str] = None,
        **kwargs,
    ) -> asyncio.Task:
        """Run the agent fully in the background.
        Bidirectional communication:
          • Input  → post_message / post_system_message / post_control
          • Output → events are automatically forwarded as control messages
        Returns the background task so it can be awaited/cancelled.
        """
        if self._background_task and not self._background_task.done():
            logger.warning(f"Background task already active for agent '{self.name}'")
            return self._background_task

        self.current_task = user_prompt or "background processing"
        self._status = "running_background"

        async def _background_consumer():
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
                    # Forward every workflow event as a control message (bidirectional comms)
                    control_msg = {
                        "event": "background_event",
                        "type": event,
                        "payload": payload
                        if isinstance(payload, dict)
                        else {"content": str(payload)},
                    }
                    await self.post_control(control_msg)
                    logger.debug(f"[BG] {self.name} → {event}")

            except asyncio.CancelledError:
                logger.info(f"Background run for '{self.name}' was cancelled")
                raise
            except Exception as exc:
                logger.error(f"Background run error in '{self.name}'", exc_info=True)
                await self.post_control(
                    {"event": "background_error", "error": str(exc)}
                )
            finally:
                self._status = "idle"
                self.current_task = ""
                self._background_task = None
                logger.info(f"Agent '{self.name}' background run finished")

        self._background_task = asyncio.create_task(
            _background_consumer(), name=f"bg_{self.agent_id}"
        )
        logger.info(
            f"Agent '{self.name}' started in BACKGROUND mode (task={self._background_task.get_name()})"
        )
        return self._background_task

    async def execute_loop(
        self, loop_name: str, initial_state: Optional[dict] = None, **kwargs
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Delegate loop execution to the WorkflowEngine and yield events"""
        async for event, payload in self.workflow.execute_loop(
            loop_name, initial_state, **kwargs
        ):
            yield event, payload

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
            self.state.goal = user_prompt

        self.current_task = user_prompt or (
            "chat session" if chat_mode else "headless processing"
        )
        self._status = "running"

        workflow_name = self._resolve_workflow(
            chat_mode=chat_mode, workflow_override=workflow
        )
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
        """Handle control events from the message queue.
        Returns True if the run should terminate."""
        event = msg.get("event")
        logger.debug(f"Headless received control: {event}")

        if event == "switch_workflow":
            wf_name = msg.get("name", self.default_workflow)
            logger.info(f"Switching workflow to: {wf_name}")
            try:
                self.workflow.switch_workflow(wf_name)
            except Exception as e:
                logger.warning(f"Workflow switch failed: {e}")
            return False

        elif event in ("finish", "cancelled", "break"):
            logger.info(f"Termination signal received: {event}")
            return True

        return False

    async def _run_headless_loop(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        workflow_name: str,
        stop_event: asyncio.Event,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Continuous headless consumer with proper bootstrap + timeout safety."""
        processed_initial = False

        while True:
            if stop_event.is_set():
                logger.debug("Headless run stopped via stop_event")
                break

            try:
                # Short timeout on first iteration to catch the seeded message immediately
                timeout = 1.0 if not processed_initial else 30.0
                msg = await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if not processed_initial and self.message_queue.empty():
                    logger.warning(
                        "[HEADLESS] No initial message after bootstrap — idling"
                    )
                continue

            # Control message?
            if isinstance(msg, dict) and "event" in msg:
                should_stop = await self._handle_control_message(msg)
                self.message_queue.task_done()
                if should_stop:
                    break
                continue

            # Normal user message
            content = msg.get("content") if isinstance(msg, dict) else str(msg).strip()
            if not content:
                self.message_queue.task_done()
                continue

            logger.info(f"[HEADLESS] Processing prompt: {content[:120]}...")
            processed_initial = True

            async for event, payload in self._run_workflow_once(
                user_prompt=content,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workflow_name=workflow_name,
                stop_event=stop_event,
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
        stop_event = stop_event or asyncio.Event()

        (
            system_prompt,
            temperature,
            max_tokens,
            workflow_name,
        ) = self._setup_for_run(
            user_prompt, system_prompt, temperature, max_tokens, workflow, chat_mode
        )

        # ← NEW: Ensure tools are loaded before any headless execution
        if not chat_mode:
            logger.info(f"[HEADLESS] Ensuring tools for workflow '{workflow_name}'")
            self.manager.reset_to_default_package("headless_bootstrap", self.workflow)
            self.attach_tools()  # safe, respects assigned_tools for sub-agents

        try:
            if chat_mode:
                logger.info(
                    f"Agent '{self.name}' → Starting CHAT mode with workflow '{workflow_name}'"
                )
                if user_prompt and str(user_prompt).strip():
                    await self.post_message(user_prompt)

                async for event, payload in self._run_workflow_once(
                    user_prompt="",
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    workflow_name=workflow_name,
                    stop_event=stop_event,
                ):
                    yield event, payload

            else:
                # HEADLESS MODE — now properly bootstrapped
                logger.info(
                    f"Agent '{self.name}' → Starting HEADLESS mode with workflow '{workflow_name}'"
                )

                if user_prompt and str(user_prompt).strip():
                    await self.post_message(user_prompt)
                    logger.debug(
                        f"[HEADLESS] Seeded initial prompt: {user_prompt[:100]}..."
                    )

                async for event, payload in self._run_headless_loop(
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    workflow_name=workflow_name,
                    stop_event=stop_event,
                ):
                    yield event, payload

        finally:
            self._status = "idle"
            self.current_task = ""
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
            sub_agent.state.goal = task_description
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
                sub_system = f"""You are a precise sub-agent executing one focused micro-task.

            Task: {task_description}

            Rules:
            - Complete ONLY this exact task.
            - Use only the tools you were given.
            - When finished, output a short summary and end with exactly: [FINAL_ANSWER_COMPLETE]
            - Never mention other steps."""

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

    def attach_tools(self, assigned_tools: Optional[List[str]] = None):
        """Load tools for main agent or sub-agent."""
        if getattr(self, "sub_agent", False) and assigned_tools:
            # Sub-agent: only load the explicitly assigned tools
            valid_tools = [t for t in assigned_tools if t in registry.all_actions]
            if valid_tools:
                self.manager.unload_all()  # clear previous
                self.manager.load_tools(valid_tools)
                logger.info(
                    f"[SUB-AGENT] '{self.name}' loaded {len(valid_tools)} valid tools: {valid_tools}"
                )
            else:
                logger.warning(
                    f"[SUB-AGENT] '{self.name}' – no valid tools in assigned list: {assigned_tools}"
                )
                # Minimal safe fallback for sub-agents
                core = ["FindTool", "GetContext"]
                self.manager.load_tools([t for t in core if t in registry.all_actions])
        else:
            # Main agent: load full tool_sets from config
            tool_sets = self.config.get("tool_sets", [])
            if tool_sets:
                logger.debug(f"Loading tool sets for main agent: {tool_sets}")
                self.loader.load_tool_sets(sets_to_load=tool_sets)

            # Load everything from registry (safe for main agents)
            for action_name in list(registry.all_actions.keys()):
                try:
                    self.manager.load_tools([action_name])
                except Exception as e:
                    logger.warning(f"Failed to load tool '{action_name}': {e}")

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
            return f"✅ The deployment task **{task}** has been completed successfully."


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
