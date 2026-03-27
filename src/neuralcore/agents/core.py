import yaml
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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

logger = Logger.get_logger()


class Agent:
    def __init__(
        self,
        agent_id: str,
        loader,
        app_root: Path,
        config_file: Optional[Path] = None,
        config_override: Optional[dict] = None,
        sub_agent: bool = False,
    ):
        self.agent_id: str = agent_id
        self.loader = loader
        self.app_root = app_root

        # === CONFIG HANDLING (supports both main agents and SubAgent override) ===
        self.sub_agent: bool = sub_agent
        if config_override is not None:
            logger.debug(
                f"Using config_override for agent '{agent_id}' (sub-agent mode)"
            )
            self.config = dict(config_override)
        else:
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

        self.registry = registry
        self.manager = DynamicActionManager(registry)
        self.context_manager = ContextManager(self.max_tokens)

        self.agent_tools = AgentActionHelper(self)
        self.workflow = WorkflowEngine(self)
        ToolBrowser(registry, self.manager)
        logger.debug("ToolBrowser registered")

        # Sub-agent flags (default for main agents)
        self.dispatcher: str = "user"
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._input_event: asyncio.Event = asyncio.Event()
        self._input_counter: int = 0
        self.max_sub_agents = self.config.get("max_sub_agents", 6)
        self.sub_tasks: Dict[str, Dict[str, Any]] = {}
        self._sub_task_counter: int = 0
        self.task_context: Optional["ContextManager.TaskContext"] = None
        self.assigned_tools: Optional[List[str]] = None

        # === UPGRADE: Current task/role, default workflow, background support ===
        self.current_task: str = ""
        self.current_role: str = self.config.get("role", "general_assistant")
        self.default_workflow: str = self.config.get("default_workflow", "default")
        self._status: str = "idle"
        self._background_task: Optional[asyncio.Task] = None

        # Production-ready: every Agent (main or sub) starts with clean internal state
        self._reset_state()

    def _load_agent_config(self, agent_id: str, config_file: Optional[Path]) -> dict:
        logger.debug(f"Loading config for agent_id='{agent_id}'")
        if config_file:
            logger.debug(f"Using config file: {config_file}")
            if config_file.exists():
                with open(config_file, "r") as f:
                    full_cfg = yaml.safe_load(f)
                logger.debug(
                    f"Config file loaded, keys at top level: {list(full_cfg.keys())}"
                )
                cfg = full_cfg.get("agents", {}).get(agent_id, {})
            else:
                logger.warning(f"Config file does not exist: {config_file}")
                cfg = {}
        else:
            logger.debug("Using loader.config")
            loader_config_agents = getattr(self.loader, "config", {}).get("agents", {})
            logger.debug(
                f"Agents in loader.config: {list(loader_config_agents.keys())}"
            )
            cfg = loader_config_agents.get(agent_id, {})

        if not cfg:
            logger.error(f"Agent '{agent_id}' not found in config")
            raise ValueError(f"Agent '{agent_id}' not found")
        logger.debug(f"Loaded config for agent '{agent_id}': {list(cfg.keys())}")
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

        # UPGRADE: reset runtime state
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
        else:
            item = {"role": "user", "content": str(message)}

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
        else:
            item = {"role": "system", "content": str(message)}

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
        else:
            item = {"event": "custom_control", "payload": str(control)}

        await self.message_queue.put(item)
        logger.debug(f"Agent '{self.name}' ← control posted: {str(control)[:80]}...")

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
        system_prompt = system_prompt or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self._reset_state()
        if stop_event is None:
            stop_event = asyncio.Event()

        # UPGRADE: track current task + use default workflow
        self.current_task = user_prompt or "headless processing"
        self._status = "running"

        if chat_mode:
            logger.info(
                f"Agent '{self.name}' → Starting CHAT mode with workflow 'deploy_chat'"
            )
            if user_prompt and str(user_prompt).strip():
                await self.post_message(user_prompt)

            async for event, payload in self.workflow.run(
                user_prompt="",
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_event=stop_event,
                workflow="deploy_chat",
            ):
                yield event, payload
            self._status = "idle"
            return

        logger.info(f"Agent '{self.name}' → Starting HEADLESS mode")
        target_workflow = workflow or self.default_workflow

        try:
            if user_prompt and str(user_prompt).strip():
                await self.post_message(user_prompt)

            while True:
                if stop_event.is_set():
                    logger.debug("Headless run stopped via stop_event")
                    break

                try:
                    msg = await asyncio.wait_for(self.message_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue

                if isinstance(msg, dict) and "event" in msg:
                    event = msg.get("event")
                    logger.debug(f"Headless received control: {event}")

                    if event == "switch_workflow":
                        wf_name = msg.get("name", self.default_workflow)
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

                if isinstance(msg, dict):
                    content = msg.get("content") or str(msg)
                else:
                    content = str(msg).strip()

                if not content:
                    self.message_queue.task_done()
                    continue

                logger.debug(f"Headless processing prompt: {content[:150]}...")

                async for event, payload in self.workflow.run(
                    user_prompt=content,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_event=stop_event,
                    workflow=target_workflow,
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
            self._status = "idle"
            self.current_task = ""
            logger.info(f"Agent '{self.name}' headless run finished")

    async def start_complex_deployment(
        self,
        task_description: str,
        user_facing_name: Optional[str] = None,
        sub_profile: Optional[str] = None,
        assigned_tools: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> str:
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
            "assigned_tools": assigned_tools or [],
            "progress": 0,
            "task_obj": None,
            "result": None,
            "error": None,
        }

        try:
            sub_agent = SubAgent(
                parent=self,
                task_name=display_name,
                assigned_tools=assigned_tools,
                custom_system_prompt=custom_system_prompt,
                temperature=temperature,
                max_iterations=max_iterations,
                profile=sub_profile,
                agent_id_override=f"{self.agent_id}_sub_{self._sub_task_counter}",
            )

            sub_agent.task = task_description
            sub_agent.goal = task_description
            # UPGRADE: propagate current task to sub-agent
            sub_agent.current_task = task_description
            sub_agent.current_role = f"sub-agent:{display_name}"

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
        task_ctx = getattr(sub_agent, "task_context", None)

        try:
            assigned = getattr(sub_agent, "assigned_tools", None)

            # === FORCE STRICT ISOLATION + TOOL RESTRICTION ===
            if assigned:
                logger.info(
                    f"[SUB-AGENT ISOLATION] Restricting to {len(assigned)} tools: {assigned[:10]}"
                )
                sub_agent.manager.unload_all()
                sub_agent.manager.load_tools(assigned)
                # Safe core tools only
                for t in ["GetContext", "GetDeploymentStatus"]:
                    if (
                        t in sub_agent.registry.all_actions
                        and not sub_agent.manager.is_loaded(t)
                    ):
                        sub_agent.manager.load_tools([t])
            else:
                logger.warning(
                    "[SUB-AGENT] No assigned_tools - using minimal core only"
                )
                sub_agent.manager.unload_all()
                sub_agent.manager.load_tools(["GetContext", "GetDeploymentStatus"])

            # Strong isolated prompt - sub-agent knows NOTHING about other steps
            sub_system = f"""You are a precise sub-agent executing **ONE single step only**.

            TASK: {task_description}
            """

            async for event, payload in sub_agent.workflow.run(
                user_prompt=task_description,
                system_prompt=sub_system,
                workflow="sub_agent_execute",
                temperature=0.25,
                max_tokens=10000,
            ):
                if event == "tool_result" and task_ctx is not None:
                    result_str = str(payload.get("result", ""))
                    if result_str.strip():
                        await task_ctx.add_important_result(
                            title=f"{payload.get('name', 'Tool')} Result",
                            content=result_str[:800],
                            source=payload.get("name", "tool"),
                            metadata=payload.get("args", {}),
                        )

            # === CLEAN NOISE: Keep only important data added via add_external_content ===
            self.context_manager.prune_sub_agent_noise()

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

            signal = {
                "event": "sub_task_completed",
                "task_id": task_id,
                "step": self.sub_tasks[task_id].get("step_number"),
                "summary": summary[:500],
                "success": True,
            }

            await self.post_control(signal)

            await self.post_system_message(
                f"✅ Step completed: {self.sub_tasks[task_id].get('display_name', task_id)}\n"
                f"{summary[:300]}{'...' if len(summary) > 300 else ''}"
            )

        except asyncio.CancelledError:
            self.sub_tasks[task_id].update(
                {
                    "status": "cancelled",
                    "completed_at": asyncio.get_event_loop().time(),
                }
            )
            await self.post_control(
                {"event": "sub_task_failed", "task_id": task_id, "error": "cancelled"}
            )
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
            await self.post_control(
                {"event": "sub_task_failed", "task_id": task_id, "error": str(exc)}
            )
            logger.error(f"Sub-task {task_id} failed", exc_info=True)

    def attach_tools(self, assigned_tools: Optional[List[str]] = None):
        if getattr(self, "sub_agent", False) and assigned_tools:
            if assigned_tools:
                logger.info(
                    f"[SUB-AGENT] '{self.name}' loading {len(assigned_tools)} specialized tools: "
                    f"{assigned_tools[:10]}{'...' if len(assigned_tools) > 10 else ''}"
                )
                self.manager.load_tools(assigned_tools)
            else:
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
            tool_sets = self.config.get("tool_sets", [])
            if tool_sets:
                logger.debug(f"Loading tool sets for main agent: {tool_sets}")
                self.loader.load_tool_sets(sets_to_load=tool_sets)

            for action_name in list(self.registry.all_actions.keys()):
                try:
                    self.manager.load_tools([action_name])
                except Exception as e:
                    logger.warning(f"Failed to load tool '{action_name}': {e}")

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
    """
    Production-ready specialized sub-agent for background / deployment steps.
    - Full inheritance from Agent → all queues, events, state management, and public API work.
    - Shares the parent's ContextManager exactly as required.
    - Loads only the assigned_tools (restricted toolset).
    - Clean, safe initialization with no __new__ hack.
    """

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
        # === CONFIG / PROFILE SELECTION (reused logic from old _create_sub_agent) ===
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

        # Agent ID for logging / uniqueness
        if agent_id_override is None:
            agent_id_override = f"{parent.agent_id}_sub_unknown"

        # === PROPER INHERITANCE (queues, events, manager, workflow, etc. are all initialized) ===
        super().__init__(
            agent_id=agent_id_override,
            loader=parent.loader,
            app_root=parent.app_root,
            config_override=template,
            sub_agent=True,
        )

        # Sub-agent specific wiring
        self.dispatcher = parent.agent_id
        self.assigned_tools = assigned_tools or []

        # Make name more descriptive for logs/UI
        if task_name and len(task_name) > 0:
            self.name = f"{self.name} ({task_name[:40]})"

        # === SHARED CONTEXT MANAGER (exactly as requested) ===
        self.context_manager = parent.context_manager
        self.task_context = self.context_manager.create_task_context(task_name)

        # Sub-agent tuned defaults (override anything from template)
        self.temperature = (
            temperature
            if temperature is not None
            else self.config.get("temperature", 0.25)
        )
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else self.config.get("max_iterations", 15)
        )
        self.max_tokens = self.config.get("max_tokens", 20000)
        self.max_reflections = self.config.get("max_reflections", 4)
        self.max_sub_agents = self.config.get("max_sub_agents", 4)

        # System prompt priority
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        elif not self.system_prompt.strip():
            self.system_prompt = (
                "You are a precise sub-agent. Complete the assigned task efficiently "
                "using only the tools you have been given."
            )

        # === TOOL LOADING (restricted set for this sub-agent) ===
        self.attach_tools(assigned_tools=assigned_tools)

        logger.info(
            f"[SUB-AGENT] '{self.name}' created successfully | "
            f"profile='{chosen_profile}' | "
            f"tools loaded: {len(assigned_tools) if assigned_tools else 'full set'}"
        )
