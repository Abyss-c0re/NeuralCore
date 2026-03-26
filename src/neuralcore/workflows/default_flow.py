import re
import asyncio
import json
from enum import Enum
from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet
from neuralcore.actions.manager import tool
from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger

from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

logger = Logger.get_logger()


# ==================== TOOLS ====================


@tool("ContextManager", name="GetContext", description="Search your own memory")
async def provide_context(agent, query: str):
    return await agent.context_manager.provide_context(query)


@tool(
    "DeployControls",
    name="RequestComplexAction",
    description="Use when user request requires planning, tools, multiple steps, research or code changes.",
)
async def request_complex_action(agent, reason: str):
    """Force the full multi-step orchestrator."""
    logger.info(f"[RequestComplexAction] Received complex task: {reason[:100]}...")

    agent.task = reason
    agent.goal = reason

    # Stronger switch to default workflow
    try:
        # Direct engine switch
        if hasattr(agent.workflow, "engine") and hasattr(
            agent.workflow.engine, "switch_workflow"
        ):
            agent.workflow.engine.switch_workflow("default")
            logger.info("Successfully forced switch to 'default' (multi-step planner)")
        else:
            # Fallback control message
            await agent.post_control(
                {
                    "event": "switch_workflow",
                    "name": "default",
                    "reason": reason,
                }
            )
            logger.info("Posted switch_workflow control to default")
    except Exception as e:
        logger.error(f"Failed to switch to default workflow: {e}")

    # Start the deployment as backup
    task_id = await agent.start_complex_deployment(
        task_description=reason,
        user_facing_name=reason[:60] + "...",
        assigned_tools=None,
        temperature=0.3,
    )

    return (
        f"✅ Starting complex task with full planning.\n\n"
        f"**Task:** {reason}\n"
        f"**ID:** `{task_id}`\n\n"
        f"Breaking this down into steps and executing sequentially..."
    )


@tool(
    "DeployControls",
    name="ExitDeployMode",
    description="Call when the deployment task is fully completed.",
)
async def exit_deploy_mode(agent, reason: str = "Task completed"):
    await agent.post_control(
        {"event": "finish", "reason": "deploy_mode_exit", "details": reason}
    )
    return f"✅ Deploy session ended: {reason}"


@tool("DeployControls", name="GetDeploymentStatus")
async def get_deployment_status(agent, task_id: Optional[str] = None):
    """Check status of one specific background deployment or get full overview.

    Now includes plan progress, current stage, and better formatting.
    """
    if task_id:
        # === SINGLE TASK DETAIL ===
        status = agent.sub_tasks.get(task_id)
        if not status:
            return f"❌ Task ID `{task_id}` not found."

        runtime = status.get("runtime_seconds", 0)
        progress = status.get("progress", 0)
        step_num = status.get("step_number", "?")  # we'll add this later

        output = [
            f"**Task ID:** `{task_id}`",
            f"**Step:** {step_num}",
            f"**Name:** {status.get('display_name', 'Unnamed Task')}",
            f"**Status:** {status.get('status', 'unknown').upper()}",
            f"**Runtime:** {runtime:.1f} seconds",
            f"**Progress:** {progress}%",
            f"**Description:** {status.get('description', '')[:280]}{'...' if len(status.get('description', '')) > 280 else ''}",
        ]

        assigned = status.get("assigned_tools", [])
        if assigned:
            output.append(
                f"**Assigned Tools:** {', '.join(assigned[:10])}{'...' if len(assigned) > 10 else ''}"
            )

        if status.get("status") in ("completed", "failed"):
            if status.get("result"):
                output.append(f"\n**Result:**\n{status['result']}")
            if status.get("error"):
                output.append(f"\n**Error:** {status['error']}")

        return "\n".join(output)

    else:
        # === FULL OVERVIEW ===
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
            f"**Progress:** {completed}/{total} steps completed | "
            f"Running: {running} | Failed: {failed}"
        )

        if hasattr(agent, "task") and agent.task:
            lines.append(f"\n**Main Task:** {agent.task}")

        # Current stage
        if hasattr(agent.workflow, "current_workflow"):
            lines.append(f"**Current Stage:** {agent.workflow.current_workflow}")

        # List all sub-tasks
        lines.append("\n**Steps:**")
        sorted_tasks = sorted(tasks.values(), key=lambda x: x.get("started_at", 0))

        for t in sorted_tasks:
            status_str = t.get("status", "unknown").upper()
            emoji = {
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "pending": "⏳",
            }.get(status_str.lower(), "•")

            runtime = t.get("runtime_seconds", 0)
            line = (
                f"{emoji} `{t['id']}` → **{status_str}** — {t.get('display_name', '')}"
            )

            if runtime > 5:
                line += f" ({runtime:.1f}s)"
            lines.append(line)

        # Overall completion
        if completed == total and total > 0:
            lines.append("\n✅ **All steps completed successfully.**")

        return "\n".join(lines)


class AgentFlow:
    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    class Phase(str, Enum):
        IDLE = "idle"
        CHAT = "chat"
        PLAN = "plan"
        EXECUTE = "execute"
        REFLECT = "reflect"
        DECISION = "decision"
        FINALIZE = "finalize"

    # ── Two separate workflows ─────────────────────────────────────
    def __init__(self, engine):
        self.engine = engine
        self.agent = engine.agent
        self._register_workflows()

    def _register_workflows(self):
        # 1. Default = Full ReAct agentic workflow (planning + tools)
        self.engine.register_workflow(
            name="default",
            description="Plan → Deploy one sub-agent per task → Wait → Validate",
            steps=[
                "plan_tasks",
                "deploy_sub_agents",  # NEW
                "wait_for_sub_agents",  # NEW
                "verify_goal_completion",
                "check_complete",
                "reflect_if_stuck",
                "safety_fallback",
            ],
        )

        # ── SUB-AGENT WORKFLOW (classic ReAct) ──
        self.engine.register_workflow(
            name="sub_agent_execute",
            description="Full ReAct loop for sub-agents (LLM + tools + reflection)",
            steps=[
                "llm_stream",
                "execute_if_tools",
                "verify_goal_completion",
                "check_complete",  # will behave differently for sub-agents
                "reflect_if_stuck",
                "replan_if_reflected",
                "safety_fallback",
            ],
        )
        # 2. Deploy Chat = Persistent conversation mode (this is what we want by default)
        self.engine.register_workflow(
            name="deploy_chat",
            description="Persistent natural chat mode. Switches to full planning on complex requests.",
            steps=["deploy_chat_loop"],  # ← only this step, loops via queue
        )

        # Register all _wf_* handlers
        for attr_name in dir(self):
            if attr_name.startswith("_wf_"):
                step_name = attr_name[4:]
                method = getattr(self, attr_name)
                if callable(method):
                    self.engine._step_handlers[step_name] = method

    # ==================== SYSTEM PROMPTS ====================
    def _build_chat_system_prompt(self) -> str:
        return f"""You are a helpful Deploy Agent.
        Speak naturally, concisely and friendly.
        Rules:
        - For simple questions → just reply normally.
        - If user asks for something complex → call RequestComplexAction.
        - When you see a message containing [DEPLOYMENT COMPLETE] or [DEPLOYMENT FAILED], 
        read it carefully and give the user a friendly, clear response about the result.
        Do NOT call tools unless the user asks for something new.

        Current goal: {self.agent.goal or "General assistance"}

        """

    # ==================== CHAT LOOP STEP ====================

    @workflow.set("deploy_chat", name="deploy_chat_loop")
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        """Clean persistent chat loop with safe queue handling."""

        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": state.phase.value})
            logger.info(f"Agent '{self.agent.name}' → Persistent chat mode started")

        self.agent.manager.load_toolsets("DeployControls")

        while True:
            logger.debug("Chat loop: awaiting next message...")

            try:
                raw_msg = await self.agent.message_queue.get()
            except asyncio.CancelledError:
                logger.info("Chat loop cancelled")
                break

            # We will call task_done() exactly once per get() in the finally block
            try:
                # ====================== CONTROL MESSAGE HANDLING ======================
                if isinstance(raw_msg, dict) and "event" in raw_msg:
                    event = raw_msg.get("event")
                    logger.debug(f"Control event received: {event}")

                    if event == "switch_workflow":
                        workflow_name = raw_msg.get("name", "default")
                        reason = raw_msg.get("reason", "")

                        logger.info(
                            f"Switching workflow to: {workflow_name} | reason: {reason}"
                        )

                        try:
                            self.engine.switch_workflow(workflow_name)
                        except AttributeError:
                            logger.warning(
                                "WorkflowEngine.switch_workflow not available, using post_control"
                            )
                            await self.agent.post_control(raw_msg)
                        except Exception as e:
                            logger.error(f"Workflow switch failed: {e}")

                        continue  # Let finally call task_done()

                    elif event == "cancelled":
                        if (
                            hasattr(self.agent.client, "_current_stop_event")
                            and self.agent.client._current_stop_event
                        ):
                            self.agent.client._current_stop_event.set()
                        logger.info("Cancelled current generation via control")
                        continue

                    else:
                        logger.debug(f"Unknown control event: {event}")
                        continue

                # ====================== NORMAL MESSAGE PROCESSING ======================
                if isinstance(raw_msg, dict):
                    role = raw_msg.get("role", "user")
                    content = raw_msg.get("content", "") or str(raw_msg)
                else:
                    role = "user"
                    content = str(raw_msg).strip()

                if not content and role != "system":
                    continue

                # === FAST PATH: Background deployment results ===
                if role == "system" and (
                    "[DEPLOYMENT COMPLETE]" in content
                    or "[DEPLOYMENT FAILED]" in content
                ):
                    logger.debug("Fast-path: delivering background deployment result")
                    yield (
                        "llm_response",
                        {
                            "full_reply": content,
                            "tool_calls": [],
                            "is_complete": True,
                            "is_system_alert": True,
                        },
                    )
                    await self.agent.context_manager.add_message("system", content)
                    await self.agent.context_manager.add_message("assistant", content)
                    continue

                # === NORMAL PATH: User message → LLM + tools ===
                logger.debug(
                    f"Processing normal message/role={role}: {content[:100]}..."
                )

                messages = await self.agent.context_manager.provide_context(
                    query=content,
                    max_input_tokens=self.agent.max_tokens,
                    reserved_for_output=8000,
                    system_prompt=self._build_chat_system_prompt(),
                    include_logs=False,
                    chat=True,
                )

                yield ("phase_changed", {"phase": self.Phase.CHAT.value})

                logger.debug("→ Calling _process_user_message_with_llm")

                async for ev, pl in self._process_user_message_with_llm(
                    messages, state
                ):
                    if ev in ("tool_delta", "tool_call_delta"):
                        continue
                    yield (ev, pl)

            finally:
                # CRITICAL: Call task_done() exactly once for every .get()
                self.agent.message_queue.task_done()
                logger.debug("task_done() called for this message")

    @workflow.set(
        "default",
        name="plan_tasks",
        description="Generates steps + per-step tool searches and assignments.",
    )
    async def _wf_plan_tasks(
        self, iteration: int, state: "AgentState"
    ) -> AsyncIterator:
        if not state.planned_tasks:
            state.phase = self.Phase.PLAN
            yield ("phase_changed", {"phase": state.phase.value})

            # Step 1: Generate fine-grained tasks (keep or refine your existing prompt)
            plan_prompt = f"""
            Break the following TASK into the smallest possible independent actionable steps.
            Each step should be completable by one focused sub-agent.

            TASK:
            {self.agent.task}

            Return ONLY JSON:
            {{"tasks": ["step 1 description", "step 2 description", ...]}}
            """
            raw_plan = await self.agent.client.chat(
                [{"role": "user", "content": plan_prompt}]
            )

            try:
                plan_json = json.loads(raw_plan)
                state.planned_tasks = plan_json.get("tasks", [])
            except Exception:
                # fallback parsing as before...
                state.planned_tasks = []

            if not state.planned_tasks:
                yield ("warning", "No tasks planned.")
                return

            state.current_task_index = 0
            state.task_tool_assignments = {}

            # Step 2: For EACH task, generate a targeted tool search query + search registry
            yield (
                "info",
                f"Planning {len(state.planned_tasks)} steps with per-step tool discovery...",
            )

            for i, task_desc in enumerate(state.planned_tasks):
                # Generate a precise search query for this step only
                query_prompt = f"""
                Generate 1-3 short, precise search queries to find the MOST RELEVANT tools 
                from the registry for this specific sub-task:

                SUB-TASK: {task_desc}

                Requirements:
                - Focus on capabilities needed (e.g., "web search", "code execution", "file read", "API call", etc.)
                - Be specific to the action described.
                - Return ONLY JSON: {{"queries": ["query1", "query2"]}}
                """

                raw_queries = await self.agent.client.chat(
                    [{"role": "user", "content": query_prompt}]
                )
                try:
                    queries = json.loads(raw_queries).get(
                        "queries", [task_desc]
                    )  # fallback to task desc
                except Exception:
                    queries = [task_desc]

                # Parallel search
                async def search_for_step(q: str):
                    return await asyncio.to_thread(
                        self.agent.registry.search, q, limit=10
                    )

                search_results = await asyncio.gather(
                    *[search_for_step(q) for q in queries]
                )

                # Rank tools (simple frequency + first occurrence)
                tool_scores: Dict[str, float] = {}
                for results in search_results:
                    for action, set_name in results:
                        name = action.name
                        tool_scores[name] = tool_scores.get(name, 0) + 1.0

                # Take top N
                best_tools = sorted(
                    tool_scores.items(), key=lambda x: x[1], reverse=True
                )[:8]
                assigned_tool_names = [name for name, _ in best_tools]

                state.task_tool_assignments[i] = assigned_tool_names

                yield (
                    "step_tool_assignment",
                    {
                        "step": i + 1,
                        "task": task_desc[:120],
                        "assigned_tools": assigned_tool_names,
                        "tool_count": len(assigned_tool_names),
                    },
                )

                logger.info(
                    f"Step {i + 1}: Assigned {len(assigned_tool_names)} tools → {assigned_tool_names}"
                )

        # Yield next task (as before, but now with tool assignment awareness)
        while state.planned_tasks and state.current_task_index < len(
            state.planned_tasks
        ):
            idx = state.current_task_index
            next_task = state.planned_tasks[idx]
            assigned_tools = state.task_tool_assignments.get(idx, [])

            await self.agent.context_manager.add_message(
                "system",
                f"[NEXT TASK] {next_task}\n[RECOMMENDED TOOLS] {', '.join(assigned_tools) or 'none'}",
            )

            state.current_task_index += 1
            yield (
                "planned_task",
                {"task": next_task, "assigned_tools": assigned_tools},
            )

    @workflow.set("default", name="deploy_sub_agents")
    async def _wf_deploy_sub_agents(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator:
        """Deploy ONLY the next pending step. This allows progressive execution."""
        if not state.planned_tasks:
            yield ("warning", "No planned tasks.")
            return

        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        if state.current_task_index >= len(state.planned_tasks):
            state.is_complete = True
            yield ("info", "All steps have been deployed.")
            return

        i = state.current_task_index
        task_desc = state.planned_tasks[i]
        assigned_tools = state.task_tool_assignments.get(i, [])

        name = f"Step {i + 1}/{len(state.planned_tasks)}: {task_desc[:50]}..."

        yield ("info", f"🚀 Deploying step {i + 1}/{len(state.planned_tasks)} ...")

        task_id = await self.agent.start_complex_deployment(
            task_description=task_desc,
            user_facing_name=name,
            assigned_tools=assigned_tools,
            temperature=0.25,
        )

        state.sub_task_ids = [task_id]  # only current one
        state.task_id_map[i] = task_id

        # Store step number for better status reporting
        if task_id in self.agent.sub_tasks:
            self.agent.sub_tasks[task_id]["step_number"] = i + 1

        yield (
            "sub_agent_deployed",
            {
                "task_id": task_id,
                "step": i + 1,
                "total_steps": len(state.planned_tasks),
                "description": task_desc,
                "assigned_tools": assigned_tools,
            },
        )

        logger.info(f"Deployed step {i + 1} → task {task_id}")

    @workflow.set("default", name="wait_for_sub_agents")
    async def _wf_wait_for_sub_agents(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator:
        """Wait for the current step only. When it finishes, signal to continue."""
        if not state.sub_task_ids:
            yield ("info", "No active sub-agent.")
            return

        yield ("phase_changed", {"phase": self.Phase.EXECUTE.value})

        current_task_id = state.sub_task_ids[0]  # we only run one at a time now

        if current_task_id in self.agent.sub_tasks:
            task_info = self.agent.sub_tasks[current_task_id]
            task_obj = task_info.get("task_obj")

            if task_obj and not task_obj.done():
                logger.info(
                    f"Waiting for step {state.current_task_index + 1} ({current_task_id})"
                )
                try:
                    await asyncio.wait_for(task_obj, timeout=900)
                except asyncio.TimeoutError:
                    logger.warning(f"Step {current_task_id} timed out")
                    task_info["status"] = "failed"

        # Check status
        status = self.agent.sub_tasks.get(current_task_id, {})
        step_status = status.get("status", "unknown")

        if step_status == "completed":
            yield (
                "step_completed",
                {
                    "step": state.current_task_index + 1,
                    "task_id": current_task_id,
                    "result": status.get("result", ""),
                },
            )
            state.current_task_index += 1
            state.sub_task_ids = []  # clear so next iteration can deploy next step

            if state.current_task_index >= len(state.planned_tasks):
                state.is_complete = True
                yield ("info", "✅ All steps completed!")
            else:
                yield (
                    "info",
                    f"Step {state.current_task_index} finished → preparing next step...",
                )

        else:
            yield (
                "step_failed",
                {"step": state.current_task_index + 1, "task_id": current_task_id},
            )
            state.is_complete = False

        yield (
            "progress_update",
            {
                "current_step": state.current_task_index,
                "total_steps": len(state.planned_tasks),
                "is_fully_complete": state.is_complete,
            },
        )

    @workflow.set(
        "default",
        name="llm_stream",
        description="Streams LLM response and extracts tool calls.",
    )
    async def _wf_llm_stream(self, iteration: int, state: AgentState):
        async for ev, pl in self._llm_stream_with_tools(
            iteration, state
        ):  # ← now self.
            if ev == "llm_response" and isinstance(pl, dict):
                state.full_reply = pl.get("full_reply", "")
                state.tool_calls = pl.get("tool_calls", [])
                state.is_complete = pl.get("is_complete", False)
            yield (ev, pl)
        self.engine._log_iteration_state(iteration, state)

    @workflow.set(
        "default",
        name="execute_if_tools",
        description="Executes tool calls if present.",
    )
    async def _wf_execute_if_tools(self, iteration: int, state: AgentState):
        if not state.tool_calls:
            return

        yield ("tool_calls", state.tool_calls)

        async for ev, pl in self._execute_tools(
            state.tool_calls, iteration, state
        ):  # ← now self.
            yield (ev, pl)

        state.tool_calls = []
        logger.debug(f"Cleared tool_calls after execution at iteration {iteration}")

        for r in self.agent.tool_results:
            sig = f"{r['name']}:{json.dumps(r['args'], sort_keys=True)}"
            self.agent.executed_signatures.add(sig)

        state.phase = self.Phase.DECISION
        yield ("phase_changed", {"phase": state.phase.value})

        self.engine._log_iteration_state(iteration, state)

    @workflow.set(
        "default",
        name="verify_goal_completion",
        description="Verifies goal completion.",
    )
    async def _wf_verify_goal_completion(
        self, iteration: int, state: "AgentState"
    ) -> AsyncIterator:
        """(unchanged — no executor calls)"""
        if not state.is_complete:
            return

        state.phase = self.Phase.DECISION
        yield ("phase_changed", {"phase": state.phase.value})
        yield ("goal_verification_start", {"iteration": iteration})

        summary = self.agent.context_manager.get_context_summary()
        investigation_state = self.agent.context_manager.investigation_state
        prompt = f"""
    You are a strict goal auditor.

    GOAL:
    {self.agent.goal}

    LIVE CONTEXT SUMMARY:
    {summary}

    INVESTIGATION STATE:
    {json.dumps(investigation_state, indent=2)}

    Has the agent FULLY completed the goal? Reply STRICT JSON ONLY:

    {{
        "verified": true or false,
        "reason": "one-sentence explanation",
        "missing_steps": ["list"] or []
    }}
    """

        raw_verification = await self.agent.client.chat(
            [{"role": "user", "content": prompt}]
        )

        verification: Dict[str, Any] = {}
        try:
            verification = json.loads(raw_verification.strip())
        except Exception:
            match = re.search(r"\{.*?\}", str(raw_verification), re.DOTALL)
            if match:
                try:
                    verification = json.loads(match.group())
                except Exception:
                    verification = {}
            else:
                verification = {}

        verification.setdefault("verified", False)
        verification.setdefault("reason", "parse failed" if not verification else "")
        verification.setdefault("missing_steps", [])

        yield ("goal_verification_result", verification)

        if not verification.get("verified", False):
            state.is_complete = False
            await self.agent.context_manager.add_message(
                "system", f"[GOAL VERIFICATION FAILED] {verification.get('reason')}"
            )
            yield ("info", {"message": "Goal verification FAILED — continuing"})
        else:
            yield ("info", {"message": "Goal verification PASSED"})

    @workflow.set("default", name="check_complete")
    async def _wf_check_complete(self, iteration: int, state: AgentState):
        """Only finish the entire deployment when ALL steps are done.
        Safely handles both chat_mode and headless mode.
        """
        if not state.is_complete:
            return

        # Extra safety check
        total_steps = len(getattr(state, "planned_tasks", []))
        if state.current_task_index < total_steps:
            logger.info(
                f"Still {total_steps - state.current_task_index} steps remaining. Continuing..."
            )
            state.is_complete = False
            return

        state.phase = self.Phase.FINALIZE
        yield ("phase_changed", {"phase": state.phase.value})

        is_sub = getattr(self.agent, "sub_agent", False)

        if is_sub:
            # Sub-agent finish
            summary = await self._generate_sub_agent_summary(state)
            yield ("llm_response", {"full_reply": summary, "is_complete": True})
            await self.agent.context_manager.add_message("assistant", summary)
            yield ("finish", {"reason": "sub_agent_task_complete"})
            return

        # === MAIN AGENT FULL COMPLETION ===
        try:
            friendly_summary = await self._generate_user_friendly_summary(state)
            yield (
                "llm_response",
                {"full_reply": friendly_summary, "is_complete": True},
            )
            await self.agent.context_manager.add_message("assistant", friendly_summary)
        except Exception:
            friendly_summary = (
                f"✅ All {total_steps} steps of the deployment completed successfully."
            )
            yield (
                "llm_response",
                {"full_reply": friendly_summary, "is_complete": True},
            )
            await self.agent.context_manager.add_message("assistant", friendly_summary)

        # === SMART SWITCHING ===
        is_in_chat_mode = False
        try:
            # Check if we are currently running inside the chat loop
            is_in_chat_mode = (
                hasattr(self.engine, "current_workflow")
                and self.engine.current_workflow == "deploy_chat"
            )

            if is_in_chat_mode:
                logger.info("Complex task completed → switching back to chat mode")
                await self.agent.post_control(
                    {
                        "event": "switch_workflow",
                        "name": "deploy_chat",
                        "reason": "Complex deployment finished",
                    }
                )
            else:
                logger.info(
                    "Complex task completed in headless/script mode → finishing cleanly"
                )
        except Exception as e:
            logger.warning(f"Failed to switch workflow cleanly: {e}")

        yield (
            "finish",
            {
                "reason": "deploy_task_complete",
                "switched_back_to_chat": is_in_chat_mode,
                "total_steps": total_steps,
            },
        )

    @workflow.set(
        "default",
        name="reflect_if_stuck",
        description="Triggers reflection when no progress is detected.",
    )
    async def _wf_reflect_if_stuck(self, iteration: int, state: "AgentState"):
        """(unchanged except the call below)"""
        REFLECT_INTERVAL = 5
        last_reflect_iter = getattr(state, "last_reflection_iteration", 0)

        if iteration <= 3 or (iteration - last_reflect_iter) < REFLECT_INTERVAL:
            return

        current_snapshot = {
            "planned_tasks": list(state.planned_tasks) if state.planned_tasks else [],
            "current_task_index": state.current_task_index,
            "tool_calls": list(state.tool_calls) if state.tool_calls else [],
        }

        last_snapshot = getattr(state, "last_progress_snapshot", None)
        no_progress = last_snapshot is not None and last_snapshot == current_snapshot
        state.last_progress_snapshot = current_snapshot

        if not no_progress:
            return

        state.reflection_count = getattr(state, "reflection_count", 0) + 1
        max_reflections = getattr(self.agent, "max_reflections", 6)

        if state.reflection_count >= max_reflections:
            msg = (
                f"Hard reflection limit reached ({state.reflection_count}/{max_reflections}) "
                f"at iteration {iteration} — terminating to prevent loop"
            )
            logger.warning(msg)
            yield ("warning", msg)
            yield (
                "finish",
                {
                    "reason": "max_reflection_limit_reached",
                    "reflection_count": state.reflection_count,
                    "iteration": iteration,
                },
            )
            return

        state.last_reflection_iteration = iteration

        async for event, payload in self._force_reflection(
            iteration, state
        ):  # ← now self.
            yield (event, payload)

            if event != "reflection_decision":
                continue

            decision = payload
            next_step = decision.get("next_step")
            print(next_step)

            if next_step == "tool" and decision.get("tool_name"):
                tool_name = decision["tool_name"]
                args = decision.get("arguments", {})
                state.tool_calls = [
                    {"function": {"name": tool_name, "arguments": json.dumps(args)}}
                ]
                yield (
                    "info",
                    {
                        "message": f"[REFLECTION] Enqueued tool call: {tool_name}",
                        "tool_name": tool_name,
                    },
                )
                return

            elif next_step == "llm":
                await self.agent.context_manager.add_message(
                    "system",
                    f"[REFLECTION GUIDANCE at iter {iteration}]\n"
                    f"{json.dumps(decision, indent=2)}",
                )
                yield (
                    "info",
                    "Reflection added guidance message — continuing with LLM",
                )
                return

            elif next_step == "finish":
                reason = decision.get("reason", "reflection requested termination")
                yield (
                    "finish",
                    {
                        "reason": "reflection_requested_finish",
                        "details": reason,
                        "iteration": iteration,
                    },
                )
                return

            else:
                yield ("warning", f"Reflection returned invalid next_step: {next_step}")
                return

    @workflow.set(
        "default",
        name="replan_if_reflected",
        description="Resets task list after a reflection so plan_tasks can generate a better plan with new context.",
    )
    async def _wf_replan_if_reflected(self, iteration: int, state: AgentState):
        """(unchanged)"""
        last_reflect = getattr(state, "last_reflection_iteration", 0)
        last_replan = getattr(state, "last_replan_iteration", 0)

        if (
            last_reflect == 0
            or iteration <= last_reflect + 1
            or last_replan == iteration
        ):
            return

        state.planned_tasks = []
        state.current_task_index = 0
        state.last_replan_iteration = iteration

        yield (
            "replan_triggered",
            {
                "reason": "post_reflection",
                "original_reflection_iter": last_reflect,
                "new_iteration": iteration,
            },
        )
        yield (
            "info",
            f"[REPLAN] Task list cleared after reflection at iter {last_reflect}. "
            f"plan_tasks will run again on next cycle.",
        )

    @workflow.set(
        "default",
        name="safety_fallback",
        description="Stops execution after max iterations.",
    )
    async def _wf_safety_fallback(self, iteration: int, state: AgentState):
        if iteration >= getattr(self.agent, "max_iterations", 20):
            async for ev, pl in self._generate_final_summary(state):  # ← now self.
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})
            self.engine._log_iteration_state(iteration, state)

    # ===================================================================
    # EXECUTORS — NOW INSIDE AgentFlow (relocated)
    # ===================================================================

    async def _llm_stream_with_tools(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
        is_chat_mode: bool = False,
    ) -> AsyncIterator[Tuple[str, Any]]:

        if is_chat_mode:
            # Should never happen anymore — we use _process_user_message_with_llm in chat
            raise RuntimeError(
                "is_chat_mode=True should not reach _llm_stream_with_tools"
            )

        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        # Normal EXECUTE mode (ReAct / default workflow)
        messages = await self.agent.context_manager.provide_context(
            query=state.current_task or "Continue",
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=12000,
            system_prompt=self._build_objective_reminder(),
            include_logs=True,
        )

        # ====================== LLM STREAM + TOOL EXECUTION ======================
        async def executor_callback(name: str, args: dict):
            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                raise RuntimeError(f"No executor for tool '{name}'")
            maybe = executor(**args)
            return await maybe if asyncio.iscoroutine(maybe) else maybe

        queue = await self.agent.client.stream_with_tools(
            messages=messages,
            tools=tools or self.agent.manager.get_llm_tools(),
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
            tool_choice="auto",
            executor_callback=executor_callback,
        )

        text_buffer = ""
        tool_results = []

        try:
            async for item in self.agent.client._drain_queue(queue):
                if item is None:
                    continue
                if not isinstance(item, tuple) or len(item) != 2:
                    continue

                kind, payload = item

                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)

                elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                    if isinstance(payload, dict):
                        result = payload.get("result") or payload.get("output")
                        if isinstance(result, str) and result.strip():
                            tool_results.append(result.strip())

                elif kind == "finish":
                    break
                elif kind in ("error", "cancelled"):
                    yield (kind, payload)
                    return

        except asyncio.CancelledError:
            yield ("cancelled", "Task cancelled")
            return
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield ("error", str(e))
            return

        final_reply = text_buffer.strip()
        if not final_reply and tool_results:
            final_reply = "\n\n".join(tool_results)
        if not final_reply:
            final_reply = "✅ Tool executed successfully."

        yield (
            "llm_response",
            {
                "full_reply": final_reply,
                "tool_calls": [],
                "is_complete": True,
            },
        )

        await self.agent.context_manager.add_message("assistant", final_reply)

    async def _execute_tools(
        self, tool_calls: List[Dict], iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Executes tool calls (now only used for tools that needed ConfirmationRequired).
        ConfirmationRequired is already handled by the client, so this is a clean pass-through.
        Phase change is emitted and per-tool events are yielded exactly as before.
        """
        state.phase = self.Phase.EXECUTE
        yield (
            "phase_changed",
            {"phase": state.phase.value},
        )  # ← ensure phase is always signalled

        for call in tool_calls or []:
            # Support both old raw tool_calls and the new "needs_confirmation" payload
            if "function" in call:  # old format from legacy tool_calls
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except Exception:
                    args = {}
            else:  # new needs_confirmation payload from client
                name = call.get("tool_name")
                args = call.get("details", {}).get(
                    "args", {}
                )  # assuming ConfirmationRequired stores args

            if not name:
                yield ("tool_skipped", {"name": "unknown", "reason": "no_name"})
                continue

            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in self.agent.executed_signatures:
                continue
            self.agent.executed_signatures.add(sig)

            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                yield ("tool_skipped", {"name": name, "reason": "no_executor"})
                continue

            yield ("tool_start", {"name": name, "args": args})

            try:
                maybe = executor(**args)
                result = await maybe if asyncio.iscoroutine(maybe) else maybe

                await self.agent.context_manager.record_tool_outcome(
                    name, str(result), args
                )
                await self.agent.context_manager.add_message("tool", str(result))

                self.agent.tool_results.append(
                    {"name": name, "result": result, "args": args}
                )
                yield ("tool_result", {"name": name, "result": result})

            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                await self.agent.context_manager.record_tool_outcome(name, result, args)
                yield ("tool_result", {"name": name, "result": result, "error": True})

            stop_event = getattr(self.agent.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", f"Stop after {name}")
                return

    async def _force_reflection(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = self.Phase.REFLECT
        yield ("phase_changed", {"phase": state.phase.value})

        summary = self.agent.context_manager.get_context_summary()

        prompt = f"""
        Agent is stuck after {iteration} iterations.

        TASK:
        {self.agent.task}

        LIVE CONTEXT SUMMARY (truncated):
        {summary}

        CRITICAL: If the agent appears unable to make progress despite the context, 
        choose next_step: "finish" to avoid infinite loops. 
        Only choose "llm" if a clear next reasoning step will unstick it.
        Prefer suggesting a concrete tool or new_subtask when possible.

        Return valid JSON ONLY with keys:
        - reason: why the agent is stuck
        - next_step: "tool", "llm", or "finish"
        - optional: tool_name, arguments, new_subtask, workflow_adjustments
        """

        raw_response = await self.agent.client.ask(prompt)

        raw_str = str(raw_response).strip()
        try:
            if raw_str.startswith("```json"):
                raw_str = raw_str[7:].split("```")[0].strip()
            elif raw_str.startswith("```"):
                raw_str = raw_str[3:].split("```")[0].strip()
            decision = json.loads(raw_str)
        except Exception:
            decision = {"reason": "parse failed", "next_step": "finish"}

        if not isinstance(decision, dict) or decision.get("next_step") not in (
            "tool",
            "llm",
            "finish",
        ):
            decision = {"reason": "invalid decision", "next_step": "finish"}

        if decision.get("new_subtask"):
            new_task = decision["new_subtask"]
            remaining_tasks = state.planned_tasks[state.current_task_index :]
            state.planned_tasks = [new_task] + remaining_tasks
            state.current_task_index = 0
            self.agent.context_manager.add_subtask(new_task)
            logger.info(f"[REFLECTION] Inserted new subtask: {new_task}")

        if decision.get("reason"):
            self.agent.context_manager.add_finding(f"Reflection: {decision['reason']}")

        await self.agent.context_manager.add_message(
            "system", f"[REFLECTION]\n{json.dumps(decision, indent=2)}"
        )

        adjustments = decision.get("workflow_adjustments", {})
        if isinstance(adjustments, dict) and isinstance(
            adjustments.get("reorder_steps"), list
        ):
            self.engine.workflow_steps = [  # ← still on engine
                s
                for s in adjustments["reorder_steps"]
                if s in self.engine.workflow_steps
            ]

        state.last_reflection_decision = decision
        self.engine._log_iteration_state(iteration, state)  # ← still on engine

        yield ("reflection_decision", decision)
        yield ("reflection_triggered", decision)

    async def _generate_final_summary(
        self, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = self.Phase.FINALIZE
        yield ("phase_changed", {"phase": state.phase.value})

        lines = [
            "# 🏁 Agent Execution Report",
            f"**Task:** {self.agent.task[:200]}...",
            f"**Goal:** {self.agent.goal}",
            "",
            "## 📊 ContextManager Stats",
            f"- KB items: {len(self.agent.context_manager.knowledge_base)}",
            f"- Files checked: {len(self.agent.context_manager.files_checked)}",
            f"- Tools executed: {len(self.agent.context_manager.tools_executed)}",
            f"- Archived turns: {len(self.agent.context_manager.current_topic.archived_history)}",
            "",
            "## 🛠️ Tool Results (last 10)",
        ]
        for r in self.agent.tool_results[-10:]:
            lines.append(f"- {r['name']}: {str(r.get('result', ''))[:120]}...")

        final_text = "\n".join(lines)
        yield ("final_summary", final_text)
        yield ("finish", {"reason": "task_complete", "summary": final_text})

    # Optional helper used by _llm_stream_with_tools
    def _build_objective_reminder(self) -> str:
        """You can keep or move this helper if it exists elsewhere."""
        return f"Current goal: {self.agent.goal}"

    async def _process_user_message_with_llm(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Process message in chat mode. Handle complex action switching safely."""

        logger.debug("=== ENTERING _process_user_message_with_llm ===")

        async def executor_callback(name: str, args: dict):
            logger.debug(f"[TOOL CALL] Executing {name} with args {args}")
            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                raise RuntimeError(f"No executor for tool '{name}'")
            maybe = executor(**args)
            result = await maybe if asyncio.iscoroutine(maybe) else maybe
            return result

        if (
            hasattr(self.agent.client, "_current_stop_event")
            and self.agent.client._current_stop_event
        ):
            self.agent.client._current_stop_event.clear()

        queue = await self.agent.client.stream_with_tools(
            messages=messages,
            tools=self.agent.manager.get_action_set("DeployControls"),
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
            tool_choice="auto",
            executor_callback=executor_callback,
        )

        text_buffer = ""
        tool_results = []
        complex_action_called = False
        complex_reason = ""

        try:
            async for item in self.agent.client._drain_queue(queue):
                if item is None:
                    continue
                if not isinstance(item, tuple) or len(item) != 2:
                    continue

                kind, payload = item

                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)

                elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                    if isinstance(payload, dict):
                        tool_name = payload.get("tool_name") or payload.get("name")
                        if tool_name == "RequestComplexAction":
                            complex_action_called = True
                            complex_reason = payload.get("args", {}).get("reason", "")

                        result = payload.get("result") or payload.get("output")
                        if isinstance(result, str) and result.strip():
                            tool_results.append(result.strip())

                elif kind == "finish":
                    logger.debug("STREAM FINISHED")
                    break

                elif kind in ("error", "cancelled"):
                    yield (kind, payload)
                    return

        except asyncio.CancelledError:
            yield ("cancelled", "Task cancelled")
            return
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield ("error", str(e))
            return

        final_reply = text_buffer.strip()

        if complex_action_called:
            logger.info(
                f"[CHAT → ORCHESTRATOR] Complex task detected: {complex_reason[:100]}..."
            )

            self.agent.task = complex_reason
            self.agent.goal = complex_reason

            # Force switch to default workflow
            try:
                self.engine.switch_workflow("default")
                logger.info("Successfully switched to 'default' orchestrator workflow")
            except Exception as e:
                logger.warning(f"Direct switch failed: {e}. Using control fallback.")
                await self.agent.post_control(
                    {
                        "event": "switch_workflow",
                        "name": "default",
                        "reason": complex_reason,
                    }
                )

            final_reply = (
                f"✅ Understood. Starting **multi-step orchestration**:\n"
                f"**{complex_reason[:120]}{'...' if len(complex_reason) > 120 else ''}**\n\n"
                f"Planning steps → deploying specialized sub-agents sequentially."
            )

            yield (
                "llm_response",
                {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
            )
            await self.agent.context_manager.add_message("assistant", final_reply)

            logger.debug("Complex action handled — exiting chat loop processor")
            return  # Exit here. Let the outer loop handle task_done()

        # Normal reply (no complex action)
        if not final_reply and tool_results:
            final_reply = "\n\n".join(tool_results)
        elif not final_reply:
            final_reply = "✅ Tool executed successfully."

        logger.info(f"FINAL REPLY being sent to user:\n{final_reply}")

        yield (
            "llm_response",
            {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
        )
        await self.agent.context_manager.add_message("assistant", final_reply)
        logger.debug("Message added to context")

    async def _generate_user_friendly_summary(self, state: AgentState) -> str:
        """Generates a natural, friendly summary that will be shown to the user
        right before returning to chat mode."""

        tool_results_str = "\n".join(
            f"• {r['name']}: {str(r.get('result', ''))[:400]}"
            for r in self.agent.tool_results[-12:]  # last 12 results max
        )

        prompt = f"""You are a helpful Deploy Agent. The complex task has just finished.

    Task: {self.agent.task}
    Goal: {self.agent.goal or "General deployment assistance"}

    What was actually done (tool results):
    {tool_results_str or "No tool results recorded."}

    Write a **friendly, concise, natural** message to the user (2–6 sentences max).
    - Celebrate what was accomplished
    - Mention any important outcomes or warnings
    - End by saying we're back in normal chat mode and ask how else you can help

    Tone: professional but warm and clear. No JSON. No technical jargon unless necessary.
    """

        try:
            summary = await self.agent.client.chat(
                [{"role": "user", "content": prompt}], temperature=0.7
            )
            return summary.strip()
        except Exception:
            # Fallback
            return (
                f"✅ **Task completed successfully!**\n\n"
                f"I have finished the deployment task: **{self.agent.task}**.\n"
                f"We are now back in normal chat mode. How else can I help you?"
            )

    async def _generate_sub_agent_summary(self, state: AgentState) -> str:
        return f"✅ Sub-task completed.\n\nKey results recorded in shared context."
