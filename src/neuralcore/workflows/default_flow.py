# Demo & Test Harness:
# This class is currently used to demonstrate framework features and actively test/improve their performance.
# While not yet part of the core framework, it is scheduled for migration into NeuralVoid.
# Use this for validation and exploration; its structure may change as it is finalized for integration into the new system.

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
    logger.info(f"[RequestComplexAction] Complex task: {reason[:100]}...")
    agent.task = reason
    agent.goal = reason

    # Switch orchestrator
    try:
        agent.workflow.engine.switch_workflow("orchestrator")
    except Exception:
        await agent.post_control({"event": "switch_workflow", "name": "orchestrator"})

    task_id = await agent.start_complex_deployment(
        task_description=reason,
        user_facing_name=reason[:60] + "...",
        assigned_tools=None,
        temperature=0.3,
    )

    return (
        f"✅ Starting **multi-step orchestration** for:\n"
        f"**{reason}**\n\n"
        f"Task ID: `{task_id}` — breaking it down into focused steps..."
    )


@tool("DeployControls", name="GetDeploymentStatus")
async def get_deployment_status(agent, task_id: Optional[str] = None):
    # Auto-detect current task if no ID provided
    if not task_id:
        if hasattr(agent, "state") and getattr(agent.state, "sub_task_ids", None):
            task_id = agent.state.sub_task_ids[0] if agent.state.sub_task_ids else None
        elif hasattr(agent.state, "task_id_map") and agent.state.task_id_map:
            task_id = list(agent.state.task_id_map.values())[-1]

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
    if hasattr(agent, "task") and agent.task:
        lines.append(f"\n**Main Task:** {agent.task}")
    if hasattr(agent.workflow, "current_workflow"):
        lines.append(f"**Current Stage:** {agent.workflow.current_workflow}")

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


class AgentFlow:
    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    class Phase(str, Enum):
        IDLE = "idle"
        CHAT = "chat"
        PLAN = "plan"
        EXECUTE = "execute"
        WAIT = "wait"
        FINALIZE = "finalize"

    def __init__(self, engine):
        self.engine = engine
        self.agent = engine.agent
        self._register_workflows()

    def _register_workflows(self):
        # Battle-tested: Chat mode
        self.engine.register_workflow(
            name="deploy_chat",
            description="Natural chat → RequestComplexAction switches to orchestrator",
            steps=["deploy_chat_loop"],
            hidden_toolsets=["DeployControls"],
        )

        # Battle-tested: Sub-agent micro-task execution
        self.engine.register_workflow(
            name="sub_agent_execute",
            description="Focused ReAct for ONE micro-task only",
            steps=["llm_stream", "execute_if_tools", "sub_agent_check_complete"],
            hidden_toolsets=["DeployControls"],
        )

        # New clean orchestrator (uses your current AgentState fields)
        self.engine.register_workflow(
            name="orchestrator",
            description="One-shot plan with tool hints → launch restricted sub-agents → finalize",
            steps=[
                "plan_microtasks",
                "launch_next_subtask",
                "wait_for_subtask",
                "check_orchestrator_complete",
            ],
        )

        # Register all _wf_* handlers
        for attr_name in dir(self):
            if attr_name.startswith("_wf_"):
                step_name = attr_name[4:]
                self.engine._step_handlers[step_name] = getattr(self, attr_name)

    # ==================== SYSTEM PROMPTS ====================

    def _build_chat_system_prompt(self) -> str:
        return f"""You are a helpful Deploy Agent.
        Speak naturally and concisely.
        - Simple questions → answer directly.
        - Complex requests → call **RequestComplexAction**.
        - When you see [DEPLOYMENT COMPLETE], respond friendly.

        Current goal: {self.agent.goal or "General assistance"}"""

    def _build_sub_agent_system_prompt(
        self, task_desc: str, assigned_tools: List[str]
    ) -> str:
        tools_hint = (
            f"\n\nAvailable tools: {', '.join(assigned_tools[:15])}{', ...' if len(assigned_tools) > 15 else ''}"
            if assigned_tools
            else ""
        )
        return f"""You are a precise sub-agent executing **ONE single micro-task only**.

TASK: {task_desc}{tools_hint}

CRITICAL RULES:
- Complete ONLY this exact task.
- If the task involves reading a file, use open_file_async or open_file_sync directly. Do NOT rely on browse_tools for obvious file operations.
- When you have finished the task, output a short summary and end with exactly: {AgentFlow.FINAL_ANSWER_MARKER}
- Never mention other steps or the overall project."""

    # ==================== CHAT LOOP (battle-tested - unchanged) ====================

    @workflow.set(
        "deploy_chat",
        name="deploy_chat_loop",
        toolsets=["DeployControls"],
        dynamic_allowed=True,
    )
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": "chat"})
            logger.info(f"Agent '{self.agent.name}' → Chat mode started")

        self.agent.manager.load_toolsets("DeployControls")

        while True:
            try:
                raw_msg = await asyncio.wait_for(
                    self.agent.message_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                if (
                    getattr(self.agent, "_stop_event", None)
                    and self.agent._stop_event.is_set()
                ):
                    break
                continue
            except asyncio.CancelledError:
                break

            if isinstance(raw_msg, dict) and "event" in raw_msg:
                ev = raw_msg["event"]
                if ev in ("sub_task_completed", "sub_task_failed"):
                    yield (ev, raw_msg)
                    await self.agent.post_system_message(
                        f"[STEP {ev.replace('sub_task_', '').upper()}] {raw_msg.get('task_id')}"
                    )
                elif ev == "switch_workflow":
                    self.engine.switch_workflow(raw_msg.get("name", "deploy_chat"))
                self.agent.message_queue.task_done()
                continue

            content = (
                raw_msg.get("content", "")
                if isinstance(raw_msg, dict)
                else str(raw_msg)
            )
            if not content.strip():
                self.agent.message_queue.task_done()
                continue

            messages = await self.agent.context_manager.provide_context(
                query=content, chat=True, system_prompt=self._build_chat_system_prompt()
            )

            async for ev, pl in self._process_user_message_with_llm(messages, state):
                yield ev, pl

            self.agent.message_queue.task_done()

    # ==================== NEW ORCHESTRATOR (matches your AgentState) ====================

    @workflow.set("orchestrator", name="plan_microtasks")
    async def _wf_plan_microtasks(self, iteration: int, state: AgentState):
        if state.planned_tasks:  # already planned
            return

        state.phase = self.Phase.PLAN
        yield ("phase_changed", {"phase": "plan"})

        prompt = f"""Break this task into 4-8 small independent micro-tasks.

TASK: {self.agent.task}

For each micro-task suggest the most relevant tools (e.g. open_file_async, write_file, grep, replace_block, etc.).

Return ONLY JSON:
{{
  "microtasks": [
    {{"description": "...", "suggested_tools": ["tool1", "tool2"]}},
    ...
  ]
}}"""

        raw = await self.agent.client.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(raw)
            state.planned_tasks = [t["description"] for t in data.get("microtasks", [])]
            state.task_tool_assignments = {
                i: t.get("suggested_tools", [])
                for i, t in enumerate(data.get("microtasks", []))
            }
        except Exception:
            state.planned_tasks = [self.agent.task]
            state.task_tool_assignments = {0: []}

        state.current_task_index = 0
        state.task_id_map = {}
        yield (
            "info",
            f"Planned {len(state.planned_tasks)} micro-tasks with tool hints",
        )

    @workflow.set("orchestrator", name="launch_next_subtask")
    async def _wf_launch_next_subtask(self, iteration: int, state: AgentState):
        """
        Launch all remaining micro-tasks in parallel for the current task index.
        Each micro-task is mapped to a sub-agent.
        Ensures tasks are registered in self.agent.sub_tasks before continuing.
        """
        if state.current_task_index >= len(state.planned_tasks):
            state.is_complete = True
            return

        tasks_to_launch = list(
            enumerate(
                state.planned_tasks[state.current_task_index :],
                start=state.current_task_index,
            )
        )

        launched_ids = []

        for idx, task_desc in tasks_to_launch:
            assigned_tools = state.task_tool_assignments.get(idx, [])
            name = f"Step {idx + 1}/{len(state.planned_tasks)}: {task_desc[:55]}..."

            task_id = await self.agent.start_complex_deployment(
                task_description=task_desc,
                user_facing_name=name,
                assigned_tools=assigned_tools or None,
                temperature=0.25,
                custom_system_prompt=self._build_sub_agent_system_prompt(
                    task_desc, assigned_tools
                ),
            )

            # Wait for the task to appear in sub_tasks
            wait_time = 0.0
            while task_id not in self.agent.sub_tasks and wait_time < 5.0:
                await asyncio.sleep(0.05)
                wait_time += 0.05

            if task_id not in self.agent.sub_tasks:
                logger.warning(f"Task {task_id} not registered in sub_tasks after 5s.")

            # Register in orchestrator state
            launched_ids.append(task_id)
            state.task_id_map[idx] = task_id

            # Update step number safely
            if task_id in self.agent.sub_tasks:
                self.agent.sub_tasks[task_id]["step_number"] = idx + 1

            yield (
                "sub_agent_launched",
                {
                    "step": idx + 1,
                    "task_id": task_id,
                    "description": task_desc,
                    "assigned_tools": assigned_tools,
                },
            )
            logger.info(
                f"Launched sub-task {idx + 1} → {task_id} with tools: {assigned_tools}"
            )

        state.sub_task_ids = launched_ids
        state.current_task_index = len(state.planned_tasks)

    @workflow.set("orchestrator", name="wait_for_subtask")
    async def _wf_wait_for_subtask(self, iteration: int, state: AgentState):
        """
        Wait for all currently launched sub-tasks in state.sub_task_ids to complete.
        Handles multiple sub-tasks running in parallel.
        """
        if not state.sub_task_ids:
            return

        pending_tasks = set(state.sub_task_ids)

        while pending_tasks:
            for task_id in list(pending_tasks):
                task = self.agent.sub_tasks.get(task_id)
                if task and task.get("status") in ("completed", "failed", "cancelled"):
                    yield (
                        "subtask_done",
                        {"task_id": task_id, "status": task.get("status")},
                    )
                    pending_tasks.remove(task_id)
                else:
                    yield ("waiting_for_subtask", {"task_id": task_id})
            await asyncio.sleep(0.1)

        # Once all sub-tasks are done, clear the sub_task_ids list
        state.sub_task_ids = []

        # Current task index can now advance to the end of the batch
        state.current_task_index = len(state.planned_tasks)

    @workflow.set("orchestrator", name="check_orchestrator_complete")
    async def _wf_check_orchestrator_complete(self, iteration: int, state: AgentState):
        if state.current_task_index < len(state.planned_tasks):
            state.is_complete = False
            return

        # All done
        state.phase = self.Phase.FINALIZE
        yield ("phase_changed", {"phase": "finalize"})

        summary = await self._generate_user_friendly_summary(state)
        yield ("llm_response", {"full_reply": summary, "is_complete": True})
        await self.agent.context_manager.add_message("assistant", summary)

        await self.agent.post_control(
            {"event": "switch_workflow", "name": "deploy_chat"}
        )

        yield (
            "finish",
            {
                "reason": "orchestrator_complete",
                "total_steps": len(state.planned_tasks),
            },
        )

    @workflow.set(
        "sub_agent_execute",
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
            return

        # Normal reply — be more cautious
        elif not final_reply and tool_results:
            final_reply = "\n\n".join(tool_results)
        elif not final_reply:
            final_reply = "✅ Tool executed successfully."

        # Add a safety note if we suspect partial progress
        if "README" in complex_reason or "main.py" in complex_reason:
            final_reply += "\n\n(Note: This is part of a multi-step process. More steps may still be running.)"

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
