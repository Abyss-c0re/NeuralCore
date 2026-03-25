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


@tool(
    "DeployControls",
    name="RequestComplexAction",
    description="Use when user request requires planning, tools, multiple steps, research or code changes.",
)
async def request_complex_action(agent, reason: str):
    """Switch from chat mode to full planning workflow"""
    await agent.post_control(
        {"event": "switch_workflow", "name": "default", "reason": reason}
    )
    logger.info(f"[Deploy Agent] Switching to full planning mode → Reason: {reason}")
    return f"✅ Switching to planning + execution mode for: {reason}"

    # ==================== TOOLS ====================


@tool("ContextManager", name="GetContext", description="Search your own memory")
async def provide_context(agent, query: str):
    return await agent.context_manager.provide_context(query)


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
            description="Full ReAct planning + execution",
            steps=[
                "plan_tasks",
                "llm_stream",
                "execute_if_tools",
                "verify_goal_completion",
                "check_complete",
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
Help the user with deployment, infrastructure, code, or operational tasks.

Rules:
- For simple questions or conversation → just reply normally.
- If the request needs planning, multiple steps, tools, research, or code changes → 
  call the tool `RequestComplexAction` with a short clear reason.
- Only call `ExitDeployMode` when the entire task is truly finished and user is done.

Current goal: {self.agent.goal or "General assistance"}
"""

    # ==================== CHAT LOOP STEP ====================
    @workflow.set("deploy_chat", name="deploy_chat_loop")
    async def _wf_deploy_chat_loop(self, iteration: int, state: AgentState):
        state.phase = self.Phase.CHAT
        yield ("phase_changed", {"phase": state.phase.value})

        # Only wait for user input starting from the second iteration
        if iteration > 0:
            logger.debug(
                f"Chat loop (iter {iteration}): waiting for next user message..."
            )
            # Waiting happens inside _llm_stream_with_tools
        self.agent.manager.load_toolsets("DeployControls")

        async for ev, pl in self._llm_stream_with_tools(
            iteration=iteration,
            state=state,
            tools=self.agent.manager.get_action_set("DeployControls"),
            is_chat_mode=True,
        ):
            yield (ev, pl)

        self.engine._log_iteration_state(iteration, state)

    @workflow.set(
        "default",
        name="plan_tasks",
        description="Generates tool queries and task plan.",
    )
    async def _wf_plan_tasks(
        self, iteration: int, state: "AgentState"
    ) -> AsyncIterator:
        """(unchanged)"""
        if not state.planned_tasks:
            state.phase = self.Phase.PLAN
            yield ("phase_changed", {"phase": state.phase.value})

            # ── Step 1: Generate queries
            query_prompt = f"""
            Agent must generate queries to identify tools from the registry
            that are relevant to accomplishing the following TASK:

            TASK:
            {self.agent.task}

            REQUIREMENTS:
            - Respond ONLY with a JSON array of short search queries.
            - NO explanation, NO extra text.
            JSON format: {{ "queries": ["query1", "query2", ...] }}
            """
            raw_queries = await self.agent.client.chat(
                [{"role": "user", "content": query_prompt}]
            )

            try:
                queries_list = json.loads(raw_queries).get("queries", [])
            except Exception:
                match = re.search(r"\{.*?\}", raw_queries, re.DOTALL)
                queries_list = []
                if match:
                    try:
                        queries_list = json.loads(match.group()).get("queries", [])
                    except Exception as e:
                        yield ("warning", f"Failed to parse queries JSON: {e}")

            if not queries_list:
                yield ("warning", "No queries generated; cannot plan tasks.")
                return

            # ── Step 2: Parallel registry search
            async def search_registry(query: str):
                return await asyncio.to_thread(self.agent.registry.search, query, 3)

            registry_tasks = [search_registry(q) for q in queries_list]
            search_results = await asyncio.gather(*registry_tasks)

            suggested_tools = list(
                dict.fromkeys(a.name for results in search_results for a, _ in results)
            )

            if suggested_tools:
                self.agent.manager.load_tools(suggested_tools)
                yield ("info", f"Loaded tools: {', '.join(suggested_tools)}")

            # ── Step 3: Generate plan
            tools_str = ", ".join(self.agent.manager._loaded_tools) or "none"
            plan_prompt = f"""
            Agent must plan a sequence of actionable steps to accomplish the TASK below.
            Only use tools available in the agent's dynamic toolset: {tools_str}.
            Respond ONLY with JSON. NO explanations.

            TASK:
            {self.agent.task}

            REQUIREMENTS:
            - Ordered list of concise, actionable steps.
            - Strict JSON format: {{ "tasks": ["step1", "step2", ...] }}
            """

            raw_plan = await self.agent.client.chat(
                [{"role": "user", "content": plan_prompt}]
            )

            try:
                plan_json = json.loads(raw_plan)
                state.planned_tasks = plan_json.get("tasks", [])
            except Exception:
                match = re.search(r"\{.*?\}", raw_plan, re.DOTALL)
                if match:
                    try:
                        plan_json = json.loads(match.group())
                        state.planned_tasks = plan_json.get("tasks", [])
                    except Exception as e:
                        state.planned_tasks = []
                        yield ("warning", f"Failed to parse plan JSON: {e}")
                else:
                    state.planned_tasks = []
                    yield ("warning", "No JSON found in planning response")

            state.current_task_index = 0

        # ── Step 4: Yield next task
        while state.planned_tasks and state.current_task_index < len(
            state.planned_tasks
        ):
            next_task = state.planned_tasks[state.current_task_index]
            await self.agent.context_manager.add_message(
                "system", f"[NEXT TASK REMINDER] {next_task}"
            )
            state.current_task_index += 1
            yield ("planned_task", {"task": next_task})

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

    @workflow.set(
        "default",
        name="check_complete",
        description="Determines if execution is complete.",
    )
    async def _wf_check_complete(self, iteration: int, state: AgentState):
        if iteration == 1 and not state.tool_calls and not state.planned_tasks:
            state.phase = self.Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            yield (
                "llm_response",
                {"full_reply": state.full_reply, "tool_calls": [], "is_complete": True},
            )
            yield ("finish", {"reason": "casual_complete"})
            self.engine._log_iteration_state(iteration, state)
            return

        if state.is_complete:
            state.phase = self.Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            async for ev, pl in self._generate_final_summary(state):  # ← now self.
                yield (ev, pl)
            yield ("finish", {"reason": "complete"})
            self.engine._log_iteration_state(iteration, state)

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
            state.phase = self.Phase.CHAT
        else:
            state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        if is_chat_mode:
            # === CHAT MODE ===
            if iteration > 0:
                # Wait for a new user message (subsequent turns)
                try:
                    user_msg = await asyncio.wait_for(
                        self.agent.message_queue.get(),
                        timeout=300.0,  # 5 minutes — adjust as needed
                    )
                except asyncio.TimeoutError:
                    yield ("error", "Timeout waiting for user message in chat mode")
                    return
            else:
                # FIRST iteration: Do NOT wait. Use message if already in queue, else start empty
                try:
                    user_msg = self.agent.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    user_msg = ""  # or "Hello" or any gentle starter

            # Extract query safely
            if isinstance(user_msg, dict):
                query = user_msg.get("content", "") or str(user_msg)
            else:
                query = str(user_msg).strip()

            if not query:
                query = "Start the conversation naturally."  # fallback for first turn

            logger.debug(
                f"Agent '{self.agent.name}' [iter {iteration}] processing: {query[:100]}..."
            )

            # Build clean chat context
            messages = await self.agent.context_manager.provide_context(
                query=query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=8000,
                system_prompt=self._build_chat_system_prompt(),
                include_logs=False,
                chat=True,
            )

        else:
            # Normal EXECUTE mode (unchanged)
            messages = await self.agent.context_manager.provide_context(
                query=state.current_task or "Continue",
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_objective_reminder(),
                include_logs=True,
            )

        # === The rest of your function stays exactly the same ===
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
        tool_calls = []

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
                    if kind in ("tool_complete", "needs_confirmation"):
                        tool_calls.append(payload)
                    yield (kind, payload)
                elif kind == "finish":
                    break
                elif kind == "error":
                    yield ("error", payload)
                    return
                elif kind == "cancelled":
                    yield ("cancelled", payload)
                    return

        except asyncio.CancelledError:
            yield ("cancelled", "Task cancelled")
            return

        yield (
            "llm_response",
            {
                "full_reply": text_buffer.strip(),
                "tool_calls": tool_calls,
                "is_complete": self.FINAL_ANSWER_MARKER in text_buffer,
            },
        )

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
