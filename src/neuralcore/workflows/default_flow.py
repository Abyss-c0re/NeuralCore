import re
import asyncio

import json

from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger
from neuralcore.agents.state import AgentState, Phase
from typing import AsyncIterator, Dict, Any


logger = Logger.get_logger()

logger.debug("Global workflow registry created")


class AgentFlow:
    """
    AgentFlow — Pure collection of all _wf_* step handlers
    Uses self.engine.xxx to call the executors that stay in WorkflowEngine.
    """

    def __init__(self, engine):
        self.agent = engine.agent
        self.engine = engine

    @workflow.set(
        "default",
        name="plan_tasks",
        description="Generates tool queries and task plan.",
    )
    async def _wf_plan_tasks(
        self, iteration: int, state: "AgentState"
    ) -> AsyncIterator:
        """
        Faster, fully async + parallel task planning.
        Streams progress while doing registry lookups and LLM calls in parallel.
        """
        if not state.planned_tasks:
            state.phase = Phase.PLAN
            yield ("phase_changed", {"phase": state.phase.value})

            # ── Step 1: Generate queries (async streaming)
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

            # Try JSON parsing directly
            try:
                queries_list = json.loads(raw_queries).get("queries", [])
            except Exception:
                # Fallback: regex non-greedy
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

            # ── Step 2: Parallel registry search for suggested tools
            async def search_registry(query: str):
                # Wrap synchronous registry search in thread
                return await asyncio.to_thread(self.agent.registry.search, query, 3)

            registry_tasks = [search_registry(q) for q in queries_list]
            search_results = await asyncio.gather(*registry_tasks)

            # Flatten and deduplicate
            suggested_tools = list(
                dict.fromkeys(a.name for results in search_results for a, _ in results)
            )

            if suggested_tools:
                self.agent.manager.load_tools(suggested_tools)
                yield ("info", f"Loaded tools: {', '.join(suggested_tools)}")

            # ── Step 3: Generate plan using available tools
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

            # Try JSON parsing directly first
            try:
                plan_json = json.loads(raw_plan)
                state.planned_tasks = plan_json.get("tasks", [])
            except Exception:
                # fallback: regex
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

        # ── Step 4: Yield next task if available
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
        async for ev, pl in self.engine._llm_stream_with_tools(iteration, state):
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

        async for ev, pl in self.engine._execute_tools(
            state.tool_calls, iteration, state
        ):
            yield (ev, pl)

        state.tool_calls = []
        logger.debug(f"Cleared tool_calls after execution at iteration {iteration}")

        for r in self.agent.tool_results:
            sig = f"{r['name']}:{json.dumps(r['args'], sort_keys=True)}"
            self.agent.executed_signatures.add(sig)

        state.phase = Phase.DECISION
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
        """
        Streaming-friendly, fast goal verification.
        Parses JSON safely and yields progress incrementally.
        """

        if not state.is_complete:
            return

        # ── Step 1: Update phase
        state.phase = Phase.DECISION
        yield ("phase_changed", {"phase": state.phase.value})
        yield ("goal_verification_start", {"iteration": iteration})

        # ── Step 2: Prepare context-aware prompt
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

        # ── Step 3: Ask LLM (non-blocking streaming)
        raw_verification = await self.agent.client.chat(
            [{"role": "user", "content": prompt}]
        )

        # ── Step 4: Attempt fast JSON parse
        verification: Dict[str, Any] = {}
        try:
            verification = json.loads(raw_verification.strip())
        except Exception:
            # fallback: non-greedy regex
            import re

            match = re.search(r"\{.*?\}", str(raw_verification), re.DOTALL)
            if match:
                try:
                    verification = json.loads(match.group())
                except Exception:
                    verification = {}
            else:
                verification = {}

        # ── Step 5: Ensure defaults
        verification.setdefault("verified", False)
        verification.setdefault("reason", "parse failed" if not verification else "")
        verification.setdefault("missing_steps", [])

        # ── Step 6: Yield verification result
        yield ("goal_verification_result", verification)

        # ── Step 7: Update state & context based on result
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
            state.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            yield (
                "llm_response",
                {"full_reply": state.full_reply, "tool_calls": [], "is_complete": True},
            )
            yield ("finish", {"reason": "casual_complete"})
            self.engine._log_iteration_state(iteration, state)
            return

        if state.is_complete:
            state.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            async for ev, pl in self.engine._generate_final_summary(state):
                yield (ev, pl)
            yield ("finish", {"reason": "complete"})
            self.engine._log_iteration_state(iteration, state)

    @workflow.set(
        "default",
        name="reflect_if_stuck",
        description="Triggers reflection when no progress is detected.",
    )
    async def _wf_reflect_if_stuck(self, iteration: int, state: "AgentState"):
        """
        Detect lack of progress via snapshot comparison.
        Apply hard limit on reflection count to break potential loops.
        Calls engine._force_reflection when appropriate.
        """
        REFLECT_INTERVAL = 5
        last_reflect_iter = getattr(state, "last_reflection_iteration", 0)

        # ── Skip if too early or too soon
        if iteration <= 3 or (iteration - last_reflect_iter) < REFLECT_INTERVAL:
            return

        # ── Minimal progress snapshot
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

        # ── Increment reflection counter
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

        # ── Record reflection iteration
        state.last_reflection_iteration = iteration

        # ── Invoke workflow engine reflection
        async for event, payload in self.engine._force_reflection(iteration, state):
            yield (event, payload)

            if event != "reflection_decision":
                continue

            decision = payload
            next_step = decision.get("next_step")

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
        last_reflect = getattr(state, "last_reflection_iteration", 0)
        last_replan = getattr(state, "last_replan_iteration", 0)

        # Only trigger once per reflection and after the reflection step has completed
        if (
            last_reflect == 0
            or iteration <= last_reflect + 1
            or last_replan == iteration
        ):
            return

        # Reset planning state
        state.planned_tasks = []
        state.current_task_index = 0
        state.last_replan_iteration = iteration  # dynamic flag (safe on dataclass)

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
            async for ev, pl in self.engine._generate_final_summary(state):
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})
            self.engine._log_iteration_state(iteration, state)
