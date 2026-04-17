import json

from typing import AsyncIterator, Any, Tuple

from neuralcore.agents.state import AgentState

from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder

logger = Logger.get_logger()


class AgentExecutors:
    """Handles LLM streaming, tool execution, chat loops, and sub-agent execution.
    Refactored: shared goal-driven task loop for both chat and agentic (sub-agent) modes."""

    def __init__(self, agent, phase_enum):
        self.agent = agent
        self.Phase = phase_enum

    # ====================== FAST CASUAL DETECTOR ======================

    async def _is_multi_step_task(self, query: str) -> bool:
        """Improved generic detection – now explicitly recognizes chained file operations."""
        prompt = PromptBuilder.is_multi_step_task(query)

        try:
            result = await self.agent.client.chat(
                prompt, temperature=0.0, max_tokens=10
            )
            result_upper = result.strip().upper()
            is_complex = "COMPLEX" in result_upper
            logger.info(
                f"[MULTI-STEP] LLM decided → {'COMPLEX' if is_complex else 'SIMPLE'} | Query: {query[:100]}..."
            )
            return is_complex
        except Exception:
            return False

    async def _ensure_subtasks_planned(
        self, state: AgentState, original_query: str
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Generic, LLM-driven structured planning.
        Robust expected_outcome population with FULL (non-truncated) debug logging."""
        logger.info("[PLANNING START] _ensure_subtasks_planned called")
        yield ("phase_changed", {"phase": "planning"})

        planning_prompt = PromptBuilder.task_decomposition(original_query)

        plan_text = ""

        try:
            plan_text = await self.agent.client.chat(
                planning_prompt, temperature=0.0, max_tokens=1500
            )

            logger.info(f"[PLANNING RAW] LLM returned {len(plan_text)} chars")
            logger.debug(f"[PLANNING RAW JSON]\n{plan_text}")

            plan = json.loads(plan_text.strip())

            state.planned_tasks.clear()
            state.task_expected_outcomes.clear()
            state.task_dependencies.clear()
            state.task_tool_assignments.clear()

            steps = plan.get("steps", [])
            logger.info(f"[PLANNING] Parsed {len(steps)} steps from JSON")

            for i, step in enumerate(steps):
                description = step.get("description", f"Step {i + 1}").strip()

                expected = step.get("expected_outcome", "")
                if not expected or not str(expected).strip():
                    expected = f"Step {i + 1} completed successfully (file located/analyzed/tool added)"

                state.planned_tasks.append(description)
                state.task_expected_outcomes.append(str(expected).strip())

                deps = step.get("dependencies", [])
                state.task_dependencies[i] = [
                    int(d)
                    for d in deps
                    if isinstance(d, (int, str)) and str(d).isdigit()
                ]

                if step.get("suggested_tool_category"):
                    state.task_tool_assignments[i] = [step["suggested_tool_category"]]

                logger.debug(
                    f"  → Appended Step {i}: '{description}' | expected='{expected}'"
                )

            # Safety alignment
            while len(state.task_expected_outcomes) < len(state.planned_tasks):
                state.task_expected_outcomes.append("Task step completed successfully")

            warnings = state.validate_state_integrity()
            if warnings:
                logger.warning(f"[PLANNING] State integrity warnings: {warnings}")
            else:
                logger.info(
                    "[PLANNING] State integrity check PASSED — expected_outcomes populated correctly"
                )

            logger.info(
                f"[PLANNING] Final count: {len(state.planned_tasks)} tasks, {len(state.task_expected_outcomes)} expected outcomes"
            )

            # FULL non-castrated debug (no [:80] truncation)
            for i, t in enumerate(state.planned_tasks):
                expected = state.task_expected_outcomes[i]
                deps = state.task_dependencies.get(i, [])
                logger.debug(f"  Step {i}: {t} | expected='{expected}' | deps={deps}")

            yield ("planning_complete", {"planned_tasks": state.planned_tasks})
            logger.info("[PLANNING END] Planning completed successfully")

        except json.JSONDecodeError as e:
            logger.error(f"Planning JSON parse failed: {e}\nRaw: {plan_text[:400]}...")
            state.planned_tasks = [original_query]
            state.task_expected_outcomes = ["Task completed successfully"]
            yield ("planning_fallback", {"reason": "JSON parse error"})
        except Exception as e:
            logger.error(f"Planning failed: {e}", exc_info=True)
            state.planned_tasks = [original_query]
            state.task_expected_outcomes = ["Task completed successfully"]
            yield ("planning_fallback", {"reason": str(e)})

    async def _goal_driven_task_loop(
        self,
        state: AgentState,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """FINAL STATE-DRIVEN goal loop – uses generic success indicator from Action + expected outcomes.
        Requires real completion signals before advancing or finishing. Keeps NeuralCore abstract."""

        original_query = next(
            (
                m.get("content", "").strip()
                for m in reversed(state.messages or [])
                if m.get("role") == "user" and m.get("content")
            ),
            state.current_task or "[USER REQUEST]",
        )

        # ====================== TASK PATH – FULL CLEAN RESET ======================
        logger.info("→ [MODE SWITCH] Casual → TASK → full state reset")
        state.reset_for_new_task(new_task=original_query, new_goal=original_query)
        state.planned_tasks = []  # force re-planning on first loop

        logger.info(f"[STATE-DRIVEN TASK LOOP] Starting for: {original_query[:120]}...")

        if state.goal_reached or state.loop_count > 0:
            logger.info("[DEFENSIVE] State was dirty on entry → resetting")
            state.reset_for_new_task(new_task=original_query, new_goal=original_query)
            state.planned_tasks = []

        yield ("phase_changed", {"phase": "thinking"})

        # ====================== ONE-TIME PLANNING ======================
        if not state.planned_tasks:
            is_multi_step = await self._is_multi_step_task(original_query)
            if is_multi_step:
                logger.info("[MULTI-STEP] Detected → structured planning")
                yield ("phase_changed", {"phase": "planning"})
                async for ev, pl in self._ensure_subtasks_planned(
                    state, original_query
                ):
                    yield ev, pl
            else:
                state.planned_tasks = [original_query]
                state.task_expected_outcomes = ["Task completed successfully"]

        is_multi_step = len(state.planned_tasks) > 1
        marker = "[FINAL_ANSWER_COMPLETE]"

        self.agent.manager.unload_all()
        max_loops = 25

        while not state.goal_reached and state.loop_count < max_loops:
            state.increment_loop()

            text_buffer = ""
            tools_called_this_turn = False

            # ====================== STATE-AWARE PROMPT ======================
            if is_multi_step and 0 <= state.current_task_index < len(
                state.planned_tasks
            ):
                task_desc = state.planned_tasks[state.current_task_index]

                used_tools = [r.get("name", "unknown") for r in state.tool_results]
                used_tools_str = ", ".join(set(used_tools)) if used_tools else "none"

                completed = []
                for i in range(state.current_task_index):
                    if i < len(state.tool_results):
                        tool_name = state.tool_results[i].get("name", "unknown")
                        preview = str(state.tool_results[i].get("result", ""))[:400]
                        completed.append(
                            f"Step {i} ({tool_name}) already done: {preview}..."
                        )

                completed_context = (
                    "\n".join(completed) if completed else "No steps completed yet."
                )

                remaining = []
                for i in range(state.current_task_index + 1, len(state.planned_tasks)):
                    remaining.append(f"Step {i}: {state.planned_tasks[i]}")

                remaining_context = (
                    "\n".join(remaining) if remaining else "No more steps."
                )

                current_query = PromptBuilder.sub_task_execution(
                    original_query=original_query,
                    task_desc=task_desc,
                    current_index=state.current_task_index,
                    total_tasks=len(state.planned_tasks),
                    completed_context=completed_context,
                    used_tools_str=used_tools_str,
                    remaining_context=remaining_context,
                    marker=marker,
                    loop_count=state.loop_count,
                )
            else:
                current_query = (
                    original_query
                    if state.loop_count == 1
                    else f"USER ORIGINAL REQUEST: {original_query}\n\nPrevious results are in state.tool_results. Continue."
                )

            yield ("phase_changed", {"phase": "searching_tools"})

            prev_tool_result_count = len(state.tool_results)

            messages = await self.agent.context_manager.provide_context(
                query=current_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=PromptBuilder.agent_objective_reminder(self.agent.state)
                + f"\n\nWhen you finish the current sub-task, you MUST output exactly: {marker}",
                include_logs=True,
                chat=False,
                lightweight_agentic=True,
                state=state,
            )

            queue = await self.agent.client.stream_with_tools(
                manager=self.agent.manager,
                messages=messages,
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                auto_stop_on_complete_tool=True,
            )

            try:
                async for kind, payload in self.agent.client._drain_queue(queue):
                    if kind == "content":
                        content = str(payload or "")
                        text_buffer += content
                        yield ("content_delta", content)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        tools_called_this_turn = True
                        if isinstance(payload, dict):
                            tool_name = (
                                payload.get("tool_name")
                                or payload.get("name")
                                or "unknown"
                            )
                            if "FindTool" in tool_name:
                                yield ("phase_changed", {"phase": "handling_findtool"})
                                state.loop_count = 0
                                state.empty_loops = 0
                                state.action_restarts = 0
                                state.record_findtool_call()
                                break

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield kind, payload
                        return

            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                yield "error", str(e)
                return

            final_reply = text_buffer.strip()
            has_marker = marker in final_reply
            if has_marker:
                final_reply = final_reply.replace(marker, "").strip()

            # ====================== TOOL RESULT + GENERIC SUCCESS CHECK ======================
            new_tool_result = len(state.tool_results) > prev_tool_result_count

            # Generic success indicator set by Action.__call__ (completely abstract)
            last_success = getattr(state, "last_tool_success", None)
            tool_reported_success = bool(last_success and last_success.get("success"))

            strong_completion = has_marker or tool_reported_success

            if has_marker or (new_tool_result and strong_completion):
                if (
                    not is_multi_step
                    or state.current_task_index >= len(state.planned_tasks) - 1
                ):
                    state.mark_goal_achieved(
                        "Marker or strong completion detected + all sub-tasks done"
                    )
                    break
                else:
                    logger.info(
                        f"[MULTI-STEP] Sub-task {state.current_task_index} complete → advancing"
                    )
                    state.current_task_index += 1
                    state.empty_loops = 0
                    state.last_tool_success = None  # clear after advancement
                    continue

            # ====================== SAFE ANTI-REPEAT ======================
            if tools_called_this_turn and is_multi_step and not strong_completion:
                logger.warning(
                    f"[ANTI-REPEAT FORCE] Tool called on step {state.current_task_index} but no strong completion. "
                    f"Staying on current step."
                )
                state.increment_empty_loop()
                continue

            # ====================== ACTION RESTART / EMPTY LOOP ======================
            action_restart_triggered = False
            action_continuation = None
            detected_keyword = None

            if not has_marker:
                lines = final_reply.splitlines()
                last_100_lines = "\n".join(lines[-100:])
                last_100_lower = last_100_lines.lower()

                action_keywords = ["next action:", "next action", "action:", "action"]
                for kw in action_keywords:
                    pos = last_100_lower.rfind(kw)
                    if pos != -1:
                        after_pos = last_100_lines.rfind(kw) + len(kw)
                        candidate = last_100_lines[after_pos:].strip()
                        if candidate and len(candidate) > 15:
                            action_continuation = candidate
                            action_restart_triggered = True
                            detected_keyword = kw
                            break

            if action_restart_triggered and action_continuation:
                state.increment_action_restart()
                if state.action_restarts > 3:
                    state.mark_goal_achieved("Max action restarts reached")
                    break
                else:
                    logger.info(
                        f"[Action Restart #{state.action_restarts}] Detected '{detected_keyword}'"
                    )
                    continue

            if (
                not tools_called_this_turn
                and not has_marker
                and not action_restart_triggered
            ):
                state.increment_empty_loop()
                if state.empty_loops >= 5:
                    state.mark_goal_achieved("Forced completion after empty loops")
                    break
            else:
                state.empty_loops = 0

            if tools_called_this_turn:
                yield ("phase_changed", {"phase": "executing_tools"})
                continue

        # ====================== FINAL SYNTHESIS ======================
        yield ("phase_changed", {"phase": "generating_final_answer"})

        # IMPROVED: rich, explicit synthesis prompt that includes full task context + tool results
        synthesis_query = PromptBuilder.final_synthesis(original_query)
        if state.tool_results:
            synthesis_query += "\n\nTool results summary:\n" + "\n".join(
                f"• {r.get('name', 'unknown')}: {str(r.get('result', ''))[:500]}..."
                for r in state.tool_results[-3:]  # last 3 results
            )

        final_messages = await self.agent.context_manager.provide_context(
            query=synthesis_query,
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=8000,
            system_prompt=PromptBuilder.agent_objective_reminder(self.agent.state)
            + "\n\nFINAL ANSWER MODE\nProvide a clear, complete summary of what was accomplished.",
            include_logs=True,
            chat=False,
            state=state,
        )

        final_reply = await self.agent.client.chat(
            final_messages, temperature=0.0, top_p=0.1
        )

        await self.agent.context_manager.add_message("assistant", final_reply)

        yield (
            "llm_response",
            {
                "full_reply": final_reply,
                "tool_calls": [],
                "is_complete": state.goal_reached,
            },
        )

        # ====================== CONDITIONAL RESET ======================
        if state.goal_reached:
            logger.info("Multi-step task completed successfully → full reset")
            state.reset_for_new_task()
        else:
            logger.warning(
                f"Loop ended without explicit goal (loop_count={state.loop_count}) "
                f"→ light reset (preserving planned_tasks for debugging/continuation)"
            )
            state.status = "idle"
            state.is_complete = True
