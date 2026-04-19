import json
from typing import AsyncIterator, Any, Tuple

from neuralcore.agents.state import AgentState

from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder

logger = Logger.get_logger()


async def is_multi_step_task(agent, query: str) -> bool:
    """Improved generic detection – now explicitly recognizes chained file operations."""
    prompt = PromptBuilder.is_multi_step_task(query)

    try:
        result = await agent.client.chat(prompt, temperature=0.0, max_tokens=10)
        result_upper = result.strip().upper()
        is_complex = "COMPLEX" in result_upper
        logger.info(
            f"[MULTI-STEP] LLM decided → {'COMPLEX' if is_complex else 'SIMPLE'} | Query: {query[:100]}..."
        )
        return is_complex
    except Exception:
        return False


async def ensure_subtasks_planned(
    agent,
    state: AgentState,
) -> AsyncIterator[Tuple[str, Any]]:
    """Generic, LLM-driven structured planning.
    Robust expected_outcome population with FULL (non-truncated) debug logging."""
    logger.info("[PLANNING START] _ensure_subtasks_planned called")
    yield ("phase_changed", {"phase": "planning"})

    planning_prompt = PromptBuilder.task_decomposition(state.task)

    plan_text = ""

    try:
        plan_text = await agent.client.chat(
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
                int(d) for d in deps if isinstance(d, (int, str)) and str(d).isdigit()
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
        state.planned_tasks = [state.task]
        state.task_expected_outcomes = ["Task completed successfully"]
        yield ("planning_fallback", {"reason": "JSON parse error"})
    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        state.planned_tasks = [state.task]
        state.task_expected_outcomes = ["Task completed successfully"]
        yield ("planning_fallback", {"reason": str(e)})


async def classify_intent(agent, query: str) -> str:
    """Enhanced intent classification using rich context from the agent's context manager.

    Falls back gracefully and stays fast (low token usage).
    """
    if not query or not query.strip():
        return "CASUAL"

    try:
        context_messages = await agent.context_manager.provide_context(
            query=query,
            max_input_tokens=8000,
            reserved_for_output=512,
            system_prompt="You are an expert at classifying user intent precisely and quickly.",
            lightweight_agentic=True,
            state=agent.state if hasattr(agent, "state") else None,
            include_logs=True,
            chat=True,
        )

        # Append the actual classification instruction as the final user message
        classification_prompt = PromptBuilder.classify_intent(query)  #

        if context_messages and context_messages[-1]["role"] == "user":
            context_messages[-1]["content"] = classification_prompt
        else:
            context_messages.append({"role": "user", "content": classification_prompt})

        result = await agent.client.chat(
            context_messages,  # full list of messages (not just a string)
            temperature=0.0,
            max_tokens=20,
        )

        cleaned = result.strip().upper()
        return "CASUAL" if "CASUAL" in cleaned else "TASK"

    except Exception as e:
        logger.warning(f"Enhanced classify_intent failed, falling back: {e}")
        # Original lightweight fallback
        return "CASUAL" if len(query.split()) < 25 else "TASK"


async def goal_driven_task_loop(
    agent, state: AgentState, target_loop: str
) -> AsyncIterator[Tuple[str, Any]]:
    """FINAL STATE-DRIVEN goal loop – uses generic success indicator from Action + expected outcomes.
    Requires real completion signals before advancing or finishing. Keeps NeuralCore abstract."""

    is_multi_step = len(state.planned_tasks) > 1
    marker = PromptBuilder.FINAL_ANSWER_MARKER
    state.increment_loop()

    text_buffer = ""
    tools_called_this_turn = False

    # ====================== STATE-AWARE PROMPT ======================
    if is_multi_step and 0 <= state.current_task_index < len(state.planned_tasks):
        task_desc = state.planned_tasks[state.current_task_index]

        used_tools = [r.get("name", "unknown") for r in state.tool_results]
        used_tools_str = ", ".join(set(used_tools)) if used_tools else "none"

        completed = []
        for i in range(state.current_task_index):
            if i < len(state.tool_results):
                tool_name = state.tool_results[i].get("name", "unknown")
                preview = str(state.tool_results[i].get("result", ""))[:400]
                completed.append(f"Step {i} ({tool_name}) already done: {preview}...")

        completed_context = (
            "\n".join(completed) if completed else "No steps completed yet."
        )

        remaining = []
        for i in range(state.current_task_index + 1, len(state.planned_tasks)):
            remaining.append(f"Step {i}: {state.planned_tasks[i]}")

        remaining_context = "\n".join(remaining) if remaining else "No more steps."

        current_query = PromptBuilder.sub_task_execution(
            original_query=state.task,
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
            state.task
            if state.loop_count == 1
            else f"USER ORIGINAL REQUEST: {state.task}\n\nPrevious results are in state.tool_results. Continue."
        )

    yield ("phase_changed", {"phase": "searching_tools"})

    prev_tool_result_count = len(state.tool_results)

    if not current_query:
        current_query = await agent.wait_for_incoming_message(
            role="user", return_content_only=True
        )

    messages = await agent.context_manager.provide_context(
        query=current_query,
        max_input_tokens=agent.max_tokens,
        reserved_for_output=12000,
        system_prompt=PromptBuilder.agent_objective_reminder(agent.state)
        + f"\n\nWhen you finish the current sub-task, you MUST output exactly: {marker}",
        include_logs=True,
        chat=False,
        lightweight_agentic=True,
        state=state,
    )

    queue = await agent.client.stream_with_tools(
        manager=agent.manager,
        messages=messages,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        tool_choice="auto",
        auto_stop_on_complete_tool=True,
    )

    try:
        async for kind, payload in agent.client._drain_queue(queue):
            if kind == "content":
                content = str(payload or "")
                text_buffer += content
                yield ("content_delta", content)

            elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                tools_called_this_turn = True
                if isinstance(payload, dict):
                    tool_name = (
                        payload.get("tool_name") or payload.get("name") or "unknown"
                    )
                    if "FindTool" in tool_name:
                        yield ("phase_changed", {"phase": "handling_findtool"})
                        state.record_findtool_call()
                        state.request_loop_restart(
                            reason="FindTool called",
                            target_loop=target_loop,
                        )
                        yield ("phase_changed", {"phase": "restarting_loop"})
                        return  # ← FindTool: early exit + restart

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
    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))
    strong_completion = has_marker or tool_reported_success

    if has_marker or (new_tool_result and strong_completion):
        if (
            not is_multi_step
            or state.current_task_index >= len(state.planned_tasks) - 1
        ):
            msg = "Marker or strong completion detected + all sub-tasks done"
            state.mark_goal_achieved(msg)
            state.request_loop_stop(reason=msg, target_loop=target_loop)
            # fall through to synthesis below
        else:
            reason = (
                f"[MULTI-STEP] Sub-task {state.current_task_index} complete → advancing"
            )
            logger.info(reason)
            state.current_task_index += 1
            state.empty_loops = 0
            state.last_tool_success = None
            state.request_loop_restart(reason=reason, target_loop=target_loop)
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    # ====================== SAFE ANTI-REPEAT ======================
    if tools_called_this_turn and is_multi_step and not strong_completion:
        logger.warning(
            f"[ANTI-REPEAT FORCE] Tool called on step {state.current_task_index} but no strong completion. "
            f"Staying on current step."
        )
        state.increment_empty_loop()
        state.request_loop_restart(
            reason="Tool called but no strong completion.",
            target_loop=target_loop,
        )
        yield ("phase_changed", {"phase": "restarting_loop"})
        return

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
            reason = "Max action restarts reached"
            state.mark_goal_achieved(reason)
            state.request_loop_stop(reason=reason, target_loop=target_loop)
        else:
            logger.info(
                f"[Action Restart #{state.action_restarts}] Detected '{detected_keyword}'"
            )
            state.request_loop_restart(
                reason="Action restart requested",
                target_loop=target_loop,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    if not tools_called_this_turn and not has_marker and not action_restart_triggered:
        state.increment_empty_loop()
        if state.empty_loops >= 5:
            reason = "Forced completion after empty loops"
            state.mark_goal_achieved(reason)
            state.request_loop_stop(reason=reason, target_loop=target_loop)
        else:
            state.request_loop_restart(
                reason="Empty loop continuation",
                target_loop=target_loop,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return
    else:
        state.empty_loops = 0

    # ====================== NORMAL TOOL EXECUTION (restart only if NOT complete) ======================
    if tools_called_this_turn:
        if state.goal_reached:
            # already handled above – fall through to synthesis
            pass
        else:
            yield ("phase_changed", {"phase": "executing_tools"})
            state.request_loop_restart(
                reason="Tool executed",
                target_loop=target_loop,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    # ====================== FINAL SYNTHESIS – ONLY ON TRUE COMPLETION ======================
    if state.goal_reached:
        yield ("phase_changed", {"phase": "generating_final_answer"})

        synthesis_query = PromptBuilder.final_synthesis(state.task)
        if state.tool_results:
            synthesis_query += "\n\nTool results summary:\n" + "\n".join(
                f"• {r.get('name', 'unknown')}: {str(r.get('result', ''))[:500]}..."
                for r in state.tool_results[-3:]
            )

        final_messages = await agent.context_manager.provide_context(
            query=synthesis_query,
            max_input_tokens=agent.max_tokens,
            reserved_for_output=12000,
            system_prompt="FINAL ANSWER MODE\nProvide a clear, complete summary of what was accomplished.",
            include_logs=True,
            chat=False,
            state=state,
        )

        # ====================== STREAMED FINAL SYNTHESIS ======================
        text_buffer = ""

        final_reply = await agent.client.chat(
            final_messages, temperature=0.0, top_p=0.1
        )

        await agent.add_message("assistant", final_reply)

        yield (
            "llm_response",
            {
                "full_reply": final_reply,
                "tool_calls": [],
                "is_complete": True,
            },
        )

        logger.info("Multi-step task completed successfully → full reset")
        state.reset_for_new_task()

    else:
        # Fallback safety net
        logger.warning("Loop ended without explicit goal or restart – forcing restart")
        state.request_loop_restart(reason="Fallback restart", target_loop=target_loop)
        yield ("phase_changed", {"phase": "restarting_loop"})

    # ====================== CONDITIONAL RESET ======================
    if not state.goal_reached:
        state.status = "idle"
        state.is_complete = True
