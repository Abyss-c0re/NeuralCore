import json
from typing import AsyncIterator, Any, Tuple, Optional

from neuralcore.agents.state import AgentState
from neuralcore.agents.task import Task

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
    """Generic, LLM-driven structured planning using proper Task objects."""
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

        # Clear previous state
        state.planned_tasks.clear()
        state.task_expected_outcomes.clear()
        state.task_dependencies.clear()
        state.task_tool_assignments.clear()
        state.tasks.clear()
        state.completed_task_ids.clear()

        steps = plan.get("steps", [])
        logger.info(f"[PLANNING] Parsed {len(steps)} steps from JSON")

        task_map: dict[int, Task] = {}

        for i, step in enumerate(steps):
            description = step.get("description", f"Step {i + 1}").strip()

            # Ensure expected_outcome is always a non-empty string (fix for type error)
            expected = step.get("expected_outcome", "")
            if not expected or not str(expected).strip():
                expected = f"Step {i + 1} completed successfully (file located/analyzed/tool added)"

            expected_str: str = str(expected).strip()

            task = Task(
                description=description,
                expected_outcome=expected_str,  # now guaranteed str
                metadata={
                    "original_index": i,
                    "suggested_tool_category": step.get("suggested_tool_category"),
                },
            )

            state.tasks.append(task)
            state.planned_tasks.append(description)
            state.task_expected_outcomes.append(expected_str)  # safe str

            # Dependencies (index-based for backward compatibility)
            deps = step.get("dependencies", [])
            state.task_dependencies[i] = [
                int(d) for d in deps if isinstance(d, (int, str)) and str(d).isdigit()
            ]

            if step.get("suggested_tool_category"):
                state.task_tool_assignments[i] = [step["suggested_tool_category"]]

            task_map[i] = task

            logger.debug(
                f"  → Created Task {i}: '{description[:100]}...' | expected='{expected_str}'"
            )

        # Resolve dependencies to real task_ids
        for i, task in enumerate(state.tasks):
            dep_indices = state.task_dependencies.get(i, [])
            dep_ids = [task_map[d].task_id for d in dep_indices if d in task_map]
            task.dependencies = dep_ids
            task._dependency_set = set(dep_ids)

        # Build root task for hierarchy
        if state.tasks:
            state.root_task = Task(description=state.task or "Root user task")
            for t in state.tasks:
                state.root_task.add_subtask(t)

        warnings = state.validate_state_integrity()
        if warnings:
            logger.warning(f"[PLANNING] State integrity warnings: {warnings}")
        else:
            logger.info("[PLANNING] State integrity check PASSED")

        logger.info(
            f"[PLANNING] Created {len(state.tasks)} Task objects "
            f"({len(state.planned_tasks)} backward-compatible entries)"
        )

        # Full debug logging
        for i, task in enumerate(state.tasks):
            logger.debug(
                f"  Task {i}: {task.description[:100]}... | "
                f"expected='{task.expected_outcome}' | deps={task.dependencies}"
            )

        yield ("planning_complete", {"planned_tasks": state.planned_tasks})
        logger.info("[PLANNING END] Planning completed successfully")

    except json.JSONDecodeError as e:
        logger.error(f"Planning JSON parse failed: {e}\nRaw: {plan_text[:400]}...")
        fallback_task = Task(
            description=state.task, expected_outcome="Task completed successfully"
        )
        state.tasks = [fallback_task]
        state.root_task = Task(description=state.task)
        state.root_task.add_subtask(fallback_task)
        state.planned_tasks = [state.task]
        state.task_expected_outcomes = ["Task completed successfully"]
        yield ("planning_fallback", {"reason": "JSON parse error"})

    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        fallback_task = Task(
            description=state.task, expected_outcome="Task completed successfully"
        )
        state.tasks = [fallback_task]
        state.root_task = Task(description=state.task)
        state.root_task.add_subtask(fallback_task)
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
    """Robust multi-step loop using Task objects.
    Fixed: infinite restart on single-tool tasks + type safety."""

    is_multi_step = len(state.tasks) > 1
    marker = PromptBuilder.FINAL_ANSWER_MARKER
    state.increment_loop()

    text_buffer = ""
    tools_called_this_turn = False

    # ====================== STATE-AWARE PROMPT ======================
    current_task: Optional[Task] = None
    if 0 <= state.current_task_index < len(state.tasks):
        current_task = state.tasks[state.current_task_index]

    if current_task:
        task_desc = current_task.description

        used_tools = [r.get("name", "unknown") for r in state.tool_results]
        used_tools_str = ", ".join(set(used_tools)) if used_tools else "none"

        completed = []
        for i in range(state.current_task_index):
            if i < len(state.tool_results):
                tool_name = state.tool_results[i].get("name", "unknown")
                preview = str(state.tool_results[i].get("result", ""))[:400]
                completed.append(f"Step {i} ({tool_name}) done: {preview}...")

        completed_context = (
            "\n".join(completed) if completed else "No steps completed yet."
        )

        remaining = [
            f"Step {i}: {state.tasks[i].description}"
            for i in range(state.current_task_index + 1, len(state.tasks))
        ]
        remaining_context = "\n".join(remaining) if remaining else "No more steps."

        expected_outcome: str = (
            current_task.expected_outcome or "Tool executed successfully"
        )

        current_query = PromptBuilder.sub_task_execution(
            original_query=state.task,
            task_desc=task_desc,
            current_index=state.current_task_index,
            total_tasks=len(state.tasks),
            completed_context=completed_context,
            used_tools_str=used_tools_str,
            remaining_context=remaining_context,
            marker=marker,
            loop_count=state.loop_count,
            expected_outcome=expected_outcome,
        )
    else:
        current_query = (
            state.task
            if state.loop_count == 1
            else f"USER ORIGINAL REQUEST: {state.task}\n\nPrevious results are in state.tool_results. Continue."
        )

    yield ("phase_changed", {"phase": "searching_tools"})

    if not current_query:
        current_query = await agent.wait_for_incoming_message(
            role="user", return_content_only=True
        )
    if not current_query:
        state.request_loop_restart(reason="No query detected", target_loop=target_loop)
        return

    messages = await agent.context_manager.provide_context(
        query=current_query,
        max_input_tokens=agent.max_tokens,
        reserved_for_output=12000,
        system_prompt=PromptBuilder.agent_objective_reminder(state)
        + "\n\n"
        + PromptBuilder.sub_task_execution_system_prompt()
        + f"\n\nWhen you finish the current sub-task and expected_outcome is verified, output exactly: {marker}",
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
                        state.record_findtool_call()
                        state.request_loop_restart(
                            reason="FindTool called", target_loop=target_loop
                        )
                        yield ("phase_changed", {"phase": "restarting_loop"})
                        return

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

    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))
    strong_completion = has_marker or tool_reported_success

    # ====================== UNIFIED COMPLETION LOGIC ======================
    if tools_called_this_turn and current_task:
        # Mark current Task as completed on tool success (fixes single-tool infinite loop)
        current_task.complete(
            result=state.tool_results[-1].get("result") if state.tool_results else None
        )
        if current_task.status == "completed":
            state.completed_task_ids.add(current_task.task_id)

        # Single-step task or final step → mark goal achieved and exit loop cleanly
        if not is_multi_step or state.current_task_index >= len(state.tasks) - 1:
            state.mark_goal_achieved("Task completed via successful tool execution")
            return  # Let outer goal_achieved condition break the loop

        # Multi-step continuation
        yield ("phase_changed", {"phase": "executing_tools"})
        state.request_loop_restart(
            reason="Tool executed — advancing to next sub-task",
            target_loop=target_loop,
        )
        yield ("phase_changed", {"phase": "restarting_loop"})
        return

    # ====================== LLM VALIDATION (multi-step only) ======================
    if is_multi_step and strong_completion and current_task and not state.goal_reached:
        logger.info(
            f"[STEP VALIDATION] Starting LLM-based validation for step {state.current_task_index + 1}"
        )

        last_result_str = (
            str(state.tool_results[-1].get("result", ""))
            if state.tool_results
            else "No tool results yet."
        )

        validation_query = PromptBuilder.step_validation_prompt(
            current_task=current_task,
            last_result_str=last_result_str,
            total_tasks=len(state.tasks),
            current_idx=state.current_task_index,
        )

        try:
            validation_messages = await agent.context_manager.provide_context(
                query=validation_query,
                max_input_tokens=agent.max_tokens,
                reserved_for_output=2500,
                system_prompt=PromptBuilder.validation_system_prompt(),
                include_logs=True,
                chat=False,
                lightweight_agentic=False,
                state=state,
            )

            validation_result = await agent.client.chat(
                validation_messages, temperature=0.0, max_tokens=20
            )

            validation_clean = validation_result.strip().upper()
            outcome_met = "YES" in validation_clean

            logger.info(
                f"[STEP VALIDATION] LLM decided: {validation_clean} → "
                f"{'OUTCOME MET' if outcome_met else 'OUTCOME NOT MET'}"
            )

        except Exception as e:
            logger.warning(f"Validation LLM call failed: {e}. Falling back to marker.")
            outcome_met = has_marker

        if not outcome_met and not has_marker:
            logger.warning(
                f"[STEP VALIDATION FAILED] Step {state.current_task_index} not met."
            )
            state.increment_empty_loop()
            state.request_loop_restart(
                reason="LLM validation failed — outcome not met",
                target_loop=target_loop,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

        logger.info(
            f"[STEP VALIDATION PASSED] Sub-task {state.current_task_index + 1} validated"
        )

        state.current_task_index += 1

        if state.current_task_index >= len(state.tasks):
            state.mark_goal_achieved("All planned sub-tasks completed and validated")
        else:
            if hasattr(state, "advance_to_next_ready_task"):
                state.advance_to_next_ready_task()

            state.empty_loops = 0
            state.action_restarts = 0
            state.request_loop_restart(
                reason=f"Sub-task validated → advancing",
                target_loop=target_loop,
                reset_counters=False,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    # ====================== SAFE ANTI-REPEAT ======================
    if tools_called_this_turn and is_multi_step and not strong_completion:
        logger.warning(
            f"[ANTI-REPEAT FORCE] Tool called on step {state.current_task_index} but no strong completion."
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
                reason="Action restart requested", target_loop=target_loop
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
                reason="Empty loop continuation", target_loop=target_loop
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return
    else:
        state.empty_loops = 0

    # ====================== FINAL SYNTHESIS ======================
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
            system_prompt=PromptBuilder.final_synthesis_system_prompt(),
            include_logs=True,
            chat=False,
            state=state,
        )

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

        logger.info("Task completed successfully → full reset")
        state.reset_for_new_task()

    else:
        logger.warning("Loop ended without explicit goal or restart – forcing restart")
        state.request_loop_restart(reason="Fallback restart", target_loop=target_loop)
        yield ("phase_changed", {"phase": "restarting_loop"})

    if not state.goal_reached:
        state.status = "idle"
        state.is_complete = True
