import json
from typing import AsyncIterator, Any, Tuple, Optional

from neuralcore.agents.state import AgentState
from neuralcore.agents.task import Task

from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder

logger = Logger.get_logger()


async def is_multi_step_task(agent, query: str) -> bool:
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
    agent, state: AgentState
) -> AsyncIterator[Tuple[str, Any]]:
    logger.info("[PLANNING START] _ensure_subtasks_planned called")
    yield ("phase_changed", {"phase": "planning"})

    planning_prompt = PromptBuilder.task_decomposition(state.task)
    plan_text = ""

    try:
        plan_text = await agent.client.chat(
            planning_prompt, temperature=0.0, max_tokens=1500
        )
        logger.info(f"[PLANNING RAW] LLM returned {len(plan_text)} chars")

        plan = json.loads(plan_text.strip())
        steps = plan.get("steps", [])

        # Ensure expected_outcome is always a safe non-empty string before building
        for step in steps:
            expected = step.get("expected_outcome", "")
            if not expected or not str(expected).strip():
                step["expected_outcome"] = (
                    "Step completed successfully (file located/analyzed/tool added)"
                )

        # === CENTRALIZED: use AgentState.build_tasks_from_plan for consistency ===
        state.build_tasks_from_plan(steps)
        state.current_task_index = 0

        logger.info(
            f"[PLANNING] Created {len(state.tasks)} Task objects via build_tasks_from_plan"
        )

        for i, task in enumerate(state.tasks):
            logger.debug(
                f"  Task {i}: {task.description[:80]}... | expected='{task.expected_outcome}' | deps={task.dependencies}"
            )

        yield ("planning_complete", {"planned_tasks": state.planned_tasks})
        logger.info("[PLANNING END] Planning completed successfully")

    except json.JSONDecodeError as e:
        logger.error(f"Planning JSON parse failed: {e}")
        fallback = Task(
            description=state.task, expected_outcome="Task completed successfully"
        )
        state.tasks = [fallback]
        state.root_task = Task(description=state.task)
        state.root_task.add_subtask(fallback)
        state.planned_tasks = [state.task]
        state.task_expected_outcomes = ["Task completed successfully"]
        state.current_task_index = 0
        yield ("planning_fallback", {"reason": "JSON parse error"})

    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        fallback = Task(
            description=state.task, expected_outcome="Task completed successfully"
        )
        state.tasks = [fallback]
        state.root_task = Task(description=state.task)
        state.root_task.add_subtask(fallback)
        state.planned_tasks = [state.task]
        state.task_expected_outcomes = ["Task completed successfully"]
        state.current_task_index = 0
        yield ("planning_fallback", {"reason": str(e)})


async def classify_intent(agent, query: str) -> str:
    if not query or not query.strip():
        return "CASUAL"
    try:
        context_messages = await agent.context_manager.provide_context(
            query=query,
            max_input_tokens=8000,
            reserved_for_output=512,
            system_prompt="You are an expert at classifying user intent precisely and quickly.",
            lightweight_agentic=True,
            state=getattr(agent, "state", None),
            include_logs=True,
            chat=True,
        )
        classification_prompt = PromptBuilder.classify_intent(query)
        if context_messages and context_messages[-1]["role"] == "user":
            context_messages[-1]["content"] = classification_prompt
        else:
            context_messages.append({"role": "user", "content": classification_prompt})

        result = await agent.client.chat(
            context_messages, temperature=0.0, max_tokens=20
        )
        cleaned = result.strip().upper()
        return "CASUAL" if "CASUAL" in cleaned else "TASK"
    except Exception as e:
        logger.warning(f"classify_intent failed, falling back: {e}")
        return "CASUAL" if len(query.split()) < 25 else "TASK"


async def goal_driven_task_loop(
    agent, state: AgentState, target_loop: str
) -> AsyncIterator[Tuple[str, Any]]:
    """Robust multi-step loop with persistent tool management via AgentState."""
    is_multi_step = len(state.tasks) > 1
    marker = PromptBuilder.FINAL_ANSWER_MARKER
    state.increment_loop()

    text_buffer = ""
    tools_called_this_turn = False

    current_task: Optional[Task] = state.get_current_task()

    if current_task:
        if current_task.status == "pending":
            current_task.start(agent)

        task_desc = current_task.description
        suggested_tool = current_task.suggested_tool or ""

        # === PERSISTENT TOOL MANAGEMENT ===
        if not state.skip_manager_this_turn and suggested_tool:
            last_tool = state.last_used_tool

            # Unload only the previous tool if different
            if last_tool and last_tool != suggested_tool:
                try:
                    agent.manager.unload_tools([last_tool])
                    logger.debug(f"[TOOL MGMT] Unloaded previous tool: {last_tool}")
                except Exception as e:
                    logger.warning(f"[TOOL MGMT] Failed to unload {last_tool}: {e}")

            # Load the recommended tool for this step
            if not agent.manager.is_loaded(suggested_tool):
                try:
                    agent.manager.load_tools([suggested_tool])
                    logger.info(
                        f"[TOOL MGMT] Loaded suggested tool for step: {suggested_tool}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[TOOL MGMT] Could not pre-load {suggested_tool}: {e}"
                    )

        # Rich previous-results context (unchanged)
        previous_results_context = ""
        if state.tool_results:
            file_contents = []
            for r in state.tool_results[-5:]:
                name = r.get("name", "")
                result = r.get("result", "")
                if any(
                    k in name.lower() for k in ["read_file", "read_pdf", "pdf", "read "]
                ):
                    content = str(result)[:12000]
                    file_contents.append(
                        f"### Previously read file ({name}):\n{content}\n---"
                    )
            if file_contents:
                previous_results_context = "\n\n".join(file_contents)

        used_tools = [r.get("name", "unknown") for r in state.tool_results]
        used_tools_str = ", ".join(set(used_tools)) if used_tools else "none"

        # Enhanced preview with notice (unchanged)
        completed = []
        for i in range(state.current_task_index):
            if i < len(state.tool_results):
                tool_name = state.tool_results[i].get("name", "unknown")
                raw = str(state.tool_results[i].get("result", ""))
                if len(raw) > 400:
                    preview = (
                        raw[:350]
                        + " … [FULL CONTENT RECORDED IN CONTEXT — use previous_results_context / tool_results. DO NOT re-read.]"
                    )
                else:
                    preview = raw
                completed.append(f"Step {i} ({tool_name}) done: {preview}")

        completed_context = (
            "\n".join(completed) if completed else "No steps completed yet."
        )
        remaining = [
            f"Step {i}: {state.tasks[i].description}"
            for i in range(state.current_task_index + 1, len(state.tasks))
        ]
        remaining_context = "\n".join(remaining) if remaining else "No more steps."

        expected_outcome = current_task.expected_outcome or "Tool executed successfully"

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
            previous_results_context=previous_results_context,
        )
    else:
        current_query = (
            state.task
            if state.loop_count == 1
            else f"USER ORIGINAL REQUEST: {state.task}\n\nPrevious results are in state.tool_results. Continue."
        )

    yield ("phase_changed", {"phase": "thinking"})

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
                text_buffer += str(payload or "")
                yield ("content_delta", str(payload or ""))

            elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                tools_called_this_turn = True
                if isinstance(payload, dict):
                    tool_name = (
                        payload.get("tool_name") or payload.get("name") or "unknown"
                    )

                    yield ("tool_name", {"name": tool_name})

                    if "FindTool" in tool_name:
                        state.record_findtool_call()
                        state.skip_manager_this_turn = True  # ← PERSISTENT
                        state.request_loop_restart(
                            reason="FindTool called", target_loop=target_loop
                        )
                        yield ("phase_changed", {"phase": "restarting_loop"})
                        return
                    else:
                        # Update last used tool for next iteration
                        state.last_used_tool = tool_name

            elif kind == "finish":
                break
            elif kind in ("error", "cancelled"):
                yield kind, payload
                return
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield "error", str(e)
        return

    # Reset skip flag after a successful non-FindTool turn
    if not state.skip_manager_this_turn:
        state.skip_manager_this_turn = False

    final_reply = text_buffer.strip()
    has_marker = marker in final_reply
    if has_marker:
        final_reply = final_reply.replace(marker, "").strip()

    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))
    strong_completion = has_marker or tool_reported_success

    # ====================== UNIFIED COMPLETION LOGIC ======================
    if tools_called_this_turn and current_task:
        result = state.tool_results[-1].get("result") if state.tool_results else None
        state.mark_current_task_complete(result=result)

        if not is_multi_step or state.current_task_index >= len(state.tasks) - 1:
            state.full_reply = f"Task completed successfully. {marker}"
            state.is_complete = True
            state.last_tool_success = {"success": True}
            state.mark_goal_achieved("All sub-tasks completed")

            yield (
                "llm_response",
                {"full_reply": state.full_reply, "tool_calls": [], "is_complete": True},
            )
            yield ("phase_changed", {"phase": "completed"})
            return

        state.current_task_index += 1
        if hasattr(state, "advance_to_next_ready_task"):
            state.advance_to_next_ready_task()

        logger.info(f"[ADVANCE] Sub-task {state.current_task_index} ready → next")
        yield ("phase_changed", {"phase": "executing_tools"})
        state.request_loop_restart(
            reason="Tool executed — advancing to next sub-task", target_loop=target_loop
        )
        yield ("phase_changed", {"phase": "restarting_loop"})
        return

    # ====================== LLM VALIDATION (multi-step only) ======================
    if is_multi_step and strong_completion and current_task and not state.goal_reached:
        logger.info(
            f"[STEP VALIDATION] Starting LLM validation for step {state.current_task_index + 1}"
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
            outcome_met = "YES" in validation_result.strip().upper()
        except Exception as e:
            logger.warning(f"Validation LLM failed: {e}. Falling back to marker.")
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
            f"[STEP VALIDATION PASSED] Step {state.current_task_index + 1} validated"
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
                reason="Sub-task validated → advancing",
                target_loop=target_loop,
                reset_counters=False,
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    # ====================== SAFE ANTI-REPEAT / ACTION / EMPTY LOOP ======================
    if tools_called_this_turn and is_multi_step and not strong_completion:
        logger.warning(
            f"[ANTI-REPEAT] Tool called on step {state.current_task_index} but no strong completion."
        )
        state.increment_empty_loop()
        state.request_loop_restart(
            reason="Tool called but no strong completion.", target_loop=target_loop
        )
        yield ("phase_changed", {"phase": "restarting_loop"})
        return

    action_restart_triggered = False
    action_continuation = None

    if not has_marker:
        lines = final_reply.splitlines()
        last_100_lines = "\n".join(lines[-100:])
        last_100_lower = last_100_lines.lower()
        for kw in ["next action:", "next action", "action:", "action"]:
            pos = last_100_lower.rfind(kw)
            if pos != -1:
                candidate = last_100_lines[pos + len(kw) :].strip()
                if candidate and len(candidate) > 15:
                    action_continuation = candidate
                    action_restart_triggered = True
                    break

    if action_restart_triggered and action_continuation:
        state.increment_action_restart()
        if state.action_restarts > 3:
            state.mark_goal_achieved("Max action restarts reached")
            state.request_loop_stop(
                reason="Max action restarts reached", target_loop=target_loop
            )
        else:
            state.request_loop_restart(
                reason="Action restart requested", target_loop=target_loop
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return

    if not tools_called_this_turn and not has_marker and not action_restart_triggered:
        state.increment_empty_loop()
        if state.empty_loops >= 5:
            state.mark_goal_achieved("Forced completion after empty loops")
            state.request_loop_stop(
                reason="Forced completion after empty loops", target_loop=target_loop
            )
        else:
            state.request_loop_restart(
                reason="Empty loop continuation", target_loop=target_loop
            )
            yield ("phase_changed", {"phase": "restarting_loop"})
            return
    else:
        state.empty_loops = 0

    # ====================== FINAL SYNTHESIS (fallback only) ======================
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
            {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
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
