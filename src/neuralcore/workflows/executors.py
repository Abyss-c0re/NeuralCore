import json
from typing import AsyncIterator, Any, Tuple, List, Dict, Optional

from neuralcore.agents.state import AgentState
from neuralcore.agents.task import Task

from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder

logger = Logger.get_logger()


async def classify_intent(agent, query: str) -> str:
    if not query or not query.strip():
        return "CASUAL"
    try:
        result = await agent.client.chat(
            PromptBuilder.classify_intent(query), temperature=0.0, max_tokens=50
        )
        cleaned = result.strip().upper()
        return "CASUAL" if "CASUAL" in cleaned else "TASK"
    except Exception as e:
        logger.warning(f"classify_intent failed, falling back: {e}")
        return "CASUAL" if len(query.split()) < 25 else "TASK"


async def plan_tasks_unified(
    agent, state: "AgentState"
) -> AsyncIterator[Tuple[str, Any]]:
    """
    UNIFIED PLANNING METHOD (combines task_decomposition + is_multi_step_task logic).

    Always performs structured planning via LLM.
    Then decides SINGLE vs MULTI based on the **size** (len(steps)) of the generated plan:
    - len(steps) <= 1  → treat as singular / simple task
    - len(steps) >= 2  → treat as multi-step with dependencies

    This replaces the previous two-LLM-call approach (is_multi_step_task + separate planning)
    with a single, more efficient LLM call. The plan itself is the source of truth for complexity.
    """
    logger.info("[PLANNING START] Unified plan_tasks_unified called")
    yield ("phase_changed", {"phase": "planning"})

    planning_prompt = PromptBuilder.task_decomposition(state.task)
    plan_text = ""

    try:
        plan_text = await agent.client.chat(
            planning_prompt, temperature=0.0, max_tokens=agent.client.max_tokens
        )
        logger.info(f"[PLANNING RAW] LLM returned {len(plan_text)} chars")

        plan = json.loads(plan_text.strip())
        steps: List[Dict] = plan.get("steps", [])
        if hasattr(agent, "consolidator") and agent.consolidator:
            for step in steps:
                suggested = step.get("suggested_tool", "").strip()
                if suggested:
                    await agent.consolidator.record_llm_suggested_tool(
                        query=step.get("description", state.task), tool_name=suggested
                    )

        logger.info(
            f"[PLANNING] LLM proposed {len(steps)} step(s) for query: {state.task[:80]}..."
        )

        # === Normalize expected_outcome for safety (used in both branches) ===
        for step in steps:
            expected = step.get("expected_outcome", "")
            if not expected or not str(expected).strip():
                step["expected_outcome"] = (
                    "Step completed successfully (file located/analyzed/tool added)"
                )

        if len(steps) <= 1:
            # ── SINGLE-STEP PATH (based on plan size) ──
            if steps:
                step0 = steps[0]
                description = step0.get("description", state.task or "Single-step goal")
                expected_outcome = step0.get(
                    "expected_outcome", "Task completed successfully"
                )
            else:
                description = state.task or "Single-step goal"
                expected_outcome = "Task completed successfully"

            simple_task = Task(
                description=description,
                expected_outcome=expected_outcome,
            )
            state.tasks = [simple_task]
            state.root_task = Task(description=state.task or "Root task")
            state.root_task.add_subtask(simple_task)
            state.planned_tasks = [description]
            state.task_expected_outcomes = [expected_outcome]
            state.current_task_index = 0

            logger.info("[SINGLE-STEP] Plan size=1 → initialized single Task object")
            yield (
                "planning_complete",
                {
                    "planned_tasks": state.planned_tasks,
                    "is_single_step": True,
                    "num_steps": 1,
                },
            )

        else:
            state.build_tasks_from_plan(steps)
            state.current_task_index = 0

            logger.info(
                f"[MULTI-STEP] Plan size={len(steps)} → created {len(state.tasks)} Task objects via build_tasks_from_plan"
            )
            for i, task in enumerate(state.tasks):
                logger.debug(
                    f"  Task {i}: {task.description[:80]}... | expected='{task.expected_outcome}' | deps={task.dependencies}"
                )

            yield (
                "planning_complete",
                {
                    "planned_tasks": state.planned_tasks,
                    "is_single_step": False,
                    "num_steps": len(steps),
                },
            )
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
        yield (
            "planning_fallback",
            {"reason": "JSON parse error", "is_single_step": True},
        )

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
        yield ("planning_fallback", {"reason": str(e), "is_single_step": True})


async def goal_driven_task_loop(
    agent, state: AgentState, target_loop: str
) -> AsyncIterator[Tuple[str, Any]]:
    """Robust multi-step loop with persistent tool management + reliable final synthesis.
    NOTE: This generator is driven externally (e.g. chat_tool_loop). Only AgentState persists
    across iterations. All local variables are reset on every call."""
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

            if last_tool and last_tool != suggested_tool:
                try:
                    agent.manager.unload_tools([last_tool])
                    logger.debug(f"[TOOL MGMT] Unloaded previous tool: {last_tool}")
                except Exception as e:
                    logger.warning(f"[TOOL MGMT] Failed to unload {last_tool}: {e}")

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

        # Rich previous-results context (from persisted state.tool_results)
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

    
        used_tools_str = state.used_tools_str

        completed = []
        for i in range(state.current_task_index):
            if i < len(state.tool_results):
                tool_name = state.tool_results[i].get("name") or f"step-{i}"
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
        reserved_for_output=agent.max_tokens * 0.45,
        include_logs=True,
        chat=False,
        lightweight_agentic=True,
    )

    logger.debug(f"Task Context: {(str(messages))}")

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
                    tool_name = payload.get("tool_name") or payload.get("name")

                    if tool_name:
                        yield ("tool_name", {"name": tool_name})

                    if tool_name and "FindTool" in tool_name:
                        state.record_findtool_call()
                        state.skip_manager_this_turn = True
                        state.request_loop_restart(
                            reason="FindTool called", target_loop=target_loop
                        )
                        yield ("phase_changed", {"phase": "restarting_loop"})
                        return
                    elif tool_name:
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

    if not state.skip_manager_this_turn:
        state.skip_manager_this_turn = False

    final_reply = text_buffer.strip()
    has_marker = marker in final_reply
    if has_marker:
        final_reply = final_reply.replace(marker, "").strip()

    # ====================== UNIFIED COMPLETION LOGIC ======================
    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))
    strong_completion = has_marker or tool_reported_success

    if tools_called_this_turn and current_task:
        result = state.tool_results[-1].get("result") if state.tool_results else None

        # Record actual tool usage for training
        tool_name = state.last_used_tool  # can be None — consolidator handles it
        current_task.used_tool = tool_name

        if hasattr(agent, "consolidator") and agent.consolidator:
            success = bool(result) and current_task.status != "failed"
            await agent.consolidator.record_actual_tool_usage(
                query=current_task.description,
                tool_name=tool_name,
                success=success,
            )

        # === EXPLICIT + CLEAR ===
        current_task.complete(result=result)
        if current_task.status == "completed":
            state.completed_task_ids.add(current_task.task_id)

        # === ADVANCE TO NEXT TASK ===
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
            state.goal_achieved = True

        else:
            state.current_task_index += 1
            if hasattr(state, "advance_to_next_ready_task"):
                state.advance_to_next_ready_task()

            logger.info(f"[ADVANCE] Sub-task {state.current_task_index} ready → next")
            yield ("phase_changed", {"phase": "restarting_loop"})
            state.request_loop_restart(
                reason="Tool executed — advancing to next sub-task",
                target_loop=target_loop,
            )

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
            validation_context = await agent.context_manager.provide_context(
                query=validation_query,
                max_input_tokens=agent.max_tokens * 0.45,
                reserved_for_output=agent.client.max_tokens * 0.25,
                include_logs=True,
                chat=False,
                lightweight_agentic=False,
                return_as_string=True,
            )
            logger.debug(f"Validation Context: {validation_context}")
            validation_result = await agent.client.chat(
                validation_context, temperature=0.0, max_tokens=20
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
            yield ("phase_changed", {"phase": "restarting_loop"})
            state.request_loop_restart(
                reason="LLM validation failed — outcome not met",
                target_loop=target_loop,
            )

        logger.info(
            f"[STEP VALIDATION PASSED] Step {state.current_task_index + 1} validated"
        )
        state.current_task_index += 1

        if state.current_task_index >= len(state.tasks):
            state.mark_goal_achieved("All planned sub-tasks completed and validated")
            state.goal_achieved = True
        else:
            if hasattr(state, "advance_to_next_ready_task"):
                state.advance_to_next_ready_task()
            state.empty_loops = 0
            state.action_restarts = 0
            yield ("phase_changed", {"phase": "restarting_loop"})

            state.request_loop_restart(
                reason="Sub-task validated → advancing",
                target_loop=target_loop,
                reset_counters=False,
            )

    # ====================== SAFE ANTI-REPEAT / ACTION / EMPTY LOOP ======================
    if tools_called_this_turn and is_multi_step and not strong_completion:
        logger.warning(
            f"[ANTI-REPEAT] Tool called on step {state.current_task_index} but no strong completion."
        )
        state.increment_empty_loop()
        yield ("phase_changed", {"phase": "restarting_loop"})
        state.request_loop_restart(
            reason="Tool called but no strong completion.", target_loop=target_loop
        )

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
            yield ("phase_changed", {"phase": "restarting_loop"})
            state.request_loop_stop(
                reason="Max action restarts reached", target_loop=target_loop
            )
        else:
            yield ("phase_changed", {"phase": "restarting_loop"})
            state.request_loop_restart(
                reason="Action restart requested", target_loop=target_loop
            )

    if not tools_called_this_turn and not has_marker and not action_restart_triggered:
        state.increment_empty_loop()
        if state.empty_loops >= 5:
            state.mark_goal_achieved("Forced completion after empty loops")
            state.request_loop_stop(
                reason="Forced completion after empty loops", target_loop=target_loop
            )
        else:
            yield ("phase_changed", {"phase": "restarting_loop"})
            state.request_loop_restart(
                reason="Empty loop continuation", target_loop=target_loop
            )

    else:
        state.empty_loops = 0

    if not state.goal_reached:
        logger.warning("Loop iteration ended without goal_achieved — forcing restart")
        state.request_loop_restart(
            reason="Loop ended without goal", target_loop=target_loop
        )
        yield ("phase_changed", {"phase": "restarting_loop"})