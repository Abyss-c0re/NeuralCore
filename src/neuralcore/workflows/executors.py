import json

from typing import AsyncIterator, Dict, Any, List, Optional, Tuple


from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet
from neuralcore.utils.logger import Logger
from neuralcore.actions.manager import registry

logger = Logger.get_logger()


class AgentExecutors:
    """Handles LLM streaming, tool execution, chat loops, and sub-agent execution.
    Refactored: shared goal-driven task loop for both chat and agentic (sub-agent) modes."""

    def __init__(self, agent, phase_enum):
        self.agent = agent
        self.Phase = phase_enum

    # ====================== FAST CASUAL DETECTOR ======================
    async def _classify_intent(self, query: str) -> str:
        prompt = f"""Classify this user message as either CASUAL or TASK.

    CASUAL = greeting, small talk, "how are you", joke, opinion, thank you, chit-chat, storytelling, emotional support, philosophy, roleplay, simple general-knowledge questions.
    TASK   = anything that might need tools, search, calculation, research, file/code work, actions, multi-step goal, current events, data lookup, etc.

    User message: {query}

    Answer with **exactly one word**: CASUAL or TASK"""

        try:
            result = await self.agent.client.chat(
                prompt, temperature=0.0, max_tokens=20
            )
            return "CASUAL" if "CASUAL" in result.upper() else "TASK"
        except Exception:
            return "CASUAL" if len(query.split()) < 25 else "TASK"

    def _build_casual_system_prompt(self) -> str:
        return """You are a warm, friendly, engaging AI companion.
        Be natural, use contractions, show personality, be concise unless asked otherwise.
        Never mention tools, internal steps, thinking process, or knowledge base unless the user explicitly asks about them."""

    async def _is_multi_step_task(self, query: str) -> bool:
        """Pure LLM-based check: determines if the request naturally requires multiple steps.
        No hardcoded keywords."""

        prompt = f"""You are an expert at analyzing user requests.

        Determine if this request is:
        - SIMPLE → can be completed with **one direct action** (one tool call or one simple command)
        - COMPLEX → requires **multiple distinct steps**, planning, or multiple tool uses

        Be strict. Most file operations, listings, reads, deletes, etc. are SIMPLE.

        Examples:
        - "list files in this dir"          → SIMPLE
        - "show me the files"               → SIMPLE
        - "read config.json"                → SIMPLE
        - "delete temp.txt"                 → SIMPLE
        - "deploy a web app with database"  → COMPLEX
        - "analyze logs and create report"  → COMPLEX
        - "set up monitoring + alerts"      → COMPLEX

        Request: {query}

        Answer with **exactly one word**: SIMPLE or COMPLEX"""

        try:
            result = await self.agent.client.chat(
                prompt, temperature=0.0, max_tokens=10
            )

            result_upper = result.strip().upper()
            is_complex = "COMPLEX" in result_upper

            logger.info(
                f"[MULTI-STEP] LLM decided → {'COMPLEX (multi-step)' if is_complex else 'SIMPLE (single-step)'} | Query: {query[:100]}..."
            )

            return is_complex  # True = multi-step planning

        except Exception as e:
            logger.warning(
                f"_is_multi_step_task LLM call failed: {e}. Defaulting to single-step."
            )
            return False  # safer default: treat as simple

    async def _goal_driven_task_loop(
        self,
        original_query: str,
        state: AgentState,
        is_sub_agent: bool = False,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """TRULY STATE-DRIVEN goal loop.
        - ALL loop control lives in AgentState + its methods
        - FindTool triggers FULL state reset + immediate restart
        - If tool_results appear in state, stop loop immediately
        """

        logger.info(f"[STATE-DRIVEN TASK LOOP] Starting for: {original_query[:120]}...")

        yield ("phase_changed", {"phase": "thinking"})

        # ====================== MULTI-STEP PLANNING (once only) ======================
        if not state.planned_tasks:
            is_multi_step = await self._is_multi_step_task(original_query)
            if is_multi_step:
                logger.info("[MULTI-STEP] Detected → planning")
                yield ("phase_changed", {"phase": "planning"})
                async for ev, pl in self.agent.flow._ensure_subtasks_planned(
                    state, original_query
                ):
                    yield ev, pl
            else:
                state.planned_tasks = [original_query]

        is_multi_step = len(state.planned_tasks) > 1
        marker = "[FINAL_ANSWER_COMPLETE]"

        self.agent.manager.unload_all()

        # ====================== STATE-DRIVEN MAIN LOOP ======================
        max_loops = 8

        while not state.goal_reached and state.loop_count < max_loops:
            state.increment_loop()

            text_buffer = ""
            tools_called_this_turn = False
            tool_browser_detected = False
            browse_query = original_query

            # ====================== BUILD QUERY (state-aware) ======================
            if is_multi_step and state.current_task_index < len(state.planned_tasks):
                task_desc = state.planned_tasks[state.current_task_index]
                current_query = (
                    f"USER ORIGINAL REQUEST: {original_query}\n\n"
                    f"CURRENT SUB-TASK ({state.current_task_index + 1}/{len(state.planned_tasks)}): {task_desc}\n\n"
                    "Focus ONLY on this sub-task. "
                    f"When finished with this sub-task, end with exactly:\n{marker}\n"
                    "All previous tool results are already in state.tool_results."
                )
            else:
                current_query = (
                    original_query
                    if state.loop_count == 1
                    else (
                        f"USER ORIGINAL REQUEST: {original_query}\n\n"
                        "Previous results are in state.tool_results. Continue or prepare final answer."
                    )
                )

            yield ("phase_changed", {"phase": "searching_tools"})

            # Record tool result count BEFORE LLM turn (we detect external population)
            prev_tool_result_count = len(state.tool_results)

            messages = await self.agent.context_manager.provide_context(
                query=current_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_objective_reminder()
                + f"\n\nWhen you finish the current sub-task, you MUST output exactly: {marker}",
                include_logs=True,
                chat=False,
                state=state,
            )

            queue = await self.agent.client.stream_with_tools(
                manager=self.agent.manager,
                messages=messages,
                temperature=0.2 if not is_sub_agent else self.agent.temperature,
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
                                tool_browser_detected = True
                                browse_query = payload.get("args", {}).get(
                                    "query", original_query
                                )
                                yield ("phase_changed", {"phase": "handling_findtool"})
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

            # ====================== FINDTOOL RESTART (STATE-DRIVEN + FULL RESET) ======================
            if tool_browser_detected:
                logger.info(
                    "[FindTool] Detected → handling interception + FULL state reset"
                )

                # THIS IS THE CRITICAL FIX (original behavior restored via state)
                state.loop_count = 0
                state.empty_loops = 0
                state.action_restarts = 0

                await self._handle_browse_tools_interception(browse_query)
                continue  # ← immediate restart with clean counters

            final_reply = text_buffer.strip()
            has_marker = marker in final_reply
            if has_marker:
                final_reply = final_reply.replace(marker, "").strip()

            # ====================== KEY STATE-DRIVEN CHECK: TOOL RESULTS POPPED UP? ======================
            # We never add results ourselves — we only detect when they appear in state
            if len(state.tool_results) > prev_tool_result_count:
                logger.info(
                    f"Tool results appeared in state ({len(state.tool_results)} total). "
                    "Stopping loop and moving to final answer."
                )
                state.mark_goal_achieved("Tool results populated in state")
                break

            # ====================== ACTION RESTART LOGIC ======================
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
                    logger.warning("Max action restarts reached → forcing completion")
                    state.mark_goal_achieved("Max action restarts reached")
                    break
                else:
                    logger.info(
                        f"[Action Restart #{state.action_restarts}] Detected '{detected_keyword}'"
                    )
                    continue

            # ====================== TERMINATION / EMPTY LOOP LOGIC ======================
            if has_marker:
                if (
                    not is_multi_step
                    or state.current_task_index >= len(state.planned_tasks) - 1
                ):
                    state.mark_goal_achieved("Marker detected + all sub-tasks done")
                    break
                else:
                    logger.info(f"[MULTI-STEP] Sub-task done → advancing index")
                    state.current_task_index += 1
                    continue

            if (
                not tools_called_this_turn
                and not has_marker
                and not action_restart_triggered
            ):
                state.increment_empty_loop()
                if state.empty_loops >= 3:
                    logger.warning("3+ empty loops → forcing completion")
                    state.mark_goal_achieved("Forced completion after empty loops")
                    break
            else:
                state.empty_loops = 0

            # Tools were called but results not yet in state → continue (executor will populate)
            if tools_called_this_turn:
                yield ("phase_changed", {"phase": "executing_tools"})
                continue

        # ====================== FINAL SYNTHESIS (always reached when loop exits) ======================
        yield ("phase_changed", {"phase": "generating_final_answer"})

        final_query = (
            f"USER ORIGINAL REQUEST: {original_query}\n\n"
            f"Task is now complete (or max loops reached). "
            "Summarize using ONLY the verified results from state.tool_results. Be concise."
        )

        final_messages = await self.agent.context_manager.provide_context(
            query=final_query,
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=8000,
            system_prompt=self._build_objective_reminder() + "\n\nFINAL ANSWER MODE",
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

        if not state.goal_reached:
            logger.warning(
                f"Loop ended without explicit goal (loop_count={state.loop_count})"
            )
            state.mark_goal_achieved("Loop termination")

    # ====================== REFACTORED AGENTIC LOOP (for sub-agents) ======================
    async def agentic_loop(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Headless sub-agent loop — now uses the same robust goal-driven logic as chat TASK path."""
        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        original_query = (
            state.current_task
            or self.agent.state.goal
            or "Complete the assigned micro-task"
        )

        async for event, payload in self._goal_driven_task_loop(
            original_query=original_query,
            state=state,
            is_sub_agent=True,
        ):
            yield event, payload

    # ====================== CHAT LOOP (casual path + shared task path) ======================
    async def chat_loop(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.debug("=== CHAT LOOP ===")

        original_user_query = next(
            (
                m.get("content", "").strip()
                for m in reversed(messages or [])
                if m.get("role") == "user" and m.get("content")
            ),
            state.current_task or "[USER REQUEST]",
        )

        intent = await self._classify_intent(original_user_query)
        await self.agent.context_manager.add_message("user", original_user_query)
        logger.info(f"[CHAT INTENT] {intent} | '{original_user_query[:80]}...'")

        state.mode = "casual" if intent == "CASUAL" else "task"
        logger.info(f"→ Agent mode set to: {state.mode.upper()}")

        if intent == "CASUAL":
            logger.info("[CASUAL MODE] Pure basic chat")
            yield ("phase_changed", {"phase": "casual chat"})
            state.mode = "casual"

            casual_messages = await self.agent.context_manager.provide_context(
                query=original_user_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_casual_system_prompt(),
                include_logs=False,
                chat=True,
            )

            final_reply = await self.agent.client.chat(
                casual_messages, temperature=0.85, top_p=0.95
            )

            await self.agent.context_manager.add_message("assistant", final_reply)
            yield ("llm_response", {"full_reply": final_reply, "is_complete": True})
            return

        # TASK path — use the shared robust loop
        async for event, payload in self._goal_driven_task_loop(
            original_query=original_user_query,
            state=state,
            is_sub_agent=False,
        ):
            yield event, payload

    # ====================== SMART TWO-STAGE FindTool INTERCEPTION (unchanged) ======================
    async def _handle_browse_tools_interception(self, original_user_query: str):
        # (your exact code from the paste — unchanged)
        logger.info(f"[FindTool INTERCEPT] original='{original_user_query[:150]}...'")

        refinement_prompt = f"""You are an expert tool-router.
        Convert this user request into a SHORT, keyword-rich query (max 12 words) that will perfectly match the best tool in our registry.

        USER REQUEST: {original_user_query}

        Rules:
        - Focus ONLY on the core capability needed (e.g. "web search weather", "read pdf file", "search codebase function").
        - Use terms that appear in tool names/descriptions/tags.
        - Do NOT include specific data values (e.g. do NOT put "Poland").
        - Return ONLY the refined query as plain text, nothing else.

        Refined query:"""

        refined_query = await self.agent.client.chat(
            refinement_prompt, temperature=0.0, max_tokens=80
        )
        refined_query = refined_query.strip().strip("\"'").strip()
        logger.info(f"[FindTool] REFINED search query → '{refined_query}'")

        all_tools = registry.list_all_tools(limit=120)
        tool_list_str = "\n".join(
            f"• {t['name']} (@{t['set_name']}) — {t['description'][:110]}"
            for t in all_tools
        )

        selection_prompt = f"""REFINED INTENT (use this for matching): {refined_query}

        USER ORIGINAL REQUEST: {original_user_query}

        AVAILABLE TOOLS:
        {tool_list_str}

        You MUST pick the SINGLE best tool that can fulfill the refined intent.
        Return ONLY valid JSON (no extra text, no markdown):

        {{
        "tool_name": "exact_tool_name",
        "parameters": {{ "key": "value", ... }},
        "reason": "one short sentence"
        }}

        If no tool is clearly better than others, still pick the closest one."""

        selection_text = await self.agent.client.chat(
            selection_prompt, temperature=0.0, max_tokens=700
        )

        try:
            choice = json.loads(selection_text.strip())
            tool_name = choice.get("tool_name")
            params = choice.get("parameters", {}) or {}
            reason = choice.get("reason", "")

            if tool_name and tool_name in registry.all_actions:
                self.agent.manager.load_tools([tool_name])
                logger.info(f"[FindTool] LLM selected → {tool_name} | reason: {reason}")

                enhanced_directive = (
                    f"Use the '{tool_name}' tool to handle the request. "
                    f"Parameters: {json.dumps(params) if params else 'none needed'}.\n"
                    f"Refined intent was: {refined_query}"
                )

                await self.agent.context_manager.add_message(
                    "system", f"[SMART TOOL ROUTER] {enhanced_directive}"
                )

                await self.agent.context_manager.record_tool_outcome(
                    tool_name="FindTool",
                    result=f"Selected {tool_name} for refined query '{refined_query}'",
                    metadata={
                        "refined_query": refined_query,
                        "original": original_user_query[:200],
                    },
                )
                return

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"[FindTool] Selection JSON parse failed: {e}. Falling back..."
            )

        fallback_results = registry.search(refined_query, limit=3)
        if fallback_results:
            names = [a.name for a, _ in fallback_results]
            self.agent.manager.load_tools(names)
            logger.info(f"[FindTool] Fallback loaded via refined query: {names}")

            await self.agent.context_manager.add_message(
                "system",
                f"[SMART TOOL ROUTER FALLBACK] Loaded tools for refined intent '{refined_query}': {', '.join(names)}",
            )
        else:
            logger.warning(
                f"[FindTool] No tools found even after refinement for '{refined_query}'"
            )

    # ====================== HELPERS ======================
    def _build_objective_reminder(self) -> str:
        return f"Current goal: {self.agent.state.goal or 'No goal set'}"

    def _build_sub_agent_objective_reminder(self) -> str:
        base = f"Current goal: {self.agent.state.goal or 'Complete the assigned micro-task'}"
        return (
            base
            + "\n\nWhen you have fully completed the task, end your response with exactly: [FINAL_ANSWER_COMPLETE]"
        )
