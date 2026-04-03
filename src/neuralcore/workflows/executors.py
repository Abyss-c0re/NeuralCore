import asyncio
import json
from typing import Awaitable
from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

from neuralcore.actions.actions import Action
from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet
from neuralcore.utils.formatting import safe_json_dumps
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class AgentExecutors:
    """Handles all LLM streaming, tool execution, chat loops, and sub-agent execution.
    NOW WITH RESTORED BASIC CASUAL CHAT + full goal-driven tasks."""

    def __init__(self, agent, phase_enum):
        self.agent = agent
        self.Phase = phase_enum
        self.engine = agent.workflow

    # ====================== FAST CASUAL DETECTOR (the fix) ======================
    async def _classify_intent(self, query: str) -> str:
        """One-word CASUAL vs TASK check. Extremely strict so greetings stay natural."""
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

    # ====================== TOOL EXECUTION (unchanged) ======================
    async def _execute_tool(self, name: str, args: dict):
        executor = self.agent.manager.get_executor(name, self.agent)
        if not executor:
            raise RuntimeError(f"No executor found for tool '{name}'")
        result = executor(**args)
        return await result if asyncio.iscoroutine(result) else result

    # ====================== AGENTIC LOOP (exactly as you pasted) ======================
    async def agentic_loop(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        # (your exact code from the paste — unchanged)
        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        task_name = state.current_task or f"agentic_task_{iteration}"
        task_ctx = self.agent.context_manager.create_task_context(task_name)
        self.agent.context_manager.set_goal(self.agent.goal or task_name)
        self.agent.context_manager.add_subtask(task_name)

        max_loops = 15
        loop_count = 0
        final_answer_marker = "[FINAL_ANSWER_COMPLETE]"

        while loop_count < max_loops:
            loop_count += 1
            tool_browser_detected = False
            browse_query = state.current_task or ""
            text_buffer = ""
            tool_results: List[str] = []
            assistant_message = ""

            messages = await self.agent.context_manager.provide_context(
                query=state.current_task or "Continue and complete the task",
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_sub_agent_objective_reminder(),
                include_logs=True,
            )

            queue = await self.agent.client.stream_with_tools(
                messages=messages,
                tools=tools or self.agent.manager.get_llm_tools(),
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=self._execute_tool,
            )

            try:
                async for item in self.agent.client._drain_queue(queue):
                    if item is None:
                        continue
                    if not isinstance(item, tuple) or len(item) != 2:
                        continue

                    kind, payload = item

                    if kind == "content":
                        content = str(payload) if payload is not None else ""
                        text_buffer += content
                        assistant_message += content
                        yield ("content_delta", content)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = (
                                payload.get("tool_name")
                                or payload.get("name")
                                or "unknown"
                            )
                            result = payload.get("result") or payload.get("output")
                            args = payload.get("args", {}) or {}

                            if "FindTool" in tool_name:
                                tool_browser_detected = True
                                browse_query = args.get(
                                    "query", state.current_task or ""
                                )
                                logger.info(
                                    f"[FindTool] detected in agentic_loop. Query='{browse_query[:120]}...'"
                                )
                                break

                            content_str = (
                                json.dumps(result, ensure_ascii=False, default=str)
                                if isinstance(result, dict)
                                else str(result or "No output")
                            )

                            await self.agent.context_manager.record_tool_outcome(
                                tool_name=tool_name,
                                result=content_str,
                                metadata={
                                    "task": task_name,
                                    "loop": loop_count,
                                    "sub_agent": True,
                                },
                            )

                            await task_ctx.add_important_result(
                                title=tool_name,
                                content=content_str,
                                source="tool",
                                metadata={"loop": loop_count},
                            )

                            if content_str.strip():
                                tool_results.append(content_str.strip())

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except Exception as e:
                logger.error(f"Stream error in agentic_loop: {e}", exc_info=True)
                yield ("error", str(e))
                return

            if tool_browser_detected:
                await self._handle_browse_tools_interception(browse_query)
                continue

            final_reply = text_buffer.strip()
            if final_answer_marker in final_reply:
                final_reply = final_reply.replace(final_answer_marker, "").strip()

            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)

            if not final_reply:
                final_reply = "✅ Task completed."

            await self.agent.context_manager.add_message("assistant", final_reply)

            await task_ctx.add_important_result(
                title="Final Task Outcome",
                content=final_reply,
                source="agentic_loop_completion",
                metadata={"iteration": loop_count, "task": task_name},
            )

            if getattr(self.agent, "sub_agent", False):
                parent = getattr(self.agent, "parent", None)
                if parent:
                    try:
                        await parent.context_manager.add_external_content(
                            source_type="sub_task_final_reply",
                            content=final_reply,
                            metadata={"origin": self.agent.agent_id},
                        )
                    except Exception as e:
                        logger.warning(f"Parent propagation failed: {e}")

            self.agent.context_manager.prune_sub_agent_noise()
            self.agent.context_manager.complete_subtask(task_name)

            yield (
                "llm_response",
                {
                    "full_reply": final_reply,
                    "tool_calls": [],
                    "is_complete": True,
                },
            )
            break

        else:
            logger.warning(f"agentic_loop reached max iterations ({max_loops})")
            final_reply = "⚠️ Maximum iterations reached while executing sub-task."
            await self.agent.context_manager.add_message("assistant", final_reply)
            await task_ctx.add_important_result(
                title="Max Iterations Reached",
                content=final_reply,
                source="agentic_loop",
            )
            yield (
                "llm_response",
                {
                    "full_reply": final_reply,
                    "is_complete": True,
                },
            )

    def _build_sub_agent_objective_reminder(self) -> str:
        base = f"Current goal: {self.agent.goal or 'Complete the assigned micro-task'}"
        return (
            base
            + "\n\nWhen you have fully completed the task, end your response with exactly: [FINAL_ANSWER_COMPLETE]"
        )

    # ====================== CHAT LOOP — BASIC CASUAL RESTORED + GOAL-DRIVEN TASK PATH ======================
    # ====================== CHAT LOOP (status updates restored + mode switching) ======================
    async def chat_loop(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.debug("=== CHAT LOOP (basic casual + full status for tasks) ===")

        original_user_query = next(
            (
                m.get("content", "").strip()
                for m in reversed(messages or [])
                if m.get("role") == "user" and m.get("content")
            ),
            state.current_task or "[USER REQUEST]",
        )

        # ── EARLY CASUAL PATH (unchanged — fast & clean) ──
        intent = await self._classify_intent(original_user_query)
        logger.info(f"[CHAT INTENT] {intent} | '{original_user_query[:80]}...'")

        # Always re-evaluate mode on every message
        state.mode = "casual" if intent == "CASUAL" else "task"
        logger.info(f"→ Agent mode set to: {state.mode.upper()}")

        if intent == "CASUAL":
            logger.info("[CASUAL MODE] Pure basic chat — NO goal lock, NO tools")
            yield ("phase_changed", {"phase": "casual chat"})
            state.mode = "casual"

            # Exactly like your old basic chat: full history via ContextManager
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

            yield (
                "llm_response",
                {"full_reply": final_reply, "is_complete": True},
            )
            return  # ← critical: exit immediately

        # ── TASK PATH (full goal-driven + RESTORED status updates) ──
        logger.info("[TASK MODE] Full goal-driven tool loop")
        yield ("phase_changed", {"phase": "thinking"})  # ← restored

        max_tool_loops = 20
        loop_count = 0

        while loop_count < max_tool_loops:
            loop_count += 1
            tool_browser_detected = False
            browse_query = original_user_query
            complex_action_called = False
            complex_reason = ""
            tools_called_this_turn = False

            if loop_count == 1:
                current_query = original_user_query
            else:
                current_query = (
                    f"USER REQUEST: {original_user_query}\n\n"
                    "Previous tool results are already stored in KB.\n"
                    "You may call more tools if needed, otherwise prepare for final answer."
                )

            # Status: about to call LLM + tools
            yield ("phase_changed", {"phase": "searching_tools"})  # ← restored

            cm_messages = await self.agent.context_manager.provide_context(
                query=current_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_objective_reminder(),
                include_logs=True,
                chat=True,
            )

            queue = await self.agent.client.stream_with_tools(
                messages=cm_messages,
                tools=self.agent.manager.get_action_set("DynamicCore"),
                temperature=0.2,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=self._execute_tool,
            )

            try:
                async for item in self.agent.client._drain_queue(queue):
                    if item is None:
                        continue
                    if not isinstance(item, tuple) or len(item) != 2:
                        continue

                    kind, payload = item

                    if kind == "content":
                        pass

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = (
                                payload.get("tool_name")
                                or payload.get("name")
                                or "unknown"
                            )
                            result = payload.get("result") or payload.get("output")
                            args = payload.get("args", {}) or {}

                            if "FindTool" in tool_name:
                                tool_browser_detected = True
                                browse_query = args.get("query", original_user_query)
                                logger.info(
                                    f"[FindTool] intercepted in chat_loop. Query='{browse_query[:120]}...'"
                                )
                                yield (
                                    "phase_changed",
                                    {"phase": "handling_findtool"},
                                )  # ← restored
                                break

                            if tool_name == "RequestComplexAction":
                                complex_action_called = True
                                complex_reason = str(
                                    payload.get("args", {}).get("reason", "")
                                )
                                break

                            tools_called_this_turn = True

                            content_str = (
                                json.dumps(result, ensure_ascii=False, default=str)
                                if isinstance(result, dict)
                                else str(result or "No output")
                            )

                            await self.agent.context_manager.record_tool_outcome(
                                tool_name=tool_name,
                                result=content_str,
                                metadata={"loop": loop_count, "chat_mode": True},
                            )

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except Exception as e:
                logger.error(f"Stream error in chat_loop: {e}", exc_info=True)
                yield ("error", str(e))
                return

            if tool_browser_detected:
                await self._handle_browse_tools_interception(browse_query)
                continue

            if complex_action_called:
                self.agent.task = complex_reason
                self.agent.goal = complex_reason
                final_reply = f"✅ Starting multi-step orchestration for: **{complex_reason[:120]}**..."
                await self.agent.context_manager.add_message("assistant", final_reply)
                yield ("llm_response", {"full_reply": final_reply, "is_complete": True})
                await self.agent.post_control(
                    {"event": "switch_workflow", "name": "default"}
                )
                return

            if tools_called_this_turn and loop_count < max_tool_loops:
                yield ("phase_changed", {"phase": "executing_tools"})  # ← restored
                logger.info(f"Tools executed (loop {loop_count}) – continuing silently")
                continue

            # ── STRONG FINAL SYNTHESIS PASS ──
            yield ("phase_changed", {"phase": "generating_final_answer"})  # ← restored

            final_query = (
                f"USER ORIGINAL REQUEST: {original_user_query}\n\n"
                "You now have ALL tool results stored in the KB / provided context.\n"
                "VERY IMPORTANT RULES:\n"
                "• Answer ONLY using the exact information from the tool results in the context.\n"
                "• DO NOT invent, hallucinate, or add any files, folders, or results that were not returned.\n"
                "• Be direct, concise, and factual.\n"
                "• Do not mention tools, KB, or internal steps."
            )

            final_messages = await self.agent.context_manager.provide_context(
                query=final_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=8000,
                system_prompt=self._build_objective_reminder()
                + "\n\nYou are in FINAL ANSWER MODE. Be precise and factual.",
                include_logs=True,
                chat=True,
            )

            final_reply = await self.agent.client.chat(
                final_messages, temperature=0.0, top_p=0.1
            )

            await self.agent.context_manager.add_message("assistant", final_reply)

            yield (
                "llm_response",
                {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
            )
            break

        if loop_count >= max_tool_loops:
            logger.warning("Max tool loops reached in chat_loop")

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

        all_tools = self.agent.registry.list_all_tools(limit=120)
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

            if tool_name and tool_name in self.agent.registry.all_actions:
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

        fallback_results = self.agent.registry.search(refined_query, limit=3)
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
        return f"Current goal: {self.agent.goal or 'No goal set'}"
