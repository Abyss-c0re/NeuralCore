import asyncio
import json
from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class AgentExecutors:
    """SOTA 2026 AgentExecutors – goal-locked + loop-until-success + failure reflection.
    Fixed: 'results' is now always bound in both loops."""

    def __init__(self, agent, phase_enum):
        self.agent = agent
        self.Phase = phase_enum

    async def _execute_tool(self, name: str, args: dict):
        executor = self.agent.manager.get_executor(name, self.agent)
        if not executor:
            raise RuntimeError(f"No executor found for tool '{name}'")
        result = executor(**args)
        return await result if asyncio.iscoroutine(result) else result

    async def _execute_tools_in_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """Parallel execution + deduplication + failure flag."""
        if not tool_calls:
            return []
        seen = set()
        unique = []
        for c in tool_calls:
            key = (c["name"], json.dumps(c.get("args", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        results = await asyncio.gather(
            *(self._execute_tool(c["name"], c.get("args", {})) for c in unique),
            return_exceptions=True,
        )

        processed = []
        for call, raw in zip(unique, results):
            if isinstance(raw, Exception):
                content = f"ERROR: {str(raw)}"
            else:
                content = (
                    json.dumps(raw, ensure_ascii=False, default=str)
                    if isinstance(raw, dict)
                    else str(raw or "No output")
                )
            processed.append(
                {
                    "tool_name": call["name"],
                    "result": content,
                    "args": call.get("args", {}),
                    "is_failure": any(
                        w in content.lower()
                        for w in [
                            "error",
                            "failed",
                            "not found",
                            "unable",
                            "try again",
                            "exception",
                        ]
                    ),
                }
            )
        return processed

    async def _run_one_time_discovery_and_lock_goal(self, query: str):
        """One-time FindTool + goal lock (prevents wrong-tool loops)."""
        logger.info(f"[GOAL LOCK] Setting objective: {query[:100]}...")
        self.agent.context_manager.set_goal(query)
        await self.agent.context_manager.add_message(
            "system", f"OBJECTIVE LOCKED: {query}"
        )

        await self._handle_browse_tools_interception(query)

        # Filter FindTool for the rest of this turn (keeps your dynamic philosophy 100% intact)
        dynamic_set = self.agent.manager.get_action_set("DynamicCore")
        self._tools_for_this_turn = [
            t for t in dynamic_set.actions if getattr(t, "name", "") != "FindTool"
        ]

    async def _reflect_and_decide(
        self, query: str, loop_count: int, tool_results: List[Dict]
    ) -> bool:
        """Return True = success → go to final synthesis."""
        failures = [r for r in tool_results if r.get("is_failure")]
        if not failures:
            return True

        logger.info(f"[REFLECTION] {len(failures)} failures – re-evaluating...")
        for f in failures:
            await self.agent.context_manager.add_unknown(
                f"Tool {f['tool_name']} failed: {f['result'][:200]}"
            )

        reflect_prompt = f"""USER GOAL: {query}
We ran {loop_count} tool rounds. {len(failures)} tools returned errors or no useful output.

Current KB state: {len(self.agent.context_manager.investigation_state.get("findings", []))} findings, 
{len(self.agent.context_manager.investigation_state.get("unknowns", []))} unknowns.

Do we now have enough correct information to give a complete final answer?
Answer ONLY: YES or NEED_MORE"""

        decision = await self.agent.client.chat(
            reflect_prompt, temperature=0.0, max_tokens=120
        )
        if "YES" in decision.upper():
            logger.info("Reflection → SUCCESS. Moving to final synthesis.")
            return True

        # Wrong tool suspected → re-discover
        if any(w in decision.lower() for w in ["tool", "findtool", "better tool"]):
            logger.info("[REFLECTION] Wrong tool suspected → re-running discovery")
            await self._run_one_time_discovery_and_lock_goal(query)

        return False

    # ====================== AGENTIC LOOP ======================
    async def agentic_loop(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        task_name = state.current_task or f"agentic_task_{iteration}"
        task_ctx = self.agent.context_manager.create_task_context(task_name)
        self.agent.context_manager.add_subtask(task_name)

        await self._run_one_time_discovery_and_lock_goal(
            state.current_task or task_name
        )

        max_loops = 12
        loop_count = 0
        final_marker = "[FINAL_ANSWER_COMPLETE]"

        text_buffer = ""  # always bound
        tool_results: List[Dict] = []
        include_logs_this_turn = False  # start without logs, enable on failure retry

        while loop_count < max_loops:
            loop_count += 1
            text_buffer = ""  # reset per iteration
            tool_results = []  # reset per iteration

            messages = await self.agent.context_manager.provide_context(
                query=state.current_task or "Continue and complete the task",
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_sub_agent_objective_reminder(),
                include_logs=include_logs_this_turn,  # logs only on retry after failure
                # force logs on every retry
            )

            queue = await self.agent.client.stream_with_tools(
                messages=messages,
                tools=self._tools_for_this_turn,
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=self._execute_tool,
            )

            tool_calls_this_turn: List[Dict] = []

            async for item in self.agent.client._drain_queue(queue):
                if item is None or not isinstance(item, tuple) or len(item) != 2:
                    continue
                kind, payload = item

                if kind == "content":
                    text_buffer += str(payload or "")
                    yield ("content_delta", str(payload or ""))

                elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                    if isinstance(payload, dict):
                        name = (
                            payload.get("tool_name") or payload.get("name") or "unknown"
                        )
                        args = payload.get("args", {}) or {}
                        tool_calls_this_turn.append({"name": name, "args": args})

                elif kind == "finish":
                    break
                elif kind in ("error", "cancelled"):
                    yield (kind, payload)
                    return

            if tool_calls_this_turn:
                tool_results = await self._execute_tools_in_parallel(
                    tool_calls_this_turn
                )
                for r in tool_results:
                    await self.agent.context_manager.record_tool_outcome(
                        tool_name=r["tool_name"],
                        result=r["result"],
                        metadata={
                            "task": task_name,
                            "loop": loop_count,
                            "sub_agent": True,
                        },
                    )
                    await task_ctx.add_important_result(
                        title=r["tool_name"],
                        content=r["result"],
                        source="tool",
                        metadata={"loop": loop_count, "failure": r["is_failure"]},
                    )

            success = await self._reflect_and_decide(
                state.current_task or task_name, loop_count, tool_results
            )
            if success or not tool_calls_this_turn:
                break

            # Failed → we are restarting the loop → enable logs for next iteration
            include_logs_this_turn = True

        # Final reply
        final_reply = text_buffer.strip()
        if final_marker in final_reply:
            final_reply = final_reply.replace(final_marker, "").strip()
        if not final_reply:
            final_reply = "✅ Task completed."

        await self.agent.context_manager.add_message("assistant", final_reply)
        await task_ctx.add_important_result(
            title="Final Task Outcome",
            content=final_reply,
            source="agentic_loop_completion",
        )

        if getattr(self.agent, "sub_agent", False) and getattr(
            self.agent, "parent", None
        ):
            await self.agent.parent.context_manager.add_external_content(
                source_type="sub_task_final_reply",
                content=final_reply,
                metadata={"origin": self.agent.agent_id},
            )

        self.agent.context_manager.prune_sub_agent_noise()
        self.agent.context_manager.complete_subtask(task_name)

        yield ("llm_response", {"full_reply": final_reply, "is_complete": True})

    # ====================== CHAT LOOP ======================
    async def chat_loop(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.debug(
            "=== ENTERING SOTA chat_loop (goal-locked + retry-on-failure + logs on retry) ==="
        )

        original_user_query = next(
            (
                m.get("content", "").strip()
                for m in reversed(messages or [])
                if m.get("role") == "user" and m.get("content")
            ),
            state.current_task or "[USER REQUEST]",
        )

        chat_task_ctx = self.agent.context_manager.create_task_context("chat_user_turn")
        await self._run_one_time_discovery_and_lock_goal(original_user_query)

        max_tool_loops = 12
        loop_count = 0

        include_logs_this_turn = (
            False  # start clean, enable logs only after failure/retry
        )
        tool_results: List[Dict] = []  # always bound

        while loop_count < max_tool_loops:
            loop_count += 1
            tool_results = []  # reset for this iteration
            tool_calls_this_turn: List[Dict] = []
            complex_action_called = False
            complex_reason = ""

            current_query = (
                original_user_query
                if loop_count == 1
                else f"USER REQUEST: {original_user_query}\nPrevious results are in KB. Continue until goal is achieved."
            )

            cm_messages = await self.agent.context_manager.provide_context(
                query=current_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_objective_reminder(),
                include_logs=include_logs_this_turn,  # logs only on retry after failure
                chat=True,
            )

            queue = await self.agent.client.stream_with_tools(
                messages=cm_messages,
                tools=self._tools_for_this_turn,
                temperature=0.2,
                max_tokens=self.agent.max_tokens,
                tool_choice="auto",
                executor_callback=self._execute_tool,
            )

            async for item in self.agent.client._drain_queue(queue):
                if item is None or not isinstance(item, tuple) or len(item) != 2:
                    continue
                kind, payload = item

                if kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                    if isinstance(payload, dict):
                        name = (
                            payload.get("tool_name") or payload.get("name") or "unknown"
                        )
                        args = payload.get("args", {}) or {}
                        if name == "RequestComplexAction":
                            complex_action_called = True
                            complex_reason = str(
                                payload.get("args", {}).get("reason", "")
                            )
                            break
                        tool_calls_this_turn.append({"name": name, "args": args})

                elif kind == "finish":
                    break
                elif kind in ("error", "cancelled"):
                    yield (kind, payload)
                    return

            if complex_action_called:
                # your original complex action handling (unchanged)
                self.agent.task = complex_reason
                self.agent.goal = complex_reason
                final_reply = f"✅ Starting multi-step orchestration for: **{complex_reason[:120]}**..."
                await self.agent.context_manager.add_message("assistant", final_reply)
                yield ("llm_response", {"full_reply": final_reply, "is_complete": True})
                await self.agent.post_control(
                    {"event": "switch_workflow", "name": "default"}
                )
                return

            if tool_calls_this_turn:
                tool_results = await self._execute_tools_in_parallel(
                    tool_calls_this_turn
                )
                for r in tool_results:
                    await self.agent.context_manager.record_tool_outcome(
                        tool_name=r["tool_name"],
                        result=r["result"],
                        metadata={"loop": loop_count, "chat_mode": True},
                    )
                    await chat_task_ctx.add_important_result(
                        title=r["tool_name"],
                        content=r["result"],
                        source="tool",
                        metadata={"failure": r["is_failure"]},
                    )

            # === REFLECTION + LOOP UNTIL SUCCESS ===
            success = await self._reflect_and_decide(
                original_user_query, loop_count, tool_results
            )
            if success or not tool_calls_this_turn:
                break

            # Failed → restarting the loop → enable logs for next iteration
            include_logs_this_turn = True

        # === FINAL SYNTHESIS ===
        final_query = (
            f"USER ORIGINAL REQUEST: {original_user_query}\n"
            "Answer ONLY using the exact information from the KB/tool results. "
            "Be direct, concise, and factual."
        )
        final_messages = await self.agent.context_manager.provide_context(
            query=final_query,
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=8000,
            system_prompt=self._build_objective_reminder()
            + "\n\nYou are in FINAL ANSWER MODE.",
            include_logs=True,  # final pass always includes logs
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

    def _build_sub_agent_objective_reminder(self) -> str:
        base = f"Current goal: {self.agent.goal or 'Complete the assigned micro-task'}"
        return (
            base + "\n\nWhen fully complete, end with exactly: [FINAL_ANSWER_COMPLETE]"
        )

    # ====================== SMART TWO-STAGE FindTool INTERCEPTION ======================
    async def _handle_browse_tools_interception(self, original_user_query: str):
        """
        SMART interception (exactly as requested):
          1. LLM distills raw user request into a clean, short tool-search query
             (optimized for registry lexical/fuzzy matching).
          2. LLM selects the SINGLE best tool + parameters from the full live registry.
          3. Load the chosen tool(s).
          4. Inject an enhanced system directive so the next stream restart uses:
             "use web search to check the weather in Poland".
        Fully dynamic — no hardcoding, works with any externally-loaded @tool.
        """
        logger.info(f"[FindTool INTERCEPT] original='{original_user_query[:150]}...'")

        # ── Stage 1: Refine into precise tool-discovery query ──
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

        # ── Stage 2: LLM picks the SINGLE best tool using the refined query ──
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

                # Enhanced directive → this becomes the "restart stream with enhanced original prompt"
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

        # ── Fallback: refined-query lexical search ──
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
