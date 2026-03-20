import asyncio
import hashlib
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable, Tuple

from neuralcore.core.client import LLMClient
from neuralcore.actions.actions import Action, ActionSet
from neuralcore.actions.manager import DynamicActionManager
from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger

ToolProvider = Union[ActionSet, DynamicActionManager, List[Dict[str, Any]]]
ToolExecutorGetter = Optional[Callable[[str], Optional["Action"]]]

logger = Logger.get_logger()


class AgentRunner:
    """
    Compatible upgraded AgentRunner.

    Keeps:
    - __init__ parameters unchanged
    - run() parameters unchanged
    - yield event names unchanged

    Improvements:
    - explicit phase/state tracking
    - stronger goal anchoring every turn
    - stagnation detection
    - reflection with forced strategy shift
    - casual-chat fast exit without task reports
    - tool browser discovery handled as a real state transition
    - duplicate tool / no-progress loop control
    """

    class Phase(str, Enum):
        CASUAL = "casual"
        TOOL_DISCOVERY = "tool_discovery"
        EXECUTE = "execute"
        REFLECT = "reflect"
        FINALIZE = "finalize"

    def __init__(
        self,
        client: "LLMClient",
        max_iterations: int = 25,
        max_reflections: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 12048,
    ):
        self.client = client
        self.max_iterations = max_iterations
        self.max_reflections = max_reflections
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.executed_signatures: set[tuple] = set()
        self.reflection_count = 0
        self.tool_results: List[Dict] = []
        self.messages: List[Dict] = []
        self._stop_event: Optional[asyncio.Event] = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _should_stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _safe_truncate(self, text: str, limit: int) -> str:
        text = text or ""
        return text if len(text) <= limit else text[:limit].rstrip() + "..."

    def _tool_signature(self, name: str, args: Dict[str, Any]) -> tuple:
        try:
            normalized = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        except Exception:
            normalized = str(args)
        return (name, normalized)

    def _turn_fingerprint(self, reply: str, tool_calls: Any) -> str:
        payload = {
            "reply": (reply or "")[:600],
            "tool_calls": [
                (
                    c.get("function", {}).get("name"),
                    c.get("function", {}).get("arguments"),
                )
                for c in (tool_calls or [])
                if isinstance(c, dict)
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _is_casual_prompt(self, prompt: str) -> bool:
        t = (prompt or "").strip().lower()
        if not t:
            return True

        casual_markers = (
            "hello",
            "hi ",
            "hey",
            "how are you",
            "thanks",
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
            "what's up",
            "whats up",
            "ok",
            "okay",
            "cool",
            "nice",
        )
        if any(m in t for m in casual_markers):
            return True

        if len(t.split()) <= 5 and "?" not in t:
            return True

        return False

    def _looks_like_final_answer(self, reply: str) -> bool:
        t = (reply or "").strip().lower()
        if not t:
            return False

        unresolved_markers = (
            "i need more information",
            "i need more info",
            "cannot proceed",
            "can't proceed",
            "please provide",
            "need clarification",
            "i'm not sure",
            "i am not sure",
            "unable to",
            "i would need",
        )
        if any(m in t for m in unresolved_markers):
            return False

        if t.endswith("?") and len(t.split()) > 10:
            return False

        return True

    def _is_real_task(self, user_prompt: str) -> bool:
        """
        Determines whether the interaction is a real task vs casual conversation.
        Uses execution signals first, then prompt intent heuristics.
        """
        if self.tool_results:
            return True
        if len(self.executed_signatures) > 0:
            return True
        if self.reflection_count > 0:
            return True

        prompt = (user_prompt or "").lower()
        task_keywords = [
            "create",
            "build",
            "generate",
            "write",
            "fix",
            "debug",
            "analyze",
            "search",
            "find",
            "install",
            "run",
            "execute",
            "explain how",
            "step by step",
            "make",
            "edit",
            "improve",
            "refactor",
            "convert",
            "compare",
            "summarize",
            "plan",
        ]
        return any(k in prompt for k in task_keywords)

    def _get_context_summary(self, context_manager) -> str:
        try:
            if hasattr(context_manager, "get_context_summary"):
                return context_manager.get_context_summary() or ""
        except Exception:
            pass
        return ""

    def _recent_failures_text(self, context_manager, limit: int = 5) -> str:
        try:
            nf = getattr(context_manager, "fs_state", {}).get("negative_findings", [])
            if not nf:
                return ""
            tail = nf[-limit:]
            lines = []
            for item in tail:
                try:
                    tool_name, path = item
                    lines.append(f"- {tool_name}: {path}")
                except Exception:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _recent_tools_text(self, context_manager, limit: int = 5) -> str:
        try:
            tools = getattr(context_manager, "tools_executed", []) or []
            tail = tools[-limit:]
            return ", ".join(str(t) for t in tail) if tail else "—"
        except Exception:
            return "—"

    def _is_tool_browser(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        n = tool_name.lower()
        return (
            "browse" in n
            or "tool_browser" in n
            or n in {"browser", "browse_tools", "toolbrowser"}
        )

    def _extract_browser_tools(self, result: Any) -> List[str]:
        """
        Best-effort extraction of tool names from browse_tools-like results.
        Supports dict/list/string payloads.
        """
        tools: List[str] = []

        if isinstance(result, dict):
            loaded = result.get("loaded_tools")
            if isinstance(loaded, list):
                tools.extend(str(x) for x in loaded if x)

            matching = result.get("matching_tools")
            if isinstance(matching, list):
                for item in matching:
                    if isinstance(item, dict) and item.get("name"):
                        tools.append(str(item["name"]))

        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and item.get("name"):
                    tools.append(str(item["name"]))

        elif isinstance(result, str):
            for token in (
                "ls",
                "cat",
                "read_file",
                "write_file",
                "touch",
                "pwd",
                "find",
            ):
                if token in result:
                    tools.append(token)

        seen = set()
        out = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:12]

    def _build_browser_followup_message(self, discovered_tools: List[str]) -> str:
        if discovered_tools:
            tools_text = ", ".join(discovered_tools)
            return (
                "TOOL DISCOVERY COMPLETE.\n"
                f"Available concrete tools: {tools_text}\n\n"
                "Next step: choose one concrete tool that directly solves the user's request. "
                "Do not reflect yet. Do not browse tools again unless the available tools are insufficient."
            )
        return (
            "TOOL DISCOVERY COMPLETE.\n"
            "Use the discovered tools directly on the next step. Do not reflect yet."
        )

    def _build_runtime_system_prompt(
        self,
        base_system_prompt: str,
        user_prompt: str,
        iteration: int,
        context_manager,
        has_tools: bool,
        stagnation_count: int,
        reflection_count: int,
    ) -> str:
        summary = self._get_context_summary(context_manager)
        failures = self._recent_failures_text(context_manager)
        recent_tools = self._recent_tools_text(context_manager)

        parts = [base_system_prompt.strip()]

        if summary:
            parts.append(summary)

        parts.append(
            "EXECUTION RULES:\n"
            "- Stay focused on the original goal.\n"
            "- Do not repeat failed actions.\n"
            "- If you have enough information, finish cleanly.\n"
            "- If you need tools, choose them explicitly and make progress.\n"
            "- If you are stuck, change strategy rather than repeating the same path."
        )

        parts.append(f"GOAL: {self._safe_truncate(user_prompt, 500)}")
        parts.append(
            f"STATE: iteration={iteration}, reflections={reflection_count}, "
            f"stagnation={stagnation_count}, has_tools={has_tools}"
        )
        parts.append(f"RECENT TOOLS: {recent_tools}")

        if failures:
            parts.append(
                "KNOWN FAILURES / DEAD ENDS (avoid repeating these exact paths):\n"
                f"{failures}"
            )

        if iteration == 1 and has_tools:
            parts.append(
                "FIRST ITERATION GUIDANCE:\n"
                "- If the task needs tool selection or the correct tool is not obvious, "
                "use the tool browser first to select the right tools.\n"
                "- If the correct tool is obvious, use it directly."
            )

        parts.append(
            "When the task is fully solved and no more actions are needed, "
            "append EXACTLY this marker at the very end of the response: "
            "[FINAL_ANSWER_COMPLETE]"
        )

        return "\n\n".join(p for p in parts if p.strip())

    def _is_exploratory_tool(self, name: str) -> bool:
        n = (name or "").lower()
        return n in {
            "search",
            "browse",
            "browse_tools",
            "tool_browser",
            "browser",
            "ls",
            "pwd",
            "find",
        }

    def _should_reflect(
        self,
        *,
        iteration: int,
        is_task_complete: bool,
        tool_calls: Any,
        no_progress_turns: int,
        same_fingerprint_count: int,
        reflection_count: int,
        casual_mode: bool,
        browser_followup_turns: int,
        tool_exploration_lock: int,
        distinct_tools_tried: int,
    ) -> bool:
        if casual_mode:
            return False
        if is_task_complete:
            return False
        if iteration <= 1:
            return False
        if reflection_count >= self.max_reflections:
            return False
        if browser_followup_turns > 0:
            return False

        if tool_exploration_lock > 0:
            return False

        if distinct_tools_tried < 2 and self.tool_results:
            return False

        if no_progress_turns < 3:
            return False

        if self.tool_results and no_progress_turns == 0:
            return False

        if not tool_calls and (no_progress_turns >= 1 or same_fingerprint_count >= 1):
            return True

        if same_fingerprint_count >= 2:
            return True

        return False

    def _should_emit_summary(self, user_prompt: str) -> bool:
        """
        Only generate a task report for real tasks, not casual conversation.
        """
        return self._is_real_task(user_prompt)

    # ---------------------------------------------------------------------
    # Reflection
    # ---------------------------------------------------------------------
    async def _force_reflection(
        self, context_manager, original_prompt: str, iteration: int
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Triggered when tool_calls are empty or stagnation is detected.
        Forces a strategy change or a clean finish.
        """
        ctx = await context_manager.provide_context(
            query="",
            max_input_tokens=self.max_tokens,
            reserved_for_output=1024,
        )

        context_text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')[:400]}"
            for m in ctx
            if m.get("content")
        )[:7000]

        recent_failures = self._recent_failures_text(context_manager)

        prompt = (
            f"Agent has run {iteration} iterations on task:\n{original_prompt}\n\n"
            f"Recent context:\n{context_text}\n\n"
        )
        if recent_failures:
            prompt += f"Known failures / dead ends:\n{recent_failures}\n\n"

        prompt += (
            "You just finished a turn without making meaningful progress.\n"
            "You MUST do exactly one of the following:\n"
            "1. Choose a different tool or a different strategy.\n"
            "2. Finish the task if it is already solved.\n\n"
            "Return a short reflection with these fields:\n"
            "- completed: yes/no\n"
            "- reason: one sentence\n"
            "- next_action: exact next tool or exact finish strategy\n"
            "- avoid: what not to repeat\n"
        )

        try:
            summary = await self.client.ask(prompt)
        except Exception as exc:
            summary = f"Reflection generation failed: {exc}"

        reflection_message = (
            f"SELF-REFLECTION (Iteration {iteration}):\n{summary}\n\n"
            f"Original task: {original_prompt}\n\n"
            "Use this reflection only as a steering hint. "
            "Do not repeat the same failed tool calls or the same reasoning path."
        )

        try:
            await context_manager.add_message("system", reflection_message)
        except Exception:
            await context_manager.add_message("user", reflection_message)

        await context_manager.add_external_content(
            "reflection", summary, {"iteration": iteration, "type": "stuck_detection"}
        )

        self.reflection_count += 1
        if self.reflection_count > self.max_reflections:
            logger.warning("Agent stuck in reflection loop. Forcing exit.")
            yield ("warning", f"Agent stuck after {self.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            return

        yield ("reflection_triggered", summary)

    # ---------------------------------------------------------------------
    # Tool execution
    # ---------------------------------------------------------------------
    async def _execute_tools(
        self, tool_calls, get_exec, context_manager, iteration, state
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Execute tool calls with executor filtering and duplicate protection.

        FIXES:
        - Exploration lock applies to ALL exploratory tools (ls, find, etc.)
        - Browser tools extend exploration window
        - Distinct tool tracking for better reflection decisions
        """

        tool_results = []

        for call in tool_calls or []:
            if self._should_stop():
                yield ("cancelled", "Stop during tool execution")
                return

            if not isinstance(call, dict):
                continue

            name = call.get("function", {}).get("name")
            if not name:
                continue

            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except Exception:
                args = {}

            # ------------------------------------------------------------------
            # Duplicate protection
            # ------------------------------------------------------------------
            sig = self._tool_signature(name, args)
            if sig in self.executed_signatures:
                logger.debug(f"Skipping duplicate tool call: {name}")
                continue
            self.executed_signatures.add(sig)

            # Track distinct tools (important for reflection policy)
            state.setdefault("distinct_tools_tried", set()).add(name)

            executor = get_exec(name)

            if not executor:
                logger.warning(f"Ignoring tool '{name}' — no executor registered")
                yield (
                    "tool_skipped",
                    {"name": name, "args": args, "reason": "no executor"},
                )
                continue

            yield ("tool_start", {"name": name, "args": args})

            # ------------------------------------------------------------------
            # Execute tool
            # ------------------------------------------------------------------
            try:
                maybe_result = executor(**args)
                result = (
                    await maybe_result
                    if asyncio.iscoroutine(maybe_result)
                    else maybe_result
                )

                if self._should_stop():
                    yield ("cancelled", f"Stopped after tool '{name}'")
                    return

                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                logger.info(f"User confirmation needed for tool '{name}'")
                yield ("needs_confirmation", {**exc.__dict__, "tool_calls": tool_calls})
                return

            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                logger.warning(result)
                yield ("tool_result", {"name": name, "result": result, "error": True})

            # ------------------------------------------------------------------
            # Store result
            # ------------------------------------------------------------------
            tool_entry = {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": name,
                "content": str(result),
                "args": args,
            }
            tool_results.append(tool_entry)

            try:
                if hasattr(context_manager, "record_tool_outcome"):
                    await context_manager.record_tool_outcome(
                        name, str(result), {"iteration": iteration, **args}
                    )
                else:
                    await context_manager.add_external_content(
                        source_type="tool",
                        content=str(result),
                        metadata={"tool": name, "args": args, "iteration": iteration},
                    )
            except Exception:
                pass

            try:
                await context_manager.add_message("tool", str(result))
            except Exception:
                pass

            # ------------------------------------------------------------------
            # 🧠 EXPLORATION CONTROL (THIS IS THE FIX)
            # ------------------------------------------------------------------

            # 1. Browser tools → strong exploration phase
            if self._is_tool_browser(name):
                discovered = self._extract_browser_tools(result)

                state["browser_followup_turns"] = 2
                state["tool_exploration_lock"] = 2
                state["phase"] = self.Phase.TOOL_DISCOVERY
                state["no_progress_turns"] = 0

                browser_hint = self._build_browser_followup_message(discovered)

                try:
                    await context_manager.add_message("system", browser_hint)
                except Exception:
                    pass

                try:
                    await context_manager.add_external_content(
                        source_type="system",
                        content=browser_hint,
                        metadata={
                            "iteration": iteration,
                            "type": "tool_browser_followup",
                            "discovered_tools": discovered,
                        },
                    )
                except Exception:
                    pass

            # 2. ANY exploratory tool (ls, find, etc.) → block reflection
            elif self._is_exploratory_tool(name):
                state["tool_exploration_lock"] = max(
                    state.get("tool_exploration_lock", 0),
                    1,
                )
                state["no_progress_turns"] = 0

            # 3. Real execution tools → stay in EXECUTE phase
            else:
                state["phase"] = self.Phase.EXECUTE

            # ------------------------------------------------------------------
            # Stop handling
            # ------------------------------------------------------------------
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                logger.info(f"Stop requested during tool '{name}' execution")
                yield ("cancelled", f"Stop after tool '{name}'")
                return

        # ----------------------------------------------------------------------
        # No tool results case
        # ----------------------------------------------------------------------
        if not tool_results:
            logger.debug("No executable tool calls produced results this iteration")

            try:
                await context_manager.add_message(
                    "system",
                    "All tool calls were duplicates or skipped. "
                    "Try a DIFFERENT tool or strategy (do not repeat).",
                )
            except Exception:
                pass

        self.tool_results.extend(tool_results)

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    async def _generate_final_summary(
        self,
        original_prompt: str,
        messages: List[Dict],
        tools: ToolProvider,
        reason: str = "normal",
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.info("Generating final summary...")

        report_lines = [
            "# 🏁 Agent Execution Report",
            f"**Original Task:** {self._safe_truncate(original_prompt, 200)}",
            f"**Status:** {reason.upper()}",
            "",
            "## 📊 Summary",
            f"- **Iterations:** {len(self.messages) if self.messages else 0}",
            f"- **Tool Calls:** {len(self.tool_results)}",
            f"- **Reflections:** {self.reflection_count}",
            "",
            "## 🛠️ Tool Usage",
        ]

        if self.tool_results:
            report_lines.append("| Tool | Args | Result |")
            report_lines.append("| --- | --- | --- |")
            for r in self.tool_results:
                args_str = self._safe_truncate(str(r.get("args", {})), 60)
                result_str = self._safe_truncate(r.get("content", ""), 80)
                report_lines.append(
                    f"| {r.get('name', 'unknown')} | {args_str} | {result_str} |"
                )
        else:
            report_lines.append("- No tools were executed (pure reasoning task).")

        report_lines.extend(
            [
                "",
                "## 💬 Conversation Flow",
            ]
        )

        for msg in messages[:10]:
            role = msg.get("role", "unknown")
            content = self._safe_truncate(msg.get("content", ""), 200)
            report_lines.append(f"- **{role}**: {content}")

        report_lines.extend(
            [
                "",
                "## 🔍 Final Conclusion",
                "The agent has reached a stable end state or a safe stopping condition.",
            ]
        )

        final_text = "\n".join(report_lines)
        yield ("final_summary", final_text)
        yield ("finish", {"reason": reason, "summary": final_text})
        return

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    async def run(
        self,
        user_prompt: str,
        tools: ToolProvider,
        system_prompt: str = "",
        get_executor: Optional[ToolExecutorGetter] = None,
        context_manager=None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.info(
            "AgentRunner (Improved Design) called | prompt=%s...", user_prompt[:150]
        )

        if context_manager is None:
            raise ValueError("ContextManager is required in the redesigned AgentRunner")

        self._stop_event = stop_event or asyncio.Event()

        if isinstance(tools, (ActionSet, DynamicActionManager)):
            get_tools_func = tools.get_llm_tools
            get_exec = get_executor or tools.get_executor
            has_tool_provider = True
        else:
            get_tools_func = lambda: tools
            get_exec = get_executor or (lambda _: None)
            has_tool_provider = bool(tools)

        self.messages = []
        self.executed_signatures.clear()
        self.reflection_count = 0
        self.tool_results = []

        # add to state
        state = {
            "phase": self.Phase.TOOL_DISCOVERY
            if has_tool_provider
            else self.Phase.CASUAL,
            "last_fingerprint": None,
            "same_fingerprint_count": 0,
            "no_progress_turns": 0,
            "last_tool_result_count": 0,
            "browser_followup_turns": 0,
            # NEW
            "tool_exploration_lock": 0,
            "distinct_tools_tried": set(),
        }

        await context_manager.add_message("user", user_prompt)
        self.messages.append({"role": "user", "content": user_prompt})
        logger.info("Initial user prompt added to ContextManager")

        COMPLETION_RULE = (
            "Internal completion rule: when the task is fully solved and no more actions are needed, "
            "append exactly [FINAL_ANSWER_COMPLETE] at the very end of the response."
        )
        try:
            await context_manager.add_message("system", COMPLETION_RULE)
        except Exception:
            await context_manager.add_message("user", COMPLETION_RULE)

        iteration = 0
        while (
            self.max_iterations is None
            or self.max_iterations < 0
            or iteration < self.max_iterations
        ):
            iteration += 1
            logger.debug("Iteration %d", iteration)
            yield ("step_start", {"iteration": iteration})

            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", "Stop requested")
                return

            base_prompt = system_prompt or (
                "You are a helpful Terminal AI agent with full coding and filesystem support."
            )
            runtime_system_prompt = self._build_runtime_system_prompt(
                base_system_prompt=base_prompt,
                user_prompt=user_prompt,
                iteration=iteration,
                context_manager=context_manager,
                has_tools=has_tool_provider,
                stagnation_count=state["no_progress_turns"],
                reflection_count=self.reflection_count,
            )

            messages = await context_manager.provide_context(
                query="",
                max_input_tokens=self.max_tokens,
                reserved_for_output=2048,
                system_prompt=runtime_system_prompt,
            )

            try:
                queue = await self.client.stream_with_tools(
                    messages=messages,
                    tools=get_tools_func(),
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                    max_tokens=max_tokens
                    if max_tokens is not None
                    else self.max_tokens,
                    tool_choice="auto",
                )
            except Exception as exc:
                yield ("error", f"stream_with_tools failed: {exc}")
                return

            text_buffer = ""
            tool_calls = None

            try:
                async for kind, payload in self.client._drain_queue(queue):
                    if kind == "content":
                        text_buffer += payload
                        yield ("content_delta", payload)
                    elif kind == "tool_delta":
                        yield ("tool_call_delta", payload)
                    elif kind == "finish":
                        tool_calls = payload.get("tool_calls")
                        yield ("llm_finish", payload)
                        break
                    elif kind == "error":
                        yield ("error", payload)
                        return
            except asyncio.CancelledError:
                yield ("cancelled", "Task cancelled by user")
                return

            full_reply = (text_buffer or "").strip()

            if self._should_stop():
                yield ("cancelled", "Stopped after generation")
                return

            COMPLETION_MARKER = "[FINAL_ANSWER_COMPLETE]"
            is_task_complete = COMPLETION_MARKER in full_reply
            if is_task_complete:
                full_reply = full_reply.replace(COMPLETION_MARKER, "").strip()
                logger.info(
                    f"LLM signaled task complete (hidden marker detected) at iteration {iteration}"
                )

            if is_task_complete and tool_calls:
                logger.warning(
                    f"Completion marker present with {len(tool_calls)} tool call(s) – "
                    "ignoring tools per rule and forcing clean completion"
                )
                tool_calls = None

            fingerprint = self._turn_fingerprint(full_reply, tool_calls)
            if fingerprint == state["last_fingerprint"]:
                state["same_fingerprint_count"] += 1
            else:
                state["same_fingerprint_count"] = 0
            state["last_fingerprint"] = fingerprint

            # -----------------------------------------------------------------
            # No-tool path: assistant answered in plain text
            # -----------------------------------------------------------------
            if not tool_calls:
                try:
                    await context_manager.add_external_content(
                        source_type="assistant_output",
                        content=full_reply,
                        metadata={"iteration": iteration},
                    )
                except Exception:
                    pass

                self.messages.append({"role": "assistant", "content": full_reply})

                previous_tool_count = state["last_tool_result_count"]
                current_tool_count = len(self.tool_results)

                if current_tool_count > previous_tool_count:
                    state["no_progress_turns"] = 0
                    state["last_tool_result_count"] = current_tool_count
                else:
                    state["no_progress_turns"] += 1

                casual_mode = (
                    self._is_casual_prompt(user_prompt)
                    and len(self.executed_signatures) == 0
                    and len(self.tool_results) == 0
                    and iteration <= 2
                    and self._looks_like_final_answer(full_reply)
                )

                # Give one follow-up turn after tool browsing before any reflection.
                if state["browser_followup_turns"] > 0:
                    state["browser_followup_turns"] -= 1
                    state["no_progress_turns"] = 0
                    logger.debug(
                        "Browser discovery follow-up active; skipping reflection"
                    )
                    continue

                if is_task_complete or casual_mode:
                    reason = "task_complete" if is_task_complete else "casual_complete"

                    logger.info(
                        f"Clean exit — {reason} "
                        f"(marker={is_task_complete}, casual={casual_mode}, iter={iteration})"
                    )

                    if self._should_emit_summary(user_prompt) and (
                        iteration > 1 or is_task_complete
                    ):
                        async for event, payload in self._generate_final_summary(
                            user_prompt, self.messages, tools, reason
                        ):
                            yield (event, payload)
                    else:
                        yield ("finish", {"reason": reason})

                    return

                if self._should_reflect(
                    iteration=iteration,
                    is_task_complete=is_task_complete,
                    tool_calls=tool_calls,
                    no_progress_turns=state["no_progress_turns"],
                    same_fingerprint_count=state["same_fingerprint_count"],
                    reflection_count=self.reflection_count,
                    casual_mode=casual_mode,
                    browser_followup_turns=state["browser_followup_turns"],
                    tool_exploration_lock=state.get("tool_exploration_lock", 0),
                    distinct_tools_tried=len(state.get("distinct_tools_tried", [])),
                ):
                    state["phase"] = self.Phase.REFLECT
                    logger.debug("Stagnation detected → reflection")
                    async for event, payload in self._force_reflection(
                        context_manager,
                        original_prompt=user_prompt,
                        iteration=iteration,
                    ):
                        yield (event, payload)
                        if event == "finish":
                            return
                    state["phase"] = (
                        self.Phase.TOOL_DISCOVERY
                        if has_tool_provider
                        else self.Phase.CASUAL
                    )

                    if self.reflection_count >= self.max_reflections:
                        if self._should_emit_summary(user_prompt):
                            async for event, payload in self._generate_final_summary(
                                user_prompt, self.messages, tools, "reflection_stuck"
                            ):
                                yield (event, payload)
                        else:
                            yield ("finish", {"reason": "casual_complete"})
                        return

                continue

            # -----------------------------------------------------------------
            # Tool-calling path
            # -----------------------------------------------------------------
            state["phase"] = self.Phase.EXECUTE

            try:
                assistant_text = f"Executed {len(tool_calls)} tool call(s)"
                await context_manager.add_external_content(
                    source_type="assistant_output",
                    content=assistant_text,
                    metadata={"iteration": iteration},
                )
            except Exception:
                pass

            yield ("tool_calls", tool_calls)

            before_tool_results = len(self.tool_results)

            async for event, payload in self._execute_tools(
                tool_calls, get_exec, context_manager, iteration, state
            ):
                yield (event, payload)
                if event in ("needs_confirmation", "cancelled"):
                    return

            after_tool_results = len(self.tool_results)

            if after_tool_results > before_tool_results:
                state["no_progress_turns"] = 0
                state["last_tool_result_count"] = after_tool_results
            else:
                state["no_progress_turns"] += 1

            state["phase"] = (
                self.Phase.TOOL_DISCOVERY if has_tool_provider else self.Phase.CASUAL
            )

            if (
                state["no_progress_turns"] >= 2
                and self.reflection_count < self.max_reflections
            ):
                try:
                    await context_manager.add_message(
                        "system",
                        "No progress detected from recent tool calls. "
                        "On the next turn, change strategy or finish the task if possible.",
                    )
                except Exception:
                    pass

        logger.info("Max iterations reached.")
        yield ("warning", f"Max iterations ({self.max_iterations}) reached")

        if self._should_emit_summary(user_prompt):
            async for event, payload in self._generate_final_summary(
                user_prompt, self.messages, tools, "max_iterations_reached"
            ):
                yield (event, payload)
        else:
            yield ("finish", {"reason": "casual_max_iterations"})
