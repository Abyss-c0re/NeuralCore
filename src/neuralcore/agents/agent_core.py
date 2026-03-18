import asyncio
import json
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
    Redesigned AgentRunner with robust self-reflection and final summary.
    - If tool calls are missing, triggers a "Review & Reflect" phase.
    - Tracks tool execution to avoid infinite loops.
    - Always generates a final summary report at the end of execution.
    """

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

    async def _force_reflection(
        self, context_manager, original_prompt: str, iteration: int
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Triggered when tool_calls are empty or stuck detection is active.
        Asks LLM to evaluate progress and propose a concrete next step.
        """
        # 1. Get current context (strict limit)
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

        # 2. Ask LLM for reflection
        summary = await self.client.ask(
            f"Agent has run {iteration} iterations on task: {original_prompt}\n\n"
            f"Recent context:\n{context_text}\n\n"
            "You just finished a turn without calling any tools (or tools failed).\n"
            "Evaluate the current state:\n"
            "1. Has the original task been effectively completed?\n"
            "2. Did the previous tools provide enough data to solve the task?\n"
            "3. If not, what is the EXACT next tool or action you should take?"
        )

        # 3. Add Reflection Message
        new_query = (
            f"SELF-REFLECTION (Iteration {iteration}):\n{summary}\n\n"
            f"Original task: {original_prompt}\n\n"
            "Continue the task. Do not repeat the exact same failed tool calls. "
            "Use different tools or a completely new approach if stuck."
        )

        await context_manager.add_message("user", new_query)
        await context_manager.add_external_content(
            "reflection", summary, {"iteration": iteration, "type": "stuck_detection"}
        )

        # 4. Update internal state
        self.reflection_count += 1
        if self.reflection_count > self.max_reflections:
            logger.warning("Agent stuck in reflection loop. Forcing exit.")
            yield ("warning", f"Agent stuck after {self.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            return

        yield ("reflection_triggered", summary)

    async def _execute_tools(self, tool_calls, get_exec, context_manager, iteration):
        """Execute tool calls, track results, and update context."""
        tool_results = []
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}

            sig = (name, json.dumps(args, sort_keys=True))
            if sig in self.executed_signatures:
                continue
            self.executed_signatures.add(sig)

            yield ("tool_start", {"name": name, "args": args})

            executor = get_exec(name)
            if not executor:
                result = f"Unknown tool: {name}"
                yield (
                    "tool_result",
                    {"name": name, "result": result, "error": True},
                )
            else:
                try:
                    maybe_result = executor(**args)
                    result = (
                        await maybe_result
                        if asyncio.iscoroutine(maybe_result)
                        else maybe_result
                    )
                    yield ("tool_result", {"name": name, "result": result})
                except ConfirmationRequired as exc:
                    yield (
                        "needs_confirmation",
                        {**exc.__dict__, "tool_calls": tool_calls},
                    )
                    return
                except Exception as exc:
                    result = f"Tool failed: {exc}"
                    yield (
                        "tool_result",
                        {"name": name, "result": result, "error": True},
                    )

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": str(result),
                    }
                )

                # Add to context
                await context_manager.add_external_content(
                    source_type="tool",
                    content=str(result),
                    metadata={"tool": name, "args": args, "iteration": iteration},
                )
                await context_manager.add_message("tool", str(result))

                # Cancellation check
                stop_event = getattr(self.client, "_current_stop_event", None)
                if stop_event and getattr(stop_event, "is_set", lambda: False)():
                    yield ("cancelled", f"Stop after tool {name}")
                    return

        if not tool_results:
            yield (
                "system",
                "Previous tool calls were duplicates. Try a different approach.",
            )
        self.tool_results.extend(tool_results)

    async def run(
        self,
        user_prompt: str,
        messages_so_far: List[Dict[str, Any]],
        tools: ToolProvider,
        system_prompt: str = "",
        get_executor: Optional[ToolExecutorGetter] = None,
        context_manager=None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.info(
            "AgentRunner (Improved Design) called | prompt=%s...", user_prompt[:150]
        )

        if context_manager is None:
            raise ValueError("ContextManager is required in the redesigned AgentRunner")

        # ── Tool setup ───────────────────────────────────────
        if isinstance(tools, (ActionSet, DynamicActionManager)):
            get_tools_func = tools.get_llm_tools
            get_exec = get_executor or tools.get_executor
        else:
            get_tools_func = lambda: tools
            get_exec = get_executor or (lambda _: None)

        # ── Initialize State ─────────────────────────────────
        self.messages = []
        self.executed_signatures.clear()
        self.reflection_count = 0
        self.tool_results = []

        # ── Add initial user message to ContextManager ─────────────
        await context_manager.add_message("user", user_prompt)
        logger.info("Initial user prompt added to ContextManager")

        # ── Main Loop ─────────────────────────────────────────
        iteration = 0
        while (
            self.max_iterations is None
            or self.max_iterations < 0
            or iteration < self.max_iterations
        ):
            iteration += 1
            logger.debug("Iteration %d", iteration)
            yield ("step_start", {"iteration": iteration})

            # Cancellation guard
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", "Stop requested")
                return

            # ── Fresh Context every turn ─────────────────────────────
            messages = await context_manager.provide_context(
                query="",
                max_input_tokens=self.max_tokens,
                reserved_for_output=2048,
                system_prompt=system_prompt
                or "You are a helpful Terminal AI agent with full coding and filesystem support.",
            )

            # ── LLM Call ─────────────────────────────────────────────
            queue = await self.client.stream_with_tools(
                messages=messages,
                tools=get_tools_func(),
                temperature=temperature
                if temperature is not None
                else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                tool_choice="auto",
            )

            text_buffer = ""
            tool_calls = None

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

            full_reply = text_buffer.strip()

            # ── No Tool Calls = Review & Reflection Path ────────────
            if not tool_calls:
                await context_manager.add_message("assistant", full_reply)

                if iteration == 1:
                    logger.debug(
                        f"Turn {iteration} completed with no tool calls. Analyzing..."
                    )

                    payload = None

                    # Call reflection and consume its events
                    async for event, payload in self._force_reflection(
                        context_manager,
                        original_prompt=user_prompt,
                        iteration=iteration,
                    ):
                        yield (event, payload)

                    # After loop, payload should be bound from last yield
                    # If no events were yielded, payload might be None
                    if payload is None:
                        payload = {"reason": "normal"}

                    # If reflection yielded "finish", we stop here
                    # Otherwise, we continue the loop

                    if isinstance(payload, dict):
                        if payload.get("reason") == "reflection_stuck":
                            break  # Break loop to trigger final summary
                        # If it's a normal finish, continue with the summary
                        elif (
                            payload.get("reason") == "normal" or "reason" not in payload
                        ):
                            # Normal completion - trigger final summary after loop
                            continue
                        else:
                            yield ("review_phase", payload)
                            continue  # Continue loop to allow next action or final confirmation

                    # If payload was a string (reflection_triggered), continue loop
                    elif isinstance(payload, str):
                        yield (
                            "review_phase",
                            {"summary": payload, "type": "reflection_triggered"},
                        )
                        continue  # Continue loop to allow next action or final confirmation

                    # If we get here, payload is something else unexpected
                    else:
                        yield (
                            "review_phase",
                            {"summary": "Reflection complete", "type": "unknown"},
                        )
                        continue

            # ── Tool Calls Path ──────────────────────────────────────────────
            # Safely handle tool_calls being None
            assistant_text = full_reply or ""
            if tool_calls:
                assistant_text = f"Executed {len(tool_calls)} tool call(s)"
                await context_manager.add_message("assistant", assistant_text)

                yield ("tool_calls", tool_calls)

                tool_results = []

                # Only iterate if tool_calls is not None and not empty
                if tool_calls:
                    for call in tool_calls:
                        name = call["function"]["name"]
                        try:
                            args = json.loads(call["function"]["arguments"])
                        except Exception:
                            args = {}

                        sig = (name, json.dumps(args, sort_keys=True))
                        if sig in self.executed_signatures:
                            continue
                        self.executed_signatures.add(sig)

                        yield ("tool_start", {"name": name, "args": args})

                        executor = get_exec(name)
                        if not executor:
                            result = f"Unknown tool: {name}"
                            yield (
                                "tool_result",
                                {"name": name, "result": result, "error": True},
                            )
                        else:
                            try:
                                maybe_result = executor(**args)
                                result = (
                                    await maybe_result
                                    if asyncio.iscoroutine(maybe_result)
                                    else maybe_result
                                )
                                yield ("tool_result", {"name": name, "result": result})
                            except ConfirmationRequired as exc:
                                yield (
                                    "needs_confirmation",
                                    {**exc.__dict__, "tool_calls": tool_calls},
                                )
                                return  # Clean return for confirmation
                            except Exception as exc:
                                result = f"Tool failed: {exc}"
                                yield (
                                    "tool_result",
                                    {"name": name, "result": result, "error": True},
                                )

                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "name": name,
                                    "content": str(result),
                                }
                            )

                            await context_manager.add_external_content(
                                source_type="tool",
                                content=str(result),
                                metadata={
                                    "tool": name,
                                    "args": args,
                                    "iteration": iteration,
                                },
                            )
                            await context_manager.add_message("tool", str(result))

                    if not tool_results:
                        await context_manager.add_message(
                            "system",
                            "Previous tool calls were duplicates. Please try a different approach.",
                        )

                    # Check cancellation after tool execution
                    stop_event = getattr(self.client, "_current_stop_event", None)
                    if stop_event and getattr(stop_event, "is_set", lambda: False)():
                        yield (
                            "cancelled",
                            f"Stop after tool {tool_calls[0]['function']['name'] if tool_calls else 'unknown'}",
                        )
                        return

        # ── Max Iterations Reached ─────────────────────────────────────────
        logger.info("Max iterations reached, generating final summary...")
        yield ("warning", f"Max iterations ({self.max_iterations}) reached")

        if iteration > 1:

            # Generate final summary before exiting
            async for event, payload in self._generate_final_summary(
                user_prompt, self.messages, tools, "max_iterations_reached"
            ):
                yield (event, payload)

    async def _generate_final_summary(
        self,
        original_prompt: str,
        messages: List[Dict],
        tools: ToolProvider,
        reason: str = "normal",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Generates a final Markdown report at the end of the run.
        """
        logger.info("Generating final summary...")

        # 1. Build Report
        report_lines = [
            "# 🏁 Agent Execution Report",
            f"**Original Task:** {original_prompt[:200]}...",
            f"**Status:** {reason.upper()}",
            "",
            "## 📊 Summary",
            f"- **Iterations:** {len(messages)}",
            f"- **Tool Calls:** {len(self.tool_results)}",
            f"- **Reflections:** {self.reflection_count}",
            "",
            "## 🛠️ Tool Usage",
        ]

        if self.tool_results:
            report_lines.append("| Tool | Args | Result |")
            report_lines.append("| --- | --- | --- |")
            for r in self.tool_results:
                result_str = (
                    r.get("content", "")[:50] + "..."
                    if len(r.get("content", "")) > 50
                    else r.get("content", "")
                )
                report_lines.append(
                    f"| {r.get('name')} | {r.get('args', {})[:50]} | {result_str} |"
                )
        else:
            report_lines.append("- No tools were executed (Pure reasoning task).")

        report_lines.extend(
            [
                "",
                "## 💬 Conversation Flow",
            ]
        )

        # Truncate messages for readability
        for msg in messages[:10]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            report_lines.append(f"- **{role}**: {content}...")

        report_lines.extend(
            [
                "",
                "## 🔍 Final Conclusion",
                "Based on the conversation and tool execution above, the agent has reached a stable state.",
            ]
        )

        final_text = "\n".join(report_lines)
        yield ("final_summary", final_text)
        yield ("finish", {"reason": reason, "summary": final_text})
        return  # Clean return to close generator
