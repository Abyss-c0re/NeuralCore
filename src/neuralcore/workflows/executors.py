import asyncio
import json

from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class AgentExecutors:
    """Handles all LLM streaming, tool execution, chat loops, and sub-agent execution.

    FULL ContextManager usage + user feedback applied:
      • Tool outcomes are recorded via record_tool_outcome() → ONLY external context / KB.
      • During tool-looping turns (chat_loop): NO assistant or tool messages are added to history.
        Only final non-tool responses are committed via add_message("assistant").
        This is the perfect balance to prevent hallucinations and context bloat.
      • Pure chat (no tools): history is fully maintained as before.
      • No duplicate content: record_tool_outcome already skips identical KB entries via hash.
      • "Why the summary ('castrate')?": record_tool_outcome intentionally uses a trimmed result
        (original design) to keep KB lean and avoid flooding retrieval. Full raw output
        lives transiently in the streaming buffer + TaskContext (sub-agents). If you want
        fuller KB entries, increase the slice in ContextManager.record_tool_outcome.
    """

    def __init__(self, agent, phase_enum):
        self.agent = agent
        self.Phase = phase_enum
        self.engine = agent.workflow

    async def _execute_tool(self, name: str, args: dict):
        executor = self.agent.manager.get_executor(name, self.agent)
        if not executor:
            raise RuntimeError(f"No executor found for tool '{name}'")
        result = executor(**args)
        return await result if asyncio.iscoroutine(result) else result

    # ====================== CORE AGENTIC LOOP (FULL CM) ======================

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
        self.agent.context_manager.set_goal(self.agent.goal or task_name)
        self.agent.context_manager.add_subtask(task_name)

        max_loops = 15
        loop_count = 0
        final_answer_marker = "[FINAL_ANSWER_COMPLETE]"

        while loop_count < max_loops:
            loop_count += 1
            tool_browser_detected = False
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

                            if "BrowseTools" in tool_name:
                                tool_browser_detected = True
                                logger.info(
                                    f"[BrowseTools] detected. Restarting agentic loop {loop_count}"
                                )
                                break

                            content_str = (
                                json.dumps(result, ensure_ascii=False, default=str)
                                if isinstance(result, dict)
                                else str(result or "No output")
                            )

                            # ONLY external context + heuristics (no history pollution)
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
                continue

            final_reply = text_buffer.strip()
            if final_answer_marker in final_reply:
                final_reply = final_reply.replace(final_answer_marker, "").strip()

            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)

            if not final_reply:
                final_reply = "✅ Task completed."

            # FINAL assistant message only (no intermediates)
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

    # ====================== CHAT LOOP (FULL CM + CLEAN TOOL LOOPING) ======================


# ====================== CHAT LOOP (UPDATED) ======================


    async def chat_loop(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        logger.debug(
            "=== ENTERING chat_loop (FULL ContextManager + clean tool looping) ==="
        )

        max_tool_loops = 10
        loop_count = 0
        browse_restart_pending = False  # ← NEW FLAG

        while loop_count < max_tool_loops:
            loop_count += 1
            tool_browser_detected = False
            complex_action_called = False
            complex_reason = ""
            text_buffer = ""
            tool_results: List[str] = []

            current_query = ""
            if loop_count == 1 or browse_restart_pending:
                # Use the original user query on first turn OR after BrowseTools restart
                for m in reversed(messages or []):
                    if m.get("role") == "user" and m.get("content"):
                        current_query = m.get("content", "").strip()
                        break
                if not current_query:
                    current_query = state.current_task or "[NO USER QUERY]"
                browse_restart_pending = False  # reset flag
            else:
                current_query = "[CONTINUATION AFTER TOOL CALLS]"

            cm_messages = await self.agent.context_manager.provide_context(
                query=current_query,
                max_input_tokens=self.agent.max_tokens,
                reserved_for_output=12000,
                system_prompt=self._build_objective_reminder(),
                include_logs=True,
            )

            queue = await self.agent.client.stream_with_tools(
                messages=cm_messages,
                tools=self.agent.manager.get_action_set("DynamicCore"),
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
                        yield ("content_delta", content)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = (
                                payload.get("tool_name") or payload.get("name") or "unknown"
                            )
                            result = payload.get("result") or payload.get("output")

                            if "BrowseTools" in tool_name:
                                tool_browser_detected = True
                                logger.info(
                                    "[BrowseTools] detected → will restart LLM once with new tools"
                                )
                                break  # stop this stream immediately

                            # ... rest of tool handling unchanged ...

                            if tool_name == "RequestComplexAction":
                                complex_action_called = True
                                complex_reason = str(
                                    payload.get("args", {}).get("reason", "")
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
                                metadata={"loop": loop_count, "chat_mode": True},
                            )

                            if content_str.strip():
                                tool_results.append(content_str.strip())

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except Exception as e:
                logger.error(f"Stream error in chat_loop: {e}", exc_info=True)
                yield ("error", str(e))
                return

            # ====================== RESTART LOGIC ======================
            if tool_browser_detected:
                browse_restart_pending = True  # ← signal to reuse original query next turn
                continue  # go to next iteration of while loop (restart LLM)

            if complex_action_called:
                # ... your existing complex action code ...
                return

            # ── FINAL REPLY LOGIC (unchanged) ─────────────────────
            final_reply = text_buffer.strip()
            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)

            if not final_reply:
                final_reply = "✅ Done."

            tools_called_this_turn = len(tool_results) > 0

            if not tools_called_this_turn and final_reply:
                await self.agent.context_manager.add_message("assistant", final_reply)
                yield (
                    "llm_response",
                    {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
                )
                break

            elif tools_called_this_turn and loop_count < max_tool_loops:
                logger.info(
                    f"Tools executed (loop {loop_count}) – continuing for final answer"
                )
                continue

            else:
                if final_reply:
                    await self.agent.context_manager.add_message("assistant", final_reply)
                yield ("llm_response", {"full_reply": final_reply, "is_complete": True})
                break

        if loop_count >= max_tool_loops:
            logger.warning("Max tool loops reached in chat_loop")

    # ====================== HELPERS ======================

    def _build_objective_reminder(self) -> str:
        return f"Current goal: {self.agent.goal or 'No goal set'}"
