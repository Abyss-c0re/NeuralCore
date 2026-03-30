import asyncio
import json

from typing import AsyncIterator, Dict, Any, List, Optional, Tuple

from neuralcore.agents.state import AgentState
from neuralcore.actions.actions import ActionSet

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class AgentExecutors:
    """Handles all LLM streaming, tool execution, chat loops, and sub-agent execution."""

    def __init__(self, agent, phase_enum):
        """
        Args:
            agent: The main agent instance
            phase_enum: The Phase enum from AgentFlow (passed explicitly)
        """
        self.agent = agent
        self.Phase = phase_enum  # Receive Phase enum from AgentFlow
        self.engine = agent.workflow  # For convenience

    # ====================== EXECUTOR CALLBACK ======================

    async def _execute_tool(self, name: str, args: dict):
        """Unified tool executor callback."""
        executor = self.agent.manager.get_executor(name, self.agent)
        if not executor:
            raise RuntimeError(f"No executor found for tool '{name}'")

        result = executor(**args)
        return await result if asyncio.iscoroutine(result) else result

    # ====================== CORE AGENTIC LOOP ======================

    async def agentic_loop(
        self,
        iteration: int,
        state: AgentState,
        tools: Optional[ActionSet] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Main execution loop for sub-agents / complex tasks."""

        state.phase = self.Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        messages = await self.agent.context_manager.provide_context(
            query=state.current_task or "Continue",
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=12000,
            system_prompt=self._build_objective_reminder(),
            include_logs=True,
        )

        while True:  # Allow restart for ToolBrowser
            tool_browser_detected = False
            text_buffer = ""
            tool_results = []

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
                        text_buffer += payload
                        yield ("content_delta", payload)

                    elif kind in ("tool_delta", "tool_complete", "needs_confirmation"):
                        if isinstance(payload, dict):
                            tool_name = payload.get("tool_name") or payload.get("name")
                            result = payload.get("result") or payload.get("output")

                            if tool_name and "BrowseTools" in tool_name:
                                tool_browser_detected = True
                                logger.info(
                                    f"ToolBrowser detected ({tool_name}). Restarting stream."
                                )
                                break

                            if result:
                                try:
                                    title = f"{tool_name} result"
                                    summary = (
                                        result.get("summary")
                                        or result.get("message")
                                        or (
                                            str(result)
                                            if isinstance(result, dict)
                                            else str(result)
                                        )
                                    )
                                    await (
                                        self.agent.context_manager.add_external_content(
                                            source_type=f"task_result_{tool_name}",
                                            content=f"[{title}] {summary}",
                                            metadata={"task": tool_name},
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to store external content: {e}"
                                    )

                            if isinstance(result, str) and result.strip():
                                tool_results.append(result.strip())

                    elif kind == "finish":
                        break
                    elif kind in ("error", "cancelled"):
                        yield (kind, payload)
                        return

            except asyncio.CancelledError:
                yield ("cancelled", "Task cancelled")
                return
            except Exception as e:
                logger.error(f"Stream error in agentic_loop: {e}", exc_info=True)
                yield ("error", str(e))
                return

            if tool_browser_detected:
                continue

            # Normal completion
            final_reply = text_buffer.strip()
            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)
            if not final_reply:
                final_reply = "✅ Tool executed successfully."

            yield (
                "llm_response",
                {
                    "full_reply": final_reply,
                    "tool_calls": [],
                    "is_complete": True,
                },
            )

            await self.agent.context_manager.add_message("assistant", final_reply)

            # Sub-agent → parent context propagation
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
                        logger.warning(f"Failed to propagate to parent: {e}")

            break

    # ====================== CHAT LOOP ======================

    async def chat_loop(
        self, messages: List[Dict], state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Chat loop with tool support and complex action detection."""

        logger.debug("=== ENTERING chat_loop ===")

        max_tool_loops = 10
        loop_count = 0

        while loop_count < max_tool_loops:
            loop_count += 1
            tool_browser_detected = False
            complex_action_called = False
            complex_reason = ""
            text_buffer = ""
            tool_results: List[str] = []
            assistant_message = ""

            messages = [m for m in messages if m.get("content") is not None]

            queue = await self.agent.client.stream_with_tools(
                messages=messages,
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
                            tool_call_id = payload.get("tool_call_id")

                            if "BrowseTools" in tool_name:
                                tool_browser_detected = True
                                logger.info("[BrowseTools] Restarting stream")
                                break

                            if tool_name == "RequestComplexAction":
                                complex_action_called = True
                                complex_reason = str(
                                    payload.get("args", {}).get("reason", "")
                                )
                                break

                            if assistant_message.strip():
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": assistant_message.strip(),
                                    }
                                )

                            content_str = (
                                "Tool returned no output."
                                if result is None
                                else json.dumps(result, ensure_ascii=False, default=str)
                                if isinstance(result, dict)
                                else str(result)
                            )

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id
                                    or f"call_{len(messages)}",
                                    "name": tool_name,
                                    "content": content_str,
                                }
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

            if tool_browser_detected:
                continue

            if complex_action_called:
                self.agent.task = complex_reason
                self.agent.goal = complex_reason
                final_reply = f"✅ Starting multi-step orchestration for: **{complex_reason[:120]}**..."
                yield ("llm_response", {"full_reply": final_reply, "is_complete": True})
                await self.agent.context_manager.add_message("assistant", final_reply)
                await self.agent.post_control(
                    {"event": "switch_workflow", "name": "default"}
                )
                return

            final_reply = text_buffer.strip()
            if not final_reply and tool_results:
                final_reply = "\n\n".join(tool_results)
            if not final_reply:
                final_reply = "✅ Done."

            if (
                bool(tool_results)
                and not text_buffer.strip()
                and loop_count < max_tool_loops
            ):
                logger.info(f"Looping again for synthesis (iteration {loop_count})")
                continue

            if final_reply:
                await self.agent.context_manager.add_message("assistant", final_reply)

            yield (
                "llm_response",
                {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
            )
            break

        if loop_count >= max_tool_loops:
            logger.warning("Max tool loops reached in chat_loop")

    # ====================== HELPERS ======================

    def _build_objective_reminder(self) -> str:
        """Now lives inside AgentExecutors and no longer depends on AgentFlow."""
        return f"Current goal: {self.agent.goal or 'No goal set'}"
