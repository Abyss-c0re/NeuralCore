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
    Reusable agent loop orchestrator that uses an existing LLMClient
    without modifying it.

    Yields the same event tuples as the old LLMClient.agent_stream method.
    """

    def __init__(
        self,
        client: "LLMClient",
        max_iterations: int = 25,
        temperature: float = 0.3,
        max_tokens: int = 12048,
    ):
        self.client = client
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        logger.info("Agent Runner Called | prompt=%s", user_prompt[:200])

        # Tool accessor
        if isinstance(tools, (ActionSet, DynamicActionManager)):
            get_tools_func = tools.get_llm_tools
            get_exec = get_executor or tools.get_executor
            logger.debug("Using ActionSet/DynamicActionManager for tools")
        else:
            get_tools_func = lambda: tools
            get_exec = get_executor or (lambda name: None)
            logger.debug("Using raw tool list")

        # Prepare messages
        messages = messages_so_far.copy()
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
            logger.debug("Inserted system prompt")

        if context_manager:
            try:
                ctx_history = await context_manager.generate_prompt(user_prompt, num_messages=0)
                logger.debug("Context history length=%d", len(ctx_history) if ctx_history else 0)
                if ctx_history and "\nUser query:" in ctx_history[-1]["content"]:
                    ctx_part = ctx_history[-1]["content"].split("\nUser query:")[0].strip()
                    if ctx_part:
                        messages.insert(1, {"role": "system", "content": f"Relevant context:\n{ctx_part}"})
                        logger.info("Relevant context added to messages")
            except Exception as e:
                logger.warning("Context enrichment failed: %s", e)

        if not any(m.get("role") == "user" for m in messages):
            messages.append({"role": "user", "content": user_prompt})
            logger.debug("Appended user message")

        executed_signatures: set[tuple] = set()
        iteration = 0
        logger.info(
            "AgentRunner starting loop | max_iterations=%s | temperature=%s | messages=%d",
            self.max_iterations,
            temperature or self.temperature,
            len(messages),
        )

        while self.max_iterations is None or self.max_iterations < 0 or iteration < self.max_iterations:
            iteration += 1
            logger.debug("Iteration %d start", iteration)
            yield ("step_start", {"iteration": iteration})

            # Safe cancellation check
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and callable(getattr(stop_event, "is_set", None)) and stop_event.is_set():
                logger.warning("Cancelled before LLM call")
                yield ("cancelled", "Stop requested before LLM call")
                return

            # Core LLM call
            logger.info("Calling client.stream_with_tools | iteration=%d", iteration)
            queue = await self.client.stream_with_tools(
                messages=messages,
                tools=get_tools_func(),
                temperature = temperature if temperature is not None else self.temperature,
                max_tokens = max_tokens if max_tokens is not None else self.max_tokens,
                tool_choice="auto",
            )
            logger.debug("Queue received | type=%s", type(queue))

            text_buffer = ""
            tool_calls = None

            logger.debug("Draining LLM queue...")
            async for kind, payload in self.client._drain_queue(queue):
                logger.debug("Drained kind=%s | payload_preview=%s", kind, str(payload)[:200])

                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)
                elif kind == "tool_delta":
                    yield ("tool_call_delta", payload)
                elif kind == "finish":
                    tool_calls = payload.get("tool_calls")
                    yield ("llm_finish", payload)
                    logger.debug("LLM finish received")
                    break
                elif kind == "error":
                    logger.error("Error event received: %s", payload)
                    yield ("error", payload)
                    return

            # Safe cancellation after LLM
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and callable(getattr(stop_event, "is_set", None)) and stop_event.is_set():
                logger.warning("Cancelled after LLM response")
                yield ("cancelled", "Stop requested after LLM response")
                return

            full_reply = text_buffer.strip()
            logger.debug("Full reply length=%d", len(full_reply))

            # Final answer path
            if not tool_calls:
                if messages and messages[-1].get("role") == "assistant":
                    messages[-1]["content"] = full_reply
                    logger.debug("Updated last assistant message")
                else:
                    yield ("assistant_message", {"role": "assistant", "content": full_reply})

                yield ("final_answer", full_reply)
                logger.info("AgentRunner finished normally after %d iterations", iteration)
                return

            # Tool path
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            yield ("tool_calls", tool_calls)
            logger.debug("Tool calls yielded | count=%d", len(tool_calls))

            tool_results = []

            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except Exception:
                    args = {}
                    logger.warning("Could not parse args for tool %s", name)

                sig = (name, json.dumps(args, sort_keys=True))
                if sig in executed_signatures:
                    logger.info("Skipping duplicate tool call: %s %s", name, args)
                    continue
                executed_signatures.add(sig)

                yield ("tool_start", {"name": name, "args": args})
                logger.debug("Tool start yielded: %s", name)

                executor = get_exec(name)
                if not executor:
                    result = f"Unknown tool: {name}"
                    yield ("tool_result", {"name": name, "result": result, "error": True})
                    logger.warning("Unknown tool executed: %s", name)
                else:
                    try:
                        maybe_result = executor(**args)
                        result = await maybe_result if asyncio.iscoroutine(maybe_result) else maybe_result
                        yield ("tool_result", {"name": name, "result": result})
                        logger.debug("Tool result yielded: %s", name)
                    except ConfirmationRequired as exc:
                        yield (
                            "needs_confirmation",
                            {
                                "tool_call_id": call["id"],
                                "name": name,
                                "args": args,
                                "preview": getattr(exc, "preview", ""),
                                "action": executor,
                                "tool_calls": tool_calls,
                            },
                        )
                        logger.info("Tool requires confirmation: %s", name)
                        return
                    except Exception as exc:
                        result = f"Tool execution failed: {exc}"
                        logger.error("Tool %s failed", name, exc_info=True)
                        yield ("tool_result", {"name": name, "result": result, "error": True})

                # Safe cancellation after tool
                stop_event = getattr(self.client, "_current_stop_event", None)
                if stop_event and callable(getattr(stop_event, "is_set", None)) and stop_event.is_set():
                    logger.warning("Cancelled after tool %s", name)
                    yield ("cancelled", f"Stop requested after tool {name}")
                    return

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": str(result) if not isinstance(result, str) else result,
                })

            if tool_results:
                messages.extend(tool_results)
            else:
                yield ("warning", "No tool results produced in this iteration")
                logger.warning("No tool results produced in iteration %d", iteration)
                break

        yield ("warning", f"Max iterations ({self.max_iterations}) reached")
        yield ("finish", {"reason": "max_iterations_reached"})
        logger.info("AgentRunner run finished | max_iterations reached")


# ── Optional convenience wrapper (for simple usage) 
# 
# # Usage Headless / script / test
# result = await run_agent_once(
#     client=llm_client,
#     user_prompt=query,
#     messages=[],
#     tools=my_tools,
# )
# print(result)─────────────────────────────

async def run_agent_once(
    client: "LLMClient",
    user_prompt: str,
    messages: List[Dict],
    tools: ToolProvider,
    **runner_kwargs,
) -> str:
    """Simple collector — returns final text or error message"""
    runner = AgentRunner(client, **runner_kwargs)
    full_text = ""
    async for event_type, payload in runner.run(
        user_prompt=user_prompt,
        messages_so_far=messages,
        tools=tools,
        **runner_kwargs,
    ):
        if event_type == "content_delta":
            full_text += payload
        elif event_type == "final_answer":
            full_text = payload
        elif event_type in ("error", "cancelled"):
            full_text += f"\n\n[{event_type.upper()}] {payload or 'no details'}"
    return full_text.strip()