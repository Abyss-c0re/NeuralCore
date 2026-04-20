from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Awaitable
from inspect import signature

logger = Logger.get_logger()


class Action:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        executor: Callable,
        required: Optional[List[str]] = None,
        action_type: str = "tool",
        strict: bool = False,
        require_confirmation: bool = False,
        confirmation_preview: Optional[Callable[[dict], str]] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        # ==================== NEW: PER-AGENT + SUB-AGENT HIDING ====================
        hidden_for_agents: Optional[List[str]] = None,  # specific agent_ids
        hidden_for_subagents: bool = False,  # optional sub-agent flag
        # =========================================================================
    ):
        if action_type not in {"tool", "function"}:
            raise ValueError("action_type must be 'tool' or 'function'")

        self.name = name
        self.description = description
        self.executor = executor
        self.type = action_type
        self.strict = strict
        self.tags = tags or []
        self.usage_count = 0
        self.aliases = aliases or []

        self.require_confirmation = require_confirmation
        self.confirmation_preview = confirmation_preview or (
            lambda kwargs: f"Executing {name} with {kwargs}"
        )

        self.hidden_for_agents = (
            [str(a).strip() for a in hidden_for_agents] if hidden_for_agents else []
        )
        self.hidden_for_subagents = hidden_for_subagents

        self._bound_agent = None  # Only stores validated agent instances

        # Build schema (exclude self/agent from LLM parameters)
        props = {k: v for k, v in parameters.items() if k not in ("agent", "self")}
        params_schema: Dict[str, Any] = {"type": "object", "properties": props}

        if required:
            filtered_required = [r for r in required if r not in ("agent", "self")]
            if filtered_required:
                params_schema["required"] = filtered_required

        if strict:
            params_schema["additionalProperties"] = False

        self._raw_schema = {
            "name": name,
            "description": description,
            "parameters": params_schema,
        }

        self._search_text = " ".join(
            [self.name, self.description, " ".join(self.tags)]
        ).lower()

        # Detect if first parameter requires binding
        sig = signature(executor)
        params = list(sig.parameters.values())
        self._first_param_name = params[0].name if params else None
        self._needs_agent = self._first_param_name in ("self", "agent")

    # ====================== BINDING ======================

    def bind_agent(self, agent: Any) -> "Action":
        """Bind agent instance with proper validation based on parameter name."""
        if agent is None:
            raise ValueError(f"Cannot bind None as agent to action '{self.name}'")

        if self._first_param_name == "self":
            # Strict validation: when declared as 'self', it MUST be a real Agent
            if not self._is_valid_agent(agent):
                raise TypeError(
                    f"Action '{self.name}' is defined with 'self' as first parameter. "
                    f"It must be bound to an Agent instance (has .agent_id), "
                    f"but got {type(agent).__name__} instead.\n"
                    f"→ Use parameter name 'agent' instead of 'self' if binding non-Agent classes."
                )

        # For parameter name 'agent', we are more permissive (but still require agent_id)
        elif self._first_param_name == "agent":
            if not self._is_valid_agent(agent):
                logger.warning(
                    f"Action '{self.name}' bound to object without .agent_id "
                    f"(type: {type(agent).__name__}). This may cause issues."
                )

        self._bound_agent = agent
        logger.debug(
            f"[ACTION BIND] {self.name} → {type(agent).__name__} (agent_id={getattr(agent, 'agent_id', 'N/A')})"
        )
        return self

    def _is_valid_agent(self, obj: Any) -> bool:
        """Validate that the object is an Agent instance by checking for .agent_id attribute."""
        if obj is None:
            return False
        # Primary check: Agent class always has .agent_id
        return hasattr(obj, "agent_id") and isinstance(getattr(obj, "agent_id"), str)

    # ====================== EXECUTION ======================

    async def __call__(self, **kwargs) -> Any:
        """Execute the action — now with full support for streaming results.

        Streaming support:
        - If the executor returns an AsyncIterable (async generator or async iterator),
        it is passed directly to context_manager.record_tool_outcome() as a stream.
        - This enables chunk-by-chunk embedding without holding the entire output in memory.
        """
        logger.info(f"[ACTION START] {self.name}")
        logger.debug(f"[ACTION INPUT] {self.name} kwargs={kwargs}")

        # ==================== CONFIRMATION CHECK (unchanged) ====================
        confirmation_bypassed = kwargs.pop("_confirmation_passed", False)
        if self.require_confirmation and not confirmation_bypassed:
            preview = self.confirmation_preview(kwargs)
            logger.info(f"[ACTION CONFIRMATION REQUIRED] {self.name} preview={preview}")
            raise ConfirmationRequired(self.name, kwargs, preview)

        try:
            call_args = []
            if self._needs_agent:
                if self._bound_agent is None:
                    raise RuntimeError(
                        f"Action '{self.name}' expects agent/self but none was bound."
                    )
                call_args.append(self._bound_agent)

            # Execute the tool
            result = self.executor(*call_args, **kwargs)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                logger.debug(f"[ACTION AWAITING] {self.name}")
                result = await result

            self.usage_count += 1

            # ====================== STREAMING DETECTION & HANDLING ======================
            is_streaming = False
            if hasattr(
                result, "__aiter__"
            ):  # AsyncIterable (async generator, async iterator, etc.)
                is_streaming = True
                logger.info(
                    f"[ACTION STREAMING] {self.name} → detected async iterable result"
                )
                final_result_for_history = f"<streaming output — {self.name} — chunks will be embedded incrementally>"

            else:
                # Original sync / full-result path
                if isinstance(result, (dict, list, tuple)):
                    try:
                        final_result = json.dumps(result, ensure_ascii=False, indent=2)
                    except Exception:
                        final_result = str(result)
                else:
                    final_result = str(result) if result is not None else ""

                if not final_result or final_result.strip() in ("{}", "[]", "None", ""):
                    final_result = f"{self.name} executed successfully.\nNo output was returned by the tool."

                final_result_for_history = final_result

            success = not (isinstance(result, dict) and result.get("status") == "error")

            # ====================== GENERIC SUCCESS INDICATOR ======================
            step_success_indicator = {
                "tool_name": self.name,
                "success": success,
                "timestamp": time.time(),
                "step_index": getattr(self._bound_agent.state, "current_task_index", -1)
                if self._bound_agent is not None
                else -1,
                "streamed": is_streaming,
            }

            # ====================== RECORDING ======================
            if self._bound_agent is not None:
                if hasattr(self._bound_agent, "context_manager"):
                    try:
                        await self._bound_agent.context_manager.record_tool_outcome(
                            tool_name=self.name,
                            result=result,  # ← Pass raw result (can be AsyncIterable)
                            metadata={
                                "args": kwargs,
                                "action_type": self.type,
                                "usage_count": self.usage_count,
                                "success": success,
                                "raw_type": type(result).__name__,
                                "streamed": is_streaming,
                                "step_success_indicator": step_success_indicator,
                            },
                        )

                    except Exception as e:
                        logger.warning(f"[CONTEXT RECORD FAILED] {self.name}: {e}")

            logger.info(
                f"[ACTION SUCCESS] {self.name} | success={success} | streamed={is_streaming}"
            )
            return final_result_for_history  # Return a short placeholder for streaming case

        except ConfirmationRequired:
            raise
        except Exception as exc:
            logger.error(f"[ACTION ERROR] {self.name} error={exc}", exc_info=True)

            error_result = {
                "status": "error",
                "action": self.name,
                "error": str(exc),
                "args": kwargs,
            }

            if self._bound_agent is not None and hasattr(
                self._bound_agent, "context_manager"
            ):
                await self._bound_agent.context_manager.record_tool_outcome(
                    tool_name=self.name,
                    result=error_result,
                    metadata={"error": True},
                )

            return error_result


class ActionSet:
    """
    Manages a collection of Actions.
    Provides OpenAI-compatible `tools` list for LLMClient.
    """

    def __init__(
        self,
        name: str = "Actions",
        description: str = "",  # ← new
        actions: Optional[List[Action]] = None,  # optional convenience
    ):
        self.name = name
        self.description = description.strip()  # ← new
        self.actions: List[Action] = []
        self.by_name: Dict[str, Action] = {}

        if actions:
            for action in actions:
                self.add(action)

    def add(self, action: Action) -> None:
        if action.name in self.by_name:
            logger.error(f"[ACTIONSET DUPLICATE] {action.name}")
            raise ValueError(f"Action '{action.name}' already exists in this set")

        self.actions.append(action)
        self.by_name[action.name] = action

        logger.info(f"[ACTIONSET ADD] {action.name}")
        logger.debug(f"[ACTIONSET STATE] total_actions={len(self.actions)}")

    def get_llm_tools(
        self, include_tools: bool = True, include_functions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Returns list in modern OpenAI `tools` format, ready to pass to LLMClient.
        """
        tools = []
        for action in self.actions:
            if (include_tools and action.type == "tool") or (
                include_functions and action.type == "function"
            ):
                tool_spec = {
                    "type": "function",
                    "function": {
                        "name": action._raw_schema["name"],
                        "description": action._raw_schema["description"],
                        "parameters": action._raw_schema["parameters"],
                    },
                }
                if action.strict:
                    tool_spec["function"]["strict"] = True
                tools.append(tool_spec)
        return tools

    def get_executor(self, name: str, agent: Optional[Any] = None) -> Optional[Action]:
        """Return an Action (which acts as the executor) by name.
        Optionally binds the agent before returning.
        """
        action = self.by_name.get(name)
        if not action:
            logger.warning(f"[ACTIONSET RESOLVE FAIL] No executor for '{name}'")
            return None

        logger.debug(f"[ACTIONSET RESOLVE] Found executor for '{name}'")

        if agent is not None:
            action.bind_agent(agent)

        return action  # ← returns the Action object itself (which is callable)

    def describe(self) -> Dict[str, Any]:  # ← new helper
        """Returns a lightweight metadata dict useful for tool search / routing."""
        return {
            "name": self.name,
            "description": self.description,
            "action_count": len(self.actions),
            "action_names": [a.name for a in self.actions],
        }

    def remove(self, action: Action) -> None:
        """Remove a specific Action instance from the set."""
        if action not in self.actions:
            return
        self.actions.remove(action)
        self.by_name.pop(action.name, None)

    def remove_by_name(self, name: str) -> None:
        """Remove an action by its name (most convenient for unloading)."""
        action = self.by_name.pop(name, None)
        if action is not None:
            self.actions.remove(action)

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        desc_part = f" – {self.description[:60]}..." if self.description else ""
        return f"<ActionSet '{self.name}' ({len(self)} actions){desc_part}>"
