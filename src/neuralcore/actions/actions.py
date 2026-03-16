import asyncio

from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from src.utils.exceptions_handler import ConfirmationRequired


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
        tags: Optional[List[str]] = None,  # NEW
    ):
        if action_type not in {"tool", "function"}:
            raise ValueError("action_type must be 'tool' or 'function'")

        self.name = name
        self.description = description
        self.executor = executor
        self.type = action_type
        self.strict = strict
        self.tags = tags or []  # NEW
        self.usage_count = 0  # NEW (for ranking later)

        self.require_confirmation = require_confirmation
        self.confirmation_preview = confirmation_preview or (
            lambda kwargs: f"Executing {name} with {kwargs}"
        )

        params_schema: Dict[str, Any] = {
            "type": "object",
            "properties": parameters,
        }

        if required:
            params_schema["required"] = required

        if strict:
            params_schema["additionalProperties"] = False

        self._raw_schema = {
            "name": name,
            "description": description,
            "parameters": params_schema,
        }

        # PRECOMPUTED SEARCH TEXT
        self._search_text = " ".join(
            [self.name, self.description, " ".join(self.tags)]
        ).lower()

    async def __call__(self, **kwargs) -> Any:
        if self.require_confirmation:
            preview = self.confirmation_preview(kwargs)
            raise ConfirmationRequired(self.name, kwargs, preview)

        result = self.executor(**kwargs)

        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            result = await result

        self.usage_count += 1  # usage learning

        return result


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
        """Add an action. Raises ValueError if name already exists."""
        if action.name in self.by_name:
            raise ValueError(f"Action '{action.name}' already exists in this set")
        self.actions.append(action)
        self.by_name[action.name] = action

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

    def get_executor(self, name: str) -> Optional[Action]:
        """Return the Action object (not just the raw executor)."""
        return self.by_name.get(name)

    def describe(self) -> Dict[str, Any]:  # ← new helper
        """Returns a lightweight metadata dict useful for tool search / routing."""
        return {
            "name": self.name,
            "description": self.description,
            "action_count": len(self.actions),
            "action_names": [a.name for a in self.actions],
            # You can add more discoverable fields later, e.g.:
            # "categories": [...],
            # "domain": "web" | "math" | "files" | ...
        }

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        desc_part = f" – {self.description[:60]}..." if self.description else ""
        return f"<ActionSet '{self.name}' ({len(self)} actions){desc_part}>"


class SequenceAction(Action):
    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Action],
        propagate_context: bool = True,
        output_from: Union[int, str, None] = -1,
        # New: optional confirmation predicate or use special return value
        confirm_predicate: Optional[Callable[[Any], bool]] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters={"input": {"type": "any"}},
            executor=self._execute,  # placeholder — real logic in _execute / resume
            action_type="function",
        )
        self.steps = steps
        self.propagate_context = propagate_context
        self.output_from = output_from
        self.confirm_predicate = (
            confirm_predicate  # optional: auto-ask if returns True-ish
        )

        # Runtime state (for resume)
        self._last_state: Optional[Dict] = None

    async def _execute(self, **kwargs) -> Any:
        return await self.run(**kwargs)

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run or continue the sequence. Returns either final result or wait state."""
        if self._last_state is None:
            # Fresh start
            context = kwargs.get("input")
            step_index = 0
            results = []
        else:
            # Resume
            context = self._last_state.get("context")
            step_index = self._last_state["step_index"]
            results = self._last_state.get("results", [])
            user_response = kwargs.get("user_response")
            if user_response is None:
                raise ValueError("Resume requires 'user_response'")

            # Apply user decision (simple yes/no logic — extend as needed)
            last_result = results[-1] if results else None
            if isinstance(user_response, str) and user_response.lower() in (
                "no",
                "cancel",
                "reject",
            ):
                return {
                    "status": "cancelled_by_user",
                    "last_result": last_result,
                    "step_index": step_index - 1,
                }
            # else: continue with same context (or update if revise:...)
            if isinstance(user_response, str) and user_response.lower().startswith(
                "revise:"
            ):
                context = user_response[7:].strip()  # crude — improve parsing

        while step_index < len(self.steps):
            action = self.steps[step_index]

            step_kwargs = (
                {"input": context} if self.propagate_context else kwargs.copy()
            )
            result = await action(**step_kwargs)

            # Check for explicit "ask human" signal from the step
            if isinstance(result, dict) and "_ask" in result:
                question = result["_ask"]
                self._last_state = {
                    "context": context,
                    "step_index": step_index + 1,  # resume AFTER this step
                    "results": results + [result],
                }
                return {
                    "status": "waiting_for_human",
                    "question": question,
                    "current_context": context,
                    "step_index": step_index,
                }

            # Optional: auto-ask if confirm_predicate says so
            if self.confirm_predicate and self.confirm_predicate(result):
                self._last_state = {
                    "context": context,
                    "step_index": step_index + 1,
                    "results": results + [result],
                }
                return {
                    "status": "waiting_for_human",
                    "question": f"Approve this result?\n{repr(result)}",
                    "current_context": context,
                    "step_index": step_index,
                }

            results.append(result)
            if self.propagate_context:
                context = result

            step_index += 1

        # Finished
        self._last_state = None  # clear
        if self.output_from is None:
            final = results[-1] if results else None
        elif isinstance(self.output_from, int):
            final = results[self.output_from]  # type checker now happy
        elif isinstance(self.output_from, str):
            # Future: named result lookup
            # e.g. final = next(r for name, r in named_results if name == self.output_from)
            raise NotImplementedError("Named output_from not yet implemented")
        else:
            raise TypeError(f"Unsupported output_from: {self.output_from!r}")
        return {"status": "completed", "result": final, "all_results": results}

    async def resume(self, **kwargs) -> Dict[str, Any]:
        """Alias for run() when resuming — mainly for clearer API"""
        return await self.run(**kwargs)

    def is_waiting(self) -> bool:
        return self._last_state is not None


def sequence(
    name: str,
    description: str,
    steps: List[Action],
    *,
    propagate: bool = True,
    output_from: Union[int, str, None] = -1,
    confirm_predicate: Optional[Callable[[Any], bool]] = None,
) -> SequenceAction:
    """
    Create a composable sequence of actions — steps must be passed as a list.

    Usage:
        chain = sequence(
            name="process_text",
            description="Uppercase → wrap → finalize",
            steps=[act_a, act_b, act_c],
            propagate=True,
        )

    This version enforces the list style for maximum clarity and consistency.
    """
    return SequenceAction(
        name=name,
        description=description,
        steps=steps,  # no conversion needed — already a list
        propagate_context=propagate,
        output_from=output_from,
        confirm_predicate=confirm_predicate,
    )


def async_sequence(
    name: str,
    description: str,
    steps: List[Action],
    *,
    propagate: bool = True,
    output_from: Union[int, str, None] = -1,
    confirm_predicate: Optional[Callable[[Any], bool]] = None,
) -> SequenceAction:
    """
    Same interface as sequence(), but name hints at async or concurrent usage.
    """
    return SequenceAction(
        name=name,
        description=description,
        steps=steps,
        propagate_context=propagate,
        output_from=output_from,
        confirm_predicate=confirm_predicate,
    )
