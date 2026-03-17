import asyncio

from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from neuralcore.utils.exceptions_handler import ConfirmationRequired


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
    """
    A composite Action that executes an ordered list of other Actions (or functions),
    optionally propagating context between steps.

    Supports pausing for human confirmation / input and later resuming.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Action],
        propagate_context: bool = True,
        output_from: Union[
            int, str, None
        ] = -1,  # -1 = last step, None = all results list
        confirm_predicate: Optional[Callable[[Any], bool]] = None,
        # Optional: allow naming steps for better debugging / output_from reference
        step_names: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters={
                "input": {
                    "type": "any",
                    "description": "Initial context / input for the sequence",
                },
                "resume_token": {
                    "type": "string",
                    "description": "(internal) Used by orchestrator to resume a paused sequence",
                    "default": None,
                },
                "user_response": {
                    "type": "any",
                    "description": "Human reply when sequence is waiting",
                    "default": None,
                },
            },
            executor=self.execute,  # ← public name, more conventional
            action_type="tool",
            tags=["composite", "workflow", "multi-step", "agent"],
        )
        self.steps = steps
        self.propagate_context = propagate_context
        self.output_from = output_from
        self.confirm_predicate = confirm_predicate
        self.step_names = step_names or [f"step_{i}" for i in range(len(steps))]

        # Persistent state for resumable execution
        self._state: Optional[Dict[str, Any]] = None

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Main entry point — called by the agent / orchestrator.
        Handles both fresh start and resume.
        """
        input_data = kwargs.get("input")
        user_response = kwargs.get("user_response")
        resume_token = kwargs.get(
            "resume_token"
        )  # usually just self.name when resuming

        if self._state is None or resume_token is None:
            # Fresh execution
            self._state = {
                "step_index": 0,
                "context": input_data,
                "results": [],
                "step_outputs": {},  # name → result (for named output_from)
            }
        else:
            # Resume — we expect user_response if we were waiting
            if user_response is None and self.is_waiting():
                return {
                    "status": "error",
                    "message": "Resume requested but no user_response provided",
                    "waiting": True,
                }

        state = self._state
        step_index = state["step_index"]
        context = state["context"]
        results = state["results"]

        while step_index < len(self.steps):
            current_action = self.steps[step_index]
            step_name = self.step_names[step_index]

            # Prepare arguments for this step
            step_kwargs = (
                {"input": context} if self.propagate_context else kwargs.copy()
            )

            try:
                result = await current_action(**step_kwargs)
            except Exception as exc:
                return {
                    "status": "failed",
                    "step": step_name,
                    "step_index": step_index,
                    "error": str(exc),
                    "context": context,
                }

            # Check if this step wants to pause for human input
            if isinstance(result, dict) and "_ask" in result:
                # Save state before pausing
                state.update(
                    {
                        "step_index": step_index + 1,  # resume AFTER this step
                        "context": context,
                        "results": results + [result],
                    }
                )
                self._state = state  # persist

                return {
                    "status": "waiting_for_human",
                    "question": result["_ask"],
                    "step": step_name,
                    "step_index": step_index,
                    "context_preview": str(context)[:400] + "…"
                    if len(str(context)) > 400
                    else str(context),
                    "sequence_name": self.name,
                    # Optional: include suggested_files / other metadata from decision step
                    **{k: v for k, v in result.items() if k != "_ask"},
                }

            # Optional auto-confirmation
            if self.confirm_predicate and self.confirm_predicate(result):
                state.update(
                    {
                        "step_index": step_index + 1,
                        "context": context,
                        "results": results + [result],
                    }
                )
                self._state = state
                return {
                    "status": "waiting_for_human",
                    "question": f"Approve continuing after step '{step_name}'?\nResult preview: {repr(result)[:300]}…",
                    "step": step_name,
                    "step_index": step_index,
                }

            # Normal progress
            results.append(result)
            state["step_outputs"][step_name] = result

            if self.propagate_context:
                context = result

            state["step_index"] += 1
            step_index += 1

        # Sequence completed
        final_result = self._select_output(results, state["step_outputs"])

        completed_response = {
            "status": "completed",
            "sequence": self.name,
            "result": final_result,
            "all_results": results,
            "named_results": state["step_outputs"],
            "steps_executed": len(self.steps),
        }

        # Clean up after success
        self._state = None
        return completed_response

    def _select_output(self, results: list, named: dict) -> Any:
        if self.output_from is None:
            return results
        if self.output_from == -1:
            return results[-1] if results else None
        if isinstance(self.output_from, int):
            return (
                results[self.output_from]
                if 0 <= self.output_from < len(results)
                else None
            )
        if isinstance(self.output_from, str):
            return named.get(self.output_from)
        return None

    def is_waiting(self) -> bool:
        """Quick check if sequence is paused and awaiting human input"""
        if self._state is None:
            return False
        return self._state["step_index"] <= len(self.steps)

    def reset(self):
        """Force clear state (useful in error recovery or testing)"""
        self._state = None


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


class ActionFromSequence:
    """
    Helper class to convert any SequenceAction into a normal Action
    that can be registered in ActionSet / used by the agent exactly like
    every other tool.

    Usage:
        seq = sequence("my_chain", "description", steps=[a, b, c])
        my_action = ActionFromSequence.create(seq)          # or .from_sequence(...)

        terminal_tools.add(my_action)
    """

    @staticmethod
    def create(
        sequence: SequenceAction,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
        action_type: str = "function",
    ) -> Action:

        final_name = name or sequence.name
        final_desc = description or sequence.description
        final_tags = tags or getattr(sequence, "tags", ["composite", "workflow"])

        parameters = {
            "input": {
                "type": "string",
                "description": "Optional starting input (path, JSON string, or empty)",
                "default": "",
            },
            "user_response": {
                "type": "string",
                "description": "Reply when waiting (yes / no / cancel / only <file1> <file2> / add <file> / more)",
                "default": None,
            },
        }
        if extra_parameters:
            parameters.update(extra_parameters)

        async def wrapper_executor(**kwargs) -> Any:
            d = await sequence.execute(**kwargs)
            s = d.get("status")

            if s == "waiting_for_human":
                return (
                    d["question"]
                    + "\n\n(please reply with yes/no/cancel/more/only/add...)"
                )

            if s == "completed":
                res = d.get("result")
                if isinstance(res, dict) and "contents" in res:
                    return res["contents"]
                return res or "Sequence finished successfully"

            return f"Status: {s} — {d}"

        return Action(
            name=final_name,
            description=final_desc,
            parameters=parameters,
            executor=wrapper_executor,
            required=[],
            action_type=action_type,
            tags=final_tags,
        )

    @staticmethod
    def from_sequence(
        name: str,
        description: str,
        steps: List[Action],
        *,
        propagate: bool = True,
        output_from: Union[int, str, None] = -1,
        confirm_predicate: Optional[Callable[[Any], bool]] = None,
        # extra params for the final Action
        tags: Optional[List[str]] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """
        One-liner convenience: build sequence + wrap in one call.

        Example:
            explore_action = ActionFromSequence.from_sequence(
                name="explore_codebase",
                description="Safe codebase explorer with approval step",
                steps=[tree_action, pwd_action, ls_action, decide_step, read_step],
                tags=["codebase", "explore"],
                extra_parameters={"quick": {"type": "boolean", "default": False}}
            )
        """
        seq = sequence(
            name=name + "_internal",
            description=description,
            steps=steps,
            propagate=propagate,
            output_from=output_from,
            confirm_predicate=confirm_predicate,
        )
        return ActionFromSequence.create(
            seq,
            name=name,
            description=description,
            tags=tags,
            extra_parameters=extra_parameters,
        )
