from inspect import signature

from typing import Any, Callable, Dict, List, Optional, Union

from neuralcore.actions.actions import Action

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


class SequenceAction(Action):
    """
    Composite sequenced action.
    - Inherits schema building, binding, recording, etc. from base Action
    - DynamicActionManager binds the agent to the outer sequence
    - We forward it to inner steps
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Action],
        propagate_context: bool = True,
        output_from: Union[int, str, None] = -1,
        confirm_predicate: Optional[Callable[[Any], bool]] = None,
        dependencies: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
        tags: Optional[List[str]] = None,
    ):
        # Clean parameters that are safe for LLM (no "any")
        parameters = {
            "input": {
                "type": "string",
                "description": "Initial input/context for the sequence (string or JSON-like text)",
            },
            "user_response": {
                "type": "string",
                "description": "Optional human response when waiting",
                "default": "",
            },
        }

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            executor=self.execute,
            action_type="tool",
            tags=tags or ["composite", "workflow", "multi-step"],
            # You can add require_confirmation=False, strict=False etc. if needed
        )

        self.steps = steps
        self.propagate_context = propagate_context
        self.output_from = output_from
        self.confirm_predicate = confirm_predicate
        self._dependency_map = self._normalize_dependencies(dependencies or {})

        self._step_names_to_resolve: Optional[List[str]] = None
        self._resolved_steps: Optional[List[Action]] = None
        self._bound_agent: Optional[Any] = None
        self._state: Optional[Dict[str, Any]] = None

    def _normalize_dependencies(self, raw: Dict) -> Dict[str, Dict[str, str]]:
        normalized: Dict[str, Dict[str, str]] = {}
        for target, mapping in raw.items():
            if isinstance(mapping, str):
                normalized[target] = {"input": mapping}
            elif isinstance(mapping, dict):
                normalized[target] = mapping
        return normalized

    def bind_agent(self, agent: Any) -> "SequenceAction":
        super().bind_agent(agent)
        self._bound_agent = agent
        logger.debug(f"[SEQUENCE BIND] Agent bound to sequence '{self.name}'")

        # Pre-bind resolved inner steps
        if self._resolved_steps:
            for step in self._resolved_steps:
                if (
                    hasattr(step, "bind_agent")
                    and getattr(step, "_bound_agent", None) is None
                ):
                    try:
                        step.bind_agent(agent)
                        logger.debug(
                            f"[SEQUENCE PRE-BIND] Inner '{getattr(step, 'name', 'unknown')}'"
                        )
                    except Exception as e:
                        logger.warning(f"Pre-bind failed: {e}")
        return self

    async def execute(self, **kwargs) -> Dict[str, Any]:
        # Lazy resolution
        if self._step_names_to_resolve is not None and self._resolved_steps is None:
            from neuralcore.actions.registry import registry

            resolved = []
            for name in self._step_names_to_resolve:
                if name not in registry.all_actions:
                    raise ValueError(
                        f"[SEQUENCE] Step '{name}' not found for '{self.name}'"
                    )
                action, _ = registry.all_actions[name]
                resolved.append(action)
            self._resolved_steps = resolved
            self.steps = resolved
            logger.debug(
                f"[SEQUENCE] Lazily resolved {len(resolved)} steps for '{self.name}'"
            )

        logger.info(f"[SEQUENCE START] {self.name}")
        logger.debug(f"[SEQUENCE INPUT] {kwargs}")

        input_data = kwargs.get("input") or ""
        if isinstance(input_data, str):
            input_data = {"full_reply": input_data}

        if self._state is None:
            self._state = {
                "step_index": 0,
                "context": input_data.copy()
                if isinstance(input_data, dict)
                else {"full_reply": input_data},
                "results": [],
                "step_outputs": {},
            }

        state = self._state
        step_index = state["step_index"]
        context = state["context"]
        results = state["results"]

        while step_index < len(self.steps):
            current_action = self.steps[step_index]
            step_name = getattr(current_action, "name", f"step_{step_index}")

            logger.info(
                f"[SEQUENCE STEP {step_index + 1}/{len(self.steps)}] → {step_name}"
            )

            # Build step_kwargs from dependencies
            step_kwargs: Dict[str, Any] = {}
            if step_name in self._dependency_map:
                for target_param, source in self._dependency_map[step_name].items():
                    if source == "input":
                        value = context.get("full_reply") or input_data
                    else:
                        src_result = state["step_outputs"].get(source)
                        if (
                            isinstance(src_result, dict)
                            and src_result.get("status") == "error"
                        ):
                            return {
                                "status": "failed",
                                "step": source,
                                "error": src_result.get("error"),
                            }
                        # First-line extraction for tools returning multiline strings
                        if isinstance(src_result, str):
                            lines = [
                                ln.strip()
                                for ln in src_result.splitlines()
                                if ln.strip()
                            ]
                            value = lines[0] if lines else src_result
                        else:
                            value = src_result
                    if value is not None:
                        step_kwargs[target_param] = value

            if not step_kwargs:
                try:
                    sig = signature(current_action.executor)
                    if "input" in sig.parameters:
                        step_kwargs = {"input": context.copy()}
                except Exception:
                    pass

            logger.info(f"[STEP CALL] {step_name} with kwargs: {step_kwargs}")

            # Forward agent (DynamicActionManager owns the initial bind)
            if self._bound_agent is not None and hasattr(current_action, "bind_agent"):
                if getattr(current_action, "_bound_agent", None) is None:
                    try:
                        current_action.bind_agent(self._bound_agent)
                        logger.debug(
                            f"[SEQUENCE BIND] Forwarded agent to inner step '{step_name}'"
                        )
                    except Exception as e:
                        logger.warning(f"Agent forward failed for '{step_name}': {e}")

            # Execute via real Action.__call__
            try:
                result = await current_action(**step_kwargs)
            except Exception as exc:
                logger.error(f"[STEP ERROR] {step_name}", exc_info=True)
                result = {"status": "error", "step": step_name, "error": str(exc)}

            results.append(result)
            state["step_outputs"][step_name] = result

            if self.propagate_context:
                if isinstance(result, dict):
                    context.update(result)
                else:
                    context["_last_result"] = result

            state["step_index"] += 1
            step_index += 1

        final_result = self._select_output(results, state["step_outputs"])
        logger.info(f"[SEQUENCE COMPLETE] {self.name}")

        return {
            "status": "completed",
            "sequence": self.name,
            "result": final_result,
            "all_results": results,
            "named_results": state["step_outputs"],
        }

    def _select_output(self, results: List, named: Dict) -> Any:
        if isinstance(self.output_from, str):
            return named.get(self.output_from, results[-1] if results else None)
        if isinstance(self.output_from, int) and 0 <= self.output_from < len(results):
            return results[self.output_from]
        return results[-1] if results else None

    def is_waiting(self) -> bool:
        return self._state is not None and self._state["step_index"] <= len(self.steps)

    def reset(self):
        self._state = None


# ─────────────────────────────────────────────────────────────
# Factories (fixed parameter name)
# ─────────────────────────────────────────────────────────────


def sequence(
    name: str,
    description: str,
    steps: List[Action],
    *,
    propagate: bool = True,  # public API name (what decorator uses)
    output_from: Union[int, str, None] = -1,
    confirm_predicate: Optional[Callable[[Any], bool]] = None,
    dependencies: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
    tags: Optional[List[str]] = None,
) -> SequenceAction:
    """
    Factory to create a SequenceAction.
    Maps 'propagate' (public) to 'propagate_context' (internal).
    """
    return SequenceAction(
        name=name,
        description=description,
        steps=steps,
        propagate_context=propagate,
        output_from=output_from,
        confirm_predicate=confirm_predicate,
        dependencies=dependencies,
        tags=tags,
    )


def async_sequence(
    name: str,
    description: str,
    steps: List[Action],
    *,
    propagate: bool = True,
    output_from: Union[int, str, None] = -1,
    confirm_predicate: Optional[Callable[[Any], bool]] = None,
    dependencies: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
    tags: Optional[List[str]] = None,
) -> SequenceAction:
    """Alias for clarity in async-heavy workflows."""
    return sequence(
        name=name,
        description=description,
        steps=steps,
        propagate=propagate,  # pass through
        output_from=output_from,
        confirm_predicate=confirm_predicate,
        dependencies=dependencies,
        tags=tags,
    )
