from typing import Any, Callable, Dict, List, Optional, Union

from neuralcore.actions.actions import Action

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


class SequenceAction(Action):
    """
    A composite Action that executes an ordered list of Actions,
    propagating context between steps.

    Supports optional pausing for input (UI or human) and autonomous execution.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Action],
        propagate_context: bool = True,
        output_from: Union[int, str, None] = -1,
        confirm_predicate: Optional[Callable[[Any], bool]] = None,
        step_names: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters={
                "input": {
                    "type": "any",
                    "description": "Initial context / input for the sequence (dict or string)",
                },
                "user_response": {
                    "type": "any",
                    "description": "Human reply when sequence is waiting",
                    "default": None,
                },
                "resume_token": {
                    "type": "string",
                    "description": "(internal) Used to resume a paused sequence",
                    "default": None,
                },
                "auto_response": {
                    "type": "any",
                    "description": "Optional default response for autonomous mode",
                    "default": None,
                },
            },
            executor=self.execute,
            action_type="tool",
            tags=["composite", "workflow", "multi-step", "agent"],
        )
        self.steps = steps
        self.propagate_context = propagate_context
        self.output_from = output_from
        self.confirm_predicate = confirm_predicate
        self.step_names = step_names or [f"step_{i}" for i in range(len(steps))]
        self._state: Optional[Dict[str, Any]] = None

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the sequence with context propagation, auto-response, and optional pausing.
        Logs every key step for observability.
        """
        logger.info(f"[SEQUENCE START] {self.name}")
        logger.debug(f"[SEQUENCE INPUT] {self.name} kwargs={kwargs}")

        input_data = kwargs.get("input")
        user_response = kwargs.get("user_response")
        resume_token = kwargs.get("resume_token")
        auto_response = kwargs.get("auto_response")

        # Ensure context is always a dict
        if isinstance(input_data, str) or input_data is None:
            input_data = {"full_reply": input_data or ""}

        if self._state is None or resume_token is None:
            # Fresh execution
            self._state = {
                "step_index": 0,
                "context": input_data,
                "results": [],
                "step_outputs": {},
            }
            logger.debug(f"[SEQUENCE STATE INIT] {self.name} state={self._state}")
        else:
            # Resuming a paused sequence
            if user_response is None and self.is_waiting():
                logger.warning(
                    f"[SEQUENCE RESUME FAIL] {self.name} waiting for human input but none provided"
                )
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
            logger.info(
                f"[SEQUENCE STEP START] {self.name} → {step_name} (index={step_index})"
            )
            logger.debug(f"[SEQUENCE CONTEXT BEFORE] {context}")

            step_kwargs = (
                {"input": context} if self.propagate_context else kwargs.copy()
            )

            try:
                result = await current_action(**step_kwargs)
                logger.debug(
                    f"[STEP EXECUTED] {step_name} result_type={type(result).__name__}"
                )
            except Exception as exc:
                logger.error(
                    f"[SEQUENCE STEP ERROR] {self.name} step={step_name} index={step_index} error={exc}",
                    exc_info=True,
                )
                return {
                    "status": "failed",
                    "step": step_name,
                    "step_index": step_index,
                    "error": str(exc),
                    "context": context,
                }

            # Handle _ask with auto_response
            if isinstance(result, dict) and "_ask" in result:
                logger.info(f"[SEQUENCE STEP PAUSED] {step_name} awaiting human input")
                if auto_response is not None:
                    context["_last_user_response"] = auto_response
                    result.pop("_ask")
                    logger.debug(
                        f"[SEQUENCE AUTO RESPONSE USED] {step_name} response={auto_response}"
                    )
                else:
                    # Pause for human input
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
                        "question": result["_ask"],
                        "step": step_name,
                        "step_index": step_index,
                        "context_preview": str(context)[:400] + "…"
                        if len(str(context)) > 400
                        else str(context),
                        "sequence_name": self.name,
                        **{k: v for k, v in result.items() if k != "_ask"},
                    }

            # Optional auto-confirmation
            if self.confirm_predicate and self.confirm_predicate(result):
                logger.info(
                    f"[SEQUENCE STEP CONFIRM] {step_name} awaiting confirmation"
                )
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

            # Normal step completion
            results.append(result)
            state["step_outputs"][step_name] = result
            logger.debug(
                f"[SEQUENCE STEP COMPLETE] {step_name} result={str(result)[:500]}"
            )

            if self.propagate_context:
                if isinstance(result, dict):
                    context.update(result)
                else:
                    context["_last_result"] = result

            state["step_index"] += 1
            step_index += 1

        # Sequence completed
        final_result = self._select_output(results, state["step_outputs"])
        logger.info(f"[SEQUENCE COMPLETE] {self.name} steps_executed={len(self.steps)}")
        logger.debug(f"[SEQUENCE FINAL RESULT] {str(final_result)[:500]}")

        completed_response = {
            "status": "completed",
            "sequence": self.name,
            "result": final_result,
            "all_results": results,
            "named_results": state["step_outputs"],
            "steps_executed": len(self.steps),
        }

        self._state = None
        return completed_response

    def _select_output(self, results: list, named: dict) -> Any:
        if self.output_from is None:
            return results
        if self.output_from == -1:
            return named.get(self.step_names[-1], results[-1])
        if isinstance(self.output_from, int):
            return (
                results[self.output_from]
                if 0 <= self.output_from < len(results)
                else results[-1]
            )
        if isinstance(self.output_from, str):
            return named.get(self.output_from, results[-1])
        return results[-1]

    def is_waiting(self) -> bool:
        return self._state is not None and self._state["step_index"] <= len(self.steps)

    def reset(self):
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


class SequenceRegistry:
    def __init__(self, engine):
        self.engine = engine
        self._sequences = {}

    def register(self, sequence, name=None):
        step_name = name or sequence.name

        self._sequences[step_name] = sequence

        async def handler(iteration, state):
            # basic input → you can tweak later
            input_data = {
                "full_reply": getattr(state, "full_reply", ""),
            }

            result = await sequence.execute(input=input_data)

            status = result.get("status")

            if status == "waiting_for_human":
                yield ("needs_input", result)
                return

            if status == "failed":
                yield ("error", result)
                return

            if status == "completed":
                final = result.get("result")

                # push result back into state
                if isinstance(final, dict):
                    if "full_reply" in final:
                        state.full_reply = final["full_reply"]
                else:
                    state.full_reply = str(final)

                yield ("sequence_done", {"name": step_name, "result": final})
                return

        self.engine.register_step(step_name, handler)

# Usage 

# engine = WorkflowEngine(agent)

# from neuralcore.workflows.sequence_registry import SequenceRegistry
# seq_registry = SequenceRegistry(engine)

# seq = sequence(
#     name="process_text",
#     description="Uppercase → wrap → finalize",
#     steps=[act_a, act_b, act_c],
# )

# seq_registry.register(seq)

# engine.register_workflow(
#     name="my_flow",
#     description="test",
#     steps=[
#         "plan_tasks",
#         "process_text",   # 👈 THIS NOW WORKS
#         "llm_stream",
#     ]
# )

