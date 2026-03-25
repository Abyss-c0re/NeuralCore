import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Callable, Union

from neuralcore.agents.state import AgentState, Phase
from neuralcore.utils.logger import Logger
from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.workflows.default_flow import AgentFlow
from neuralcore.actions.sequence import SequenceRegistry
from neuralcore.utils.config import get_loader


logger = Logger.get_logger()


class WorkflowEngine:
    """
    WorkflowEngine — Loader + registries + conditions + switching + run loop + ALL executors.
    AgentFlow only supplies the _wf_* steps.
    """

    DEFAULT_WORKFLOW: List[Union[str, Dict[str, Any]]] = [
        "plan_tasks",
        "llm_stream",
        "execute_if_tools",
        "verify_goal_completion",
        "check_complete",
        "reflect_if_stuck",
        "replan_if_reflected",
        "safety_fallback",
    ]

    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    def __init__(self, agent):
        self.agent = agent

        # === REGISTRIES ===
        self.registered_workflows: Dict[str, Dict[str, Any]] = {}
        self._step_handlers: Dict[str, Optional[Callable]] = {}

        self.current_workflow_name: str = "default"
        self.workflow_steps: List[Union[str, Dict[str, Any]]] = []
        self.workflow_description: str = ""
        self._custom_conditions: Dict[
            str, Callable[[AgentState, Optional[Dict]], bool]
        ] = {}

        # === LOAD AgentFlow (now contains all _wf_* methods) ===
        self.sequence_registry = SequenceRegistry(
            self
        )  # Register sequence as step for workflow

        self._register_builtin_workflow()
        self.load_workflow_from_config()

        logger.info(
            "✅ WorkflowEngine ready — AgentFlow _wf_* handlers loaded, executors stay in engine"
        )

    # ===================================================================
    # REGISTRATION & LOADER (stays in WorkflowEngine)
    # ===================================================================
    def register_workflow(
        self, name: str, description: str, steps: List[Union[str, Dict[str, Any]]]
    ):
        self.registered_workflows[name] = {
            "description": description,
            "steps": steps.copy(),
        }
        logger.info(f"✅ Registered workflow '{name}': {description}")

    def register_step(self, name: str, handler: Callable):
        self._step_handlers[name] = handler
        logger.info(f"Registered external step: {name}")

    def register_custom_condition(
        self,
        name: str,
        handler: Callable[[AgentState, Optional[Dict[str, Any]]], bool],
        description: str = "",
    ):
        """Register a fully custom Python condition.
        handler(state, args_dict) -> bool
        """
        self._custom_conditions[name] = handler
        logger.info(
            f"✅ Registered custom condition '{name}': {description or 'no desc'}"
        )

    def _register_builtin_workflow(self):
        self.register_workflow(
            name="default",
            description="Default ReAct loop + persistent goal + efficient ContextManager",
            steps=self.DEFAULT_WORKFLOW,
        )
        flow = AgentFlow(self)

        for attr_name in dir(flow):
            if attr_name.startswith("_wf_"):
                step_name = attr_name[4:]
                method = getattr(flow, attr_name)

                if callable(method):
                    self._step_handlers[step_name] = method

    def _build_objective_reminder(self) -> str:
        return (
            f"OBJECTIVE LOCKED:\nTask: {self.agent.task}\n"
            f"You MUST NOT output {self.FINAL_ANSWER_MARKER} until the goal is 100% complete.\n"
            f"Only append exactly {self.FINAL_ANSWER_MARKER} after verification passes."
        )

    def _log_iteration_state(self, iteration: int, state: AgentState):
        state.iteration_history.append(
            {
                "iteration": iteration,
                "tool_calls": state.tool_calls.copy() if state.tool_calls else [],
                "executed_functions": state.executed_functions.copy(),
                "workflow_steps_run": self.workflow_steps.copy(),
                "full_reply": state.full_reply,
                "planned_tasks": state.planned_tasks.copy(),
                "current_task_index": state.current_task_index,
            }
        )
        snapshot_str = json.dumps(state.iteration_history[-1], indent=2)
        logger.debug(f"[Iteration {iteration} State] {snapshot_str}")

    def _has_no_tools_recently(self, state: AgentState, lookback: int = 3) -> bool:
        recent = state.iteration_history[-lookback:]
        return all(not h.get("executed_functions") for h in recent)

    def _resolve_steps(self, steps: List, depth: int = 0) -> List:
        if depth > 8:
            logger.error("Workflow include depth limit reached")
            return steps
        resolved = []
        for item in steps:
            if isinstance(item, str):
                resolved.append(item)
            elif isinstance(item, dict):
                if "include" in item:
                    name = item["include"]
                    if name in self.registered_workflows:
                        included = self.registered_workflows[name]["steps"]
                        resolved.extend(self._resolve_steps(included, depth + 1))
                    else:
                        logger.warning(f"Unknown include '{name}'")
                else:
                    resolved.append(item.copy())
            else:
                resolved.append(item)
        return resolved

    def _get_state_value(self, key: str, state: AgentState) -> Any:
        key = key.lower().replace(" ", "_").replace("-", "_")

        # Direct state attributes
        if hasattr(state, key):
            return getattr(state, key)

        # Computed helpers
        history = getattr(state, "iteration_history", [])
        if key == "iteration":
            return len(history)
        if key == "has_tools":
            return bool(getattr(state, "tool_calls", None))
        if key == "is_complete":
            return bool(getattr(state, "is_complete", False))
        if key == "reflection_count":
            return getattr(state, "reflection_count", 0)
        if key == "no_tools_recently" or key == "no_tools_last_5":
            return self._has_no_tools_recently(state, 5)
        if key == "tool_count":
            return len(getattr(state, "tool_calls", []))

        # ContextManager helpers
        if hasattr(self.agent, "context_manager"):
            cm = self.agent.context_manager
            if hasattr(cm, key):
                return getattr(cm, key)
            if key in ("knowledge_items", "kb_size"):
                return len(getattr(cm, "knowledge_base", []))

        return None

    def _compare(self, actual: Any, op: str, target: Any) -> bool:
        op = op.lower().strip()

        # ─── Automatic type coercion for numeric comparisons ───
        if isinstance(actual, (int, float)) and isinstance(target, str):
            try:
                # Try to convert target to the same type as actual
                if isinstance(actual, int):
                    target = int(target)
                else:
                    target = float(target)
            except ValueError:
                # If conversion fails, treat as mismatch (safe default)
                logger.debug(
                    f"Cannot compare numeric actual={actual} with non-numeric target={target!r}"
                )
                return False

        # ─── Normal comparison logic ───
        if op in ("eq", "==", "="):
            return actual == target
        if op in ("ne", "!=", "<>"):
            return actual != target
        if op in ("gt", ">", "greater_than"):
            return (
                actual > target if actual is not None and target is not None else False
            )
        if op in ("gte", ">=", "ge"):
            return (
                actual >= target if actual is not None and target is not None else False
            )
        if op in ("lt", "<", "less_than"):
            return (
                actual < target if actual is not None and target is not None else False
            )
        if op in ("lte", "<=", "le"):
            return (
                actual <= target if actual is not None and target is not None else False
            )
        if op in ("in", "contains"):
            return (
                target in actual
                if isinstance(actual, (list, tuple, set, str))
                else False
            )

        logger.warning(f"Unknown comparison operator: {op}")
        return False

    def _evaluate_condition(self, cond: Any, state: AgentState) -> bool:
        if cond is None:
            return True
        if isinstance(cond, bool):
            return cond
        if not isinstance(cond, dict):
            return bool(cond)

        if "custom" in cond:
            custom = cond["custom"]
            if isinstance(custom, str):
                handler = self._custom_conditions.get(custom)
                if handler is not None:
                    return handler(state, None)
                return False

            if isinstance(custom, dict):
                name = custom.get("name")
                args = custom.get("args", {})
                if not isinstance(name, str):
                    return False

                handler = self._custom_conditions.get(name)
                if handler is not None:
                    return handler(state, args)
                return False

            return False

        if "and" in cond and isinstance(cond["and"], (list, tuple)):
            return all(self._evaluate_condition(item, state) for item in cond["and"])
        if "or" in cond and isinstance(cond["or"], (list, tuple)):
            return any(self._evaluate_condition(item, state) for item in cond["or"])
        if "not" in cond:
            return not self._evaluate_condition(cond["not"], state)

        for key, val in cond.items():
            actual = self._get_state_value(key, state)
            if isinstance(val, dict):
                for op_name, target in val.items():
                    if not self._compare(actual, op_name, target):
                        return False
            else:
                if actual != val:
                    return False
        return True

    def switch_workflow(self, name: str) -> bool:
        if name not in self.registered_workflows:
            logger.warning(f"Workflow '{name}' not found")
            return False
        wf = self.registered_workflows[name]
        self.workflow_steps = self._resolve_steps(wf["steps"])
        self.workflow_description = wf["description"]
        self.current_workflow_name = name
        logger.info(f"🔄 Switched to workflow '{name}' → {self.workflow_description}")
        return True

    def load_workflow_from_config(self):

        loader = get_loader()

        loader.load_workflow_sets(self)

        logger.info(
            f"Workflow loaded: {self.current_workflow_name} — {self.workflow_description}"
        )

    # ===================================================================
    # MAIN RUNNER
    # ===================================================================
    async def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1212,
        stop_event: Optional[asyncio.Event] = None,
        workflow: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        if workflow:
            self.switch_workflow(workflow)

        self.agent._reset_state()
        self.agent.task = user_prompt
        self.agent.goal = user_prompt
        self.agent.system_prompt = system_prompt
        self.agent.temperature = temperature
        self.agent.max_tokens = max_tokens
        logger.info(f"Starting run with workflow: {self.current_workflow_name}")
        logger.info(f"Active steps: {self.workflow_steps}")

        if hasattr(self.agent, "context_manager"):
            self.agent.context_manager.set_goal(user_prompt)
            await self.agent.context_manager.add_message("user", user_prompt)

        iteration = 0
        state = AgentState()

        # ============================ CONFIGURABLE LIMITS ============================
        MAX_ITERATIONS = getattr(self.agent, "max_iterations", 20)
        MAX_STEPS_PER_ITERATION = getattr(self.agent, "max_steps_per_iteration", 100)
        # =============================================================================

        while iteration < MAX_ITERATIONS:
            iteration += 1
            steps_executed_this_iteration = 0

            yield (
                "step_start",
                {
                    "iteration": iteration,
                    "workflow": self.workflow_description,
                    "workflow_name": self.current_workflow_name,
                    "total_steps": len(self.workflow_steps),
                },
            )

            if stop_event and stop_event.is_set():
                yield ("cancelled", "User stop")
                return

            # Snapshot of steps for this iteration (supports dynamic injection)
            steps = self.workflow_steps.copy()
            step_index = 0

            while step_index < len(steps):
                steps_executed_this_iteration += 1
                if steps_executed_this_iteration > MAX_STEPS_PER_ITERATION:
                    yield (
                        "error",
                        f"Too many steps executed in iteration {iteration} "
                        f"(possible infinite go_to / insert_steps loop). "
                        f"Limit is {MAX_STEPS_PER_ITERATION}.",
                    )
                    break

                step_config = steps[step_index]

                if isinstance(step_config, dict):
                    step_name = step_config.get("name", "")
                    overrides = step_config.get("overrides", {}).copy()
                else:
                    step_name = step_config
                    overrides = {}

                handler = self._step_handlers.get(step_name)
                if not handler:
                    yield ("warning", f"Unknown step: {step_name}")
                    step_index += 1
                    continue

                # Always emit step_start_detail BEFORE condition check (better observability)
                yield (
                    "step_start_detail",
                    {
                        "step": step_name,
                        "iteration": iteration,
                        "index": step_index,
                    },
                )

                # === CONDITION ===
                if isinstance(step_config, dict):
                    condition = step_config.get("if") or step_config.get("when")
                    if condition is not None:
                        if not self._evaluate_condition(condition, state):
                            yield (
                                "step_skipped",
                                {
                                    "step": step_name,
                                    "reason": "condition_not_met",
                                    "condition": condition,
                                },
                            )
                            step_index += 1
                            continue

                # === DYNAMIC TOOLSET SWITCHING ===
                if "toolset" in overrides:
                    toolset_value = overrides.pop("toolset")
                    if toolset_value:
                        loaded_count = self.agent.manager.load_toolsets(toolset_value)
                        yield (
                            "toolset_switched",
                            {
                                "step": step_name,
                                "toolset": toolset_value,
                                "loaded_count": loaded_count,
                                "message": f"Switched tools to {toolset_value}",
                            },
                        )

                # === OVERRIDES LOGGING ===
                if overrides:
                    yield (
                        "step_overrides_applied",
                        {"step": step_name, "overrides": dict(overrides)},
                    )

                # Apply overrides
                original_params = {
                    k: getattr(self.agent, k)
                    for k in ("client", "temperature", "max_tokens", "system_prompt")
                }

                if "client" in overrides and overrides["client"] in getattr(
                    self.agent, "clients", {}
                ):
                    self.agent.client = self.agent.clients[overrides["client"]]

                for k in ("temperature", "max_tokens", "system_prompt"):
                    if k in overrides:
                        setattr(self.agent, k, overrides[k])

                # ====================== STEP-LEVEL CONTROLS ======================
                retries = 0
                timeout_sec = None
                if isinstance(step_config, dict):
                    retries = step_config.get("retries", 0)
                    timeout_sec = step_config.get("timeout")

                go_to_target: Optional[Union[str, int]] = None
                go_to_data: Optional[dict] = None
                attempt = 0
                step_success = False

                while attempt <= retries:
                    try:
                        # === TIMEOUT + HANDLER EXECUTION ===
                        if timeout_sec is not None:
                            async with asyncio.timeout(timeout_sec):
                                async for event, payload in handler(iteration, state):
                                    yield (event, payload)

                                    # === INLINE CONTROL EVENT PROCESSING ===
                                    if event == "switch_workflow":
                                        target = (
                                            payload.get("name")
                                            if isinstance(payload, dict)
                                            else payload
                                        )
                                        if isinstance(target, str):
                                            self.switch_workflow(target)

                                    elif event == "go_to":
                                        if isinstance(payload, str):
                                            go_to_target = payload
                                        elif isinstance(payload, dict):
                                            go_to_target = payload.get(
                                                "name"
                                            ) or payload.get("index")
                                            go_to_data = payload.get("data")
                                            if "offset" in payload:
                                                offset = int(payload["offset"])
                                                go_to_target = step_index + offset

                                    elif event == "break":
                                        yield (
                                            "step_break",
                                            {
                                                "step": step_name,
                                                "reason": payload or "explicit",
                                            },
                                        )
                                        step_index = len(
                                            steps
                                        )  # force exit of step loop
                                        break

                                    elif event == "finish_iteration":
                                        yield (
                                            "iteration_finished_early",
                                            payload or {},
                                        )
                                        step_index = len(steps)
                                        break

                                    elif event == "restart_iteration":
                                        yield ("iteration_restarted", payload or {})
                                        step_index = -1
                                        break

                                    elif event == "insert_steps":
                                        if isinstance(payload, dict):
                                            new_steps = payload.get("steps", [])
                                            if payload.get("at_end"):
                                                steps.extend(new_steps)
                                            else:
                                                target_name = payload.get("after")
                                                for i, cfg in enumerate(steps):
                                                    name = (
                                                        cfg.get("name", "")
                                                        if isinstance(cfg, dict)
                                                        else cfg
                                                    )
                                                    if name == target_name:
                                                        steps[i + 1 : i + 1] = new_steps
                                                        break

                                    if event in (
                                        "needs_confirmation",
                                        "cancelled",
                                        "finish",
                                    ):
                                        return

                        else:
                            # No timeout – identical logic
                            async for event, payload in handler(iteration, state):
                                yield (event, payload)

                                if event == "switch_workflow":
                                    target = (
                                        payload.get("name")
                                        if isinstance(payload, dict)
                                        else payload
                                    )
                                    if isinstance(target, str):
                                        self.switch_workflow(target)

                                elif event == "go_to":
                                    if isinstance(payload, str):
                                        go_to_target = payload
                                    elif isinstance(payload, dict):
                                        go_to_target = payload.get(
                                            "name"
                                        ) or payload.get("index")
                                        go_to_data = payload.get("data")
                                        if "offset" in payload:
                                            offset = int(payload["offset"])
                                            go_to_target = step_index + offset

                                elif event == "break":
                                    yield (
                                        "step_break",
                                        {
                                            "step": step_name,
                                            "reason": payload or "explicit",
                                        },
                                    )
                                    step_index = len(steps)
                                    break

                                elif event == "finish_iteration":
                                    yield ("iteration_finished_early", payload or {})
                                    step_index = len(steps)
                                    break

                                elif event == "restart_iteration":
                                    yield ("iteration_restarted", payload or {})
                                    step_index = -1
                                    break

                                elif event == "insert_steps":
                                    if isinstance(payload, dict):
                                        new_steps = payload.get("steps", [])
                                        if payload.get("at_end"):
                                            steps.extend(new_steps)
                                        else:
                                            target_name = payload.get("after")
                                            for i, cfg in enumerate(steps):
                                                name = (
                                                    cfg.get("name", "")
                                                    if isinstance(cfg, dict)
                                                    else cfg
                                                )
                                                if name == target_name:
                                                    steps[i + 1 : i + 1] = new_steps
                                                    break

                                if event in (
                                    "needs_confirmation",
                                    "cancelled",
                                    "finish",
                                ):
                                    return

                        step_success = True
                        break

                    except asyncio.TimeoutError:
                        yield (
                            "step_timeout",
                            {
                                "step": step_name,
                                "timeout": timeout_sec,
                                "attempt": attempt + 1,
                            },
                        )
                        if attempt == retries:
                            raise
                    except Exception as e:
                        yield (
                            "step_retry",
                            {
                                "step": step_name,
                                "attempt": attempt + 1,
                                "max_retries": retries,
                                "error": str(e),
                            },
                        )
                        if attempt == retries:
                            raise
                        await asyncio.sleep(1.0 * (2**attempt))  # exponential backoff

                    attempt += 1

                # Restore original params
                for k, v in original_params.items():
                    setattr(self.agent, k, v)

                if not step_success:
                    step_index += 1
                    continue

                yield (
                    "step_completed",
                    {
                        "step": step_name,
                        "iteration": iteration,
                        "attempts": attempt + 1,
                    },
                )

                # === NEXT STEP LOGIC (including go_to) ===
                next_step_index = step_index + 1

                if go_to_target is not None:
                    if isinstance(go_to_target, str):
                        target_index = None
                        for i, cfg in enumerate(steps):
                            name = cfg.get("name", "") if isinstance(cfg, dict) else cfg
                            if name == go_to_target:
                                target_index = i
                                break
                        if target_index is not None:
                            next_step_index = target_index
                            yield (
                                "step_go_to",
                                {
                                    "from_step": step_name,
                                    "to_step": go_to_target,
                                    "iteration": iteration,
                                    "data": go_to_data,
                                },
                            )
                        else:
                            yield (
                                "warning",
                                f"Unknown go_to target '{go_to_target}' from step '{step_name}'",
                            )
                    elif isinstance(go_to_target, int):
                        next_step_index = max(0, min(go_to_target, len(steps) - 1))
                        yield (
                            "step_go_to",
                            {
                                "from_step": step_name,
                                "to_step": next_step_index,
                                "iteration": iteration,
                                "data": go_to_data,
                            },
                        )

                step_index = next_step_index

            # End of iteration
            self._log_iteration_state(iteration, state)

        # Final summary
        async for ev, pl in self._generate_final_summary(state):
            yield (ev, pl)

    # ===================================================================
    # EXECUTORS (stay in WorkflowEngine)
    # ===================================================================
    async def _llm_stream_with_tools(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.EXECUTE
        messages = await self.agent.context_manager.provide_context(
            query=state.current_task or "Continue",
            max_input_tokens=self.agent.max_tokens,
            reserved_for_output=12000,
            system_prompt=self._build_objective_reminder(),
            include_logs=True,
        )

        queue = await self.agent.client.stream_with_tools(
            messages=messages,
            tools=self.agent.manager.get_llm_tools(),
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
            tool_choice="auto",
        )

        text_buffer = ""
        all_tool_calls: List[Dict] = []

        try:
            async for kind, payload in self.agent.client._drain_queue(queue):
                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)

                elif kind == "tool_delta":
                    yield ("tool_delta", payload)

                elif kind == "tool_complete":
                    # Use canonical signature for deduplication (same as execution layer)
                    try:
                        args_dict = json.loads(payload["function"]["arguments"])
                        sig = f"{payload['function']['name']}:{json.dumps(args_dict, sort_keys=True)}"
                    except (json.JSONDecodeError, TypeError, KeyError):
                        sig = f"{payload.get('function', {}).get('name')}:{payload.get('function', {}).get('arguments', '')}"

                    if not any(
                        f"{c.get('function', {}).get('name')}:{json.dumps(c.get('function', {}).get('arguments', ''), sort_keys=True) if isinstance(c.get('function', {}).get('arguments'), (dict, str)) else c.get('function', {}).get('arguments', '')}"
                        == sig
                        for c in all_tool_calls
                    ):
                        all_tool_calls.append(payload)
                    yield ("tool_complete", payload)

                elif kind == "finish":
                    break

                elif kind == "error":
                    yield ("error", payload)
                    return

        except asyncio.CancelledError:
            yield ("cancelled", "Task cancelled")
            return

        response_state = {
            "full_reply": text_buffer.strip(),
            "tool_calls": all_tool_calls,
            "is_complete": self.FINAL_ANSWER_MARKER in text_buffer,
        }
        yield ("llm_response", response_state)

    async def _execute_tools(
        self, tool_calls: List[Dict], iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        for call in tool_calls or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}

            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            # Skip if already executed
            if sig in self.agent.executed_signatures:
                logger.debug(f"Skipping already executed tool: {sig}")
                continue

            self.agent.executed_signatures.add(sig)

            executor = self.agent.manager.get_executor(name, self.agent)
            if not executor:
                yield ("tool_skipped", {"name": name, "reason": "no_executor"})
                continue

            yield ("tool_start", {"name": name, "args": args})

            task_id = f"{name}:{hash(json.dumps(args, sort_keys=True))}"
            if hasattr(self.agent.context_manager, "add_subtask"):
                self.agent.context_manager.add_subtask(task_id)

            try:
                # prevent recursion
                if name in ("run", "self_run"):
                    result = "Self-run tool skipped"
                    yield (
                        "tool_skipped",
                        {"name": name, "reason": "recursion prevented"},
                    )
                else:
                    maybe = executor(**args)
                    result = await maybe if asyncio.iscoroutine(maybe) else maybe

                # record outcome
                await self.agent.context_manager.record_tool_outcome(
                    name, str(result), args
                )
                await self.agent.context_manager.add_message("tool", str(result))

                if hasattr(self.agent.context_manager, "complete_subtask"):
                    self.agent.context_manager.complete_subtask(task_id)
                if hasattr(self.agent.context_manager, "add_finding"):
                    self.agent.context_manager.add_finding(
                        f"{name} → {str(result)[:200]}"
                    )

                self.agent.tool_results.append(
                    {"name": name, "result": result, "args": args}
                )
                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                yield ("needs_confirmation", {**exc.__dict__, "tool_calls": tool_calls})
                return
            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                await self.agent.context_manager.record_tool_outcome(name, result, args)
                if hasattr(self.agent.context_manager, "add_unknown"):
                    self.agent.context_manager.add_unknown(
                        f"{name} failed: {str(exc)[:200]}"
                    )
                yield ("tool_result", {"name": name, "result": result, "error": True})

            stop_event = getattr(self.agent.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", f"Stop after {name}")
                return

    async def _force_reflection(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.REFLECT
        yield ("phase_changed", {"phase": state.phase.value})

        summary = self.agent.context_manager.get_context_summary()

        prompt = f"""
        Agent is stuck after {iteration} iterations.

        TASK:
        {self.agent.task}

        LIVE CONTEXT SUMMARY (truncated):
        {summary}

        CRITICAL: If the agent appears unable to make progress despite the context, 
        choose next_step: "finish" to avoid infinite loops. 
        Only choose "llm" if a clear next reasoning step will unstick it.
        Prefer suggesting a concrete tool or new_subtask when possible.

        Return valid JSON ONLY with keys:
        - reason: why the agent is stuck
        - next_step: "tool", "llm", or "finish"
        - optional: tool_name, arguments, new_subtask, workflow_adjustments
        """

        raw_response = await self.agent.client.ask(prompt)

        # Robust JSON parsing with fallback to prevent stuck loops on malformed LLM output
        raw_str = str(raw_response).strip()
        try:
            # Strip common markdown wrappers
            if raw_str.startswith("```json"):
                raw_str = raw_str[7:].split("```")[0].strip()
            elif raw_str.startswith("```"):
                raw_str = raw_str[3:].split("```")[0].strip()
            decision = json.loads(raw_str)
        except Exception:
            decision = {"reason": "parse failed", "next_step": "finish"}

        # Safety: force valid next_step
        if not isinstance(decision, dict) or decision.get("next_step") not in (
            "tool",
            "llm",
            "finish",
        ):
            decision = {"reason": "invalid decision", "next_step": "finish"}

        if decision.get("new_subtask"):
            new_task = decision["new_subtask"]
            remaining_tasks = state.planned_tasks[state.current_task_index :]
            state.planned_tasks = [new_task] + remaining_tasks
            state.current_task_index = 0
            self.agent.context_manager.add_subtask(new_task)
            logger.info(f"[REFLECTION] Inserted new subtask: {new_task}")

        if decision.get("reason"):
            self.agent.context_manager.add_finding(f"Reflection: {decision['reason']}")

        await self.agent.context_manager.add_message(
            "system", f"[REFLECTION]\n{json.dumps(decision, indent=2)}"
        )

        # Removed internal max-reflection check and count increment (now handled in caller)
        adjustments = decision.get("workflow_adjustments", {})
        if isinstance(adjustments, dict) and isinstance(
            adjustments.get("reorder_steps"), list
        ):
            self.workflow_steps = [
                s for s in adjustments["reorder_steps"] if s in self.workflow_steps
            ]

        state.last_reflection_decision = decision
        self._log_iteration_state(iteration, state)

        yield ("reflection_decision", decision)
        yield ("reflection_triggered", decision)

    async def _generate_final_summary(
        self, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.FINALIZE
        yield ("phase_changed", {"phase": state.phase.value})

        lines = [
            "# 🏁 Agent Execution Report",
            f"**Task:** {self.agent.task[:200]}...",
            f"**Goal:** {self.agent.goal}",
            "",
            "## 📊 ContextManager Stats",
            f"- KB items: {len(self.agent.context_manager.knowledge_base)}",
            f"- Files checked: {len(self.agent.context_manager.files_checked)}",
            f"- Tools executed: {len(self.agent.context_manager.tools_executed)}",
            f"- Archived turns: {len(self.agent.context_manager.current_topic.archived_history)}",
            "",
            "## 🛠️ Tool Results (last 10)",
        ]
        for r in self.agent.tool_results[-10:]:
            lines.append(f"- {r['name']}: {str(r.get('result', ''))[:120]}...")

        final_text = "\n".join(lines)
        yield ("final_summary", final_text)
        yield ("finish", {"reason": "task_complete", "summary": final_text})
