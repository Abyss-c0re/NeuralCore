import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Callable, Union

from neuralcore.agents.state import AgentState
from neuralcore.utils.logger import Logger

from neuralcore.actions.sequence import SequenceRegistry
from neuralcore.utils.config import get_loader


logger = Logger.get_logger()


class WorkflowEngine:
    """
    WorkflowEngine — Loader + registries + conditions + switching + run loop + ALL executors.
    AgentFlow only supplies the _wf_* steps.
    """

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
        self.sequence_registry = SequenceRegistry(self)
        self.load_workflow_from_config()

        logger.info(
            "✅ WorkflowEngine ready — AgentFlow _wf_* handlers loaded, executors stay in engine"
        )

    # ===================================================================
    # REGISTRATION & LOADER (stays in WorkflowEngine)
    # ===================================================================
    def register_workflow(
        self,
        name: str,
        description: str,
        steps: List[Union[str, Dict[str, Any]]],
        hidden_toolsets: Optional[Union[str, List[str]]] = None,
    ):
        self.registered_workflows[name] = {
            "description": description,
            "steps": steps.copy(),
            "hidden_toolsets": hidden_toolsets,  # store at workflow level
        }
        logger.info(f"✅ Registered workflow '{name}': {description}")
        if hidden_toolsets:
            logger.info(f"   → Hidden toolsets: {hidden_toolsets}")

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
        """Core state accessor with rich built-in conditions for reflection, sub-tasks, and safety."""
        key = key.lower().replace(" ", "_").replace("-", "_").replace(":", "_")

        # Direct attribute or property access (safe handling)
        if hasattr(state, key):
            value = getattr(state, key)
            # Safely resolve properties
            if isinstance(value, property):
                try:
                    return value.__get__(state, type(state))
                except Exception:
                    return None
            return value

        history = getattr(state, "iteration_history", [])

        # Basic iteration & reflection
        if key == "iteration":
            return len(history)
        if key == "reflection_count":
            return getattr(state, "reflection_count", 0)
        if key == "max_reflections_reached":
            return getattr(state, "reflection_count", 0) >= getattr(
                self.agent, "max_reflections", 3
            )

        # Sub-task / Multi-agent conditions
        if hasattr(self.agent, "get_sub_tasks"):
            sub_tasks = self.agent.get_sub_tasks()

            if key in ("sub_task_count", "sub_tasks_total"):
                return len(sub_tasks)
            if key == "active_sub_tasks_count":
                return sum(
                    1 for t in sub_tasks.values() if t.get("status") == "running"
                )
            if key == "sub_tasks_completed":
                return sum(
                    1 for t in sub_tasks.values() if t.get("status") == "completed"
                )
            if key == "sub_tasks_failed":
                return sum(1 for t in sub_tasks.values() if t.get("status") == "failed")

            if key == "all_sub_tasks_completed":
                return len(sub_tasks) > 0 and all(
                    t.get("status") in ("completed", "failed", "cancelled")
                    for t in sub_tasks.values()
                )
            if key == "any_sub_task_failed":
                return any(t.get("status") == "failed" for t in sub_tasks.values())
            if key == "all_sub_tasks_successful":
                return len(sub_tasks) > 0 and all(
                    t.get("status") == "completed" for t in sub_tasks.values()
                )
            if key == "sub_tasks_success_rate":
                completed = sum(
                    1 for t in sub_tasks.values() if t.get("status") == "completed"
                )
                total = len(sub_tasks)
                return completed / total if total > 0 else 0.0

        # Reflection & Self-correction
        if key == "needs_reflection":
            reflection_count = getattr(state, "reflection_count", 0)
            max_ref = getattr(self.agent, "max_reflections", 3)
            if reflection_count >= max_ref:
                return False

            tool_results = (
                getattr(state, "tool_results", [])
                if hasattr(state, "tool_results")
                else []
            )
            if tool_results:
                last = str(tool_results[-1].get("result", "")).lower()
                if any(
                    w in last
                    for w in ["error", "failed", "uncertain", "incomplete", "try again"]
                ):
                    return True

            recent = history[-3:] if history else []
            return all(not h.get("executed_functions") for h in recent)

        # Progress detection
        if key == "no_progress_last_n":
            n = 3
            recent = history[-n:] if history else []
            return all(
                not h.get("executed_functions") and not h.get("tool_calls")
                for h in recent
            )

        if key == "error_rate_high":
            results = (
                getattr(state, "tool_results", [])[-10:]
                if hasattr(state, "tool_results")
                else []
            )
            errors = sum(
                1
                for r in results
                if "error" in str(r.get("result", "")).lower()
                or "failed" in str(r.get("result", "")).lower()
            )
            return errors >= 3

        # Human-in-the-loop
        if key in ("needs_human_approval", "pending_approval", "needs_approval"):
            return getattr(state, "needs_approval", False)

        # Last step
        if key == "last_step_was":
            if history:
                last_steps = history[-1].get("workflow_steps_run", [])
                return last_steps[-1] if last_steps else None
            return None

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
        """Enhanced condition evaluator with support for string shorthand."""
        if cond is None:
            return True
        if isinstance(cond, bool):
            return cond

        # NEW: Support simple string conditions like "needs_reflection", "all_sub_tasks_completed"
        if isinstance(cond, str):
            value = self._get_state_value(cond, state)
            return bool(value) if value is not None else False

        if not isinstance(cond, dict):
            return bool(cond)

        # Keep your original custom condition support (in case you still want it for very advanced cases)
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

        # Logical combinators
        if "and" in cond and isinstance(cond["and"], (list, tuple)):
            return all(self._evaluate_condition(item, state) for item in cond["and"])
        if "or" in cond and isinstance(cond["or"], (list, tuple)):
            return any(self._evaluate_condition(item, state) for item in cond["or"])
        if "not" in cond:
            return not self._evaluate_condition(cond["not"], state)

        # Standard key-operator-value conditions
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
            logger.warning(
                f"Workflow '{name}' not found, attempting to reload workflows"
            )
            self.load_workflow_from_config()  # reload all workflows
            if name not in self.registered_workflows:
                logger.error(f"Workflow '{name}' still not found after reload")
                return False  # give up if still missing

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

    async def _drain_control(self, control_queue) -> Optional[Tuple[str, Any]]:
        try:
            msg = await asyncio.wait_for(control_queue.get(), timeout=0.02)
        except asyncio.TimeoutError:
            return None

        if isinstance(msg, dict):
            event = msg.get("event") or msg.get("action") or msg.get("control")

            # Include sub-agent events
            if event in {
                "switch_workflow",
                "go_to",
                "break",
                "finish_iteration",
                "restart_iteration",
                "insert_steps",
                "needs_confirmation",
                "cancelled",
                "finish",
                "sub_agent_progress",
                "sub_task_completed",
                "sub_task_failed",
            }:
                payload = msg.get("payload") or msg.get("data") or msg
                control_queue.task_done()
                return event, payload

        await control_queue.put(msg)
        control_queue.task_done()
        return None

    async def _iter_handler_events(
        self,
        handler: Callable,
        iteration: int,
        state: AgentState,
        timeout_sec: Optional[float],
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Unified async event stream with optional timeout."""
        if timeout_sec is not None:
            async with asyncio.timeout(timeout_sec):
                async for event, payload in handler(iteration, state):
                    yield event, payload
        else:
            async for event, payload in handler(iteration, state):
                yield event, payload

    def _apply_insert_steps(self, steps: List, payload: dict) -> None:
        new_steps = payload.get("steps", [])
        if payload.get("at_end"):
            steps.extend(new_steps)
            return

        target_name = payload.get("after")
        for i, cfg in enumerate(steps):
            name = cfg.get("name", "") if isinstance(cfg, dict) else cfg
            if name == target_name:
                steps[i + 1 : i + 1] = new_steps
                break

    def _handle_runtime_event(
        self,
        event: str,
        payload: Any,
        *,
        step_name: str,
        iteration: int,
        step_index: int,
        steps: List,
        from_control_queue: bool = False,
    ) -> Dict[str, Any]:
        """
        Centralized event handling for both handler events and control-queue events.
        Returns a dict of instructions for the caller.
        """
        out = {
            "yield_events": [],
            "go_to_target": None,
            "go_to_data": None,
            "step_index": step_index,
            "stop_iteration": False,
            "restart_iteration": False,
            "finish_iteration": False,
            "break_iteration": False,
            "switch_workflow": None,
        }

        if event == "switch_workflow":
            target = payload.get("name") if isinstance(payload, dict) else payload
            if isinstance(target, str):
                self.switch_workflow(target)
                out["yield_events"].append(("switch_workflow", {"name": target}))
                out["switch_workflow"] = target

        elif event == "go_to":
            if isinstance(payload, str):
                out["go_to_target"] = payload
            elif isinstance(payload, dict):
                out["go_to_target"] = payload.get("name") or payload.get("index")
                out["go_to_data"] = payload.get("data")
                if "offset" in payload:
                    offset = int(payload["offset"])
                    out["go_to_target"] = step_index + offset

        elif event == "break":
            out["yield_events"].append(
                (
                    "step_break",
                    {
                        "step": step_name,
                        "reason": payload
                        or ("control_message" if from_control_queue else "explicit"),
                    },
                )
            )
            out["step_index"] = len(steps)
            out["break_iteration"] = True

        elif event == "finish_iteration":
            out["yield_events"].append(("iteration_finished_early", payload or {}))
            out["step_index"] = len(steps)
            out["finish_iteration"] = True

        elif event == "restart_iteration":
            out["yield_events"].append(("iteration_restarted", payload or {}))
            out["step_index"] = -1
            out["restart_iteration"] = True

        elif event == "insert_steps":
            if isinstance(payload, dict):
                self._apply_insert_steps(steps, payload)

        elif event in ("needs_confirmation", "cancelled", "finish"):
            out["yield_events"].append((event, payload or {}))
            out["stop_iteration"] = True

        return out

    def _resolve_next_step(
        self,
        go_to_target: Optional[Union[str, int]],
        go_to_data: Optional[dict],
        *,
        step_name: str,
        iteration: int,
        step_index: int,
        steps: List,
    ) -> Tuple[int, List[Tuple[str, Any]]]:
        """Resolve normal next step or go_to jump."""
        emitted: List[Tuple[str, Any]] = []
        next_step_index = step_index + 1

        if go_to_target is None:
            return next_step_index, emitted

        if isinstance(go_to_target, str):
            target_index = None
            for i, cfg in enumerate(steps):
                name = cfg.get("name", "") if isinstance(cfg, dict) else cfg
                if name == go_to_target:
                    target_index = i
                    break

            if target_index is not None:
                next_step_index = target_index
                emitted.append(
                    (
                        "step_go_to",
                        {
                            "from_step": step_name,
                            "to_step": go_to_target,
                            "iteration": iteration,
                            "data": go_to_data,
                        },
                    )
                )
            else:
                emitted.append(
                    (
                        "warning",
                        f"Unknown go_to target '{go_to_target}' from step '{step_name}'",
                    )
                )

        elif isinstance(go_to_target, int):
            next_step_index = max(0, min(go_to_target, len(steps) - 1))
            emitted.append(
                (
                    "step_go_to",
                    {
                        "from_step": step_name,
                        "to_step": next_step_index,
                        "iteration": iteration,
                        "data": go_to_data,
                    },
                )
            )

        return next_step_index, emitted

    async def _run_step_handler(
        self,
        handler: Callable,
        *,
        iteration: int,
        state: AgentState,
        timeout_sec: Optional[float],
    ) -> AsyncIterator[Tuple[str, Any]]:
        async for event, payload in self._iter_handler_events(
            handler, iteration, state, timeout_sec
        ):
            yield event, payload

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

        MAX_ITERATIONS = getattr(self.agent, "max_iterations", 20)
        MAX_STEPS_PER_ITERATION = getattr(self.agent, "max_steps_per_iteration", 100)

        control_queue = getattr(self.agent, "message_queue", None)

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
                    condition = step_config.get("if") or step_config.get("when")
                    retries = step_config.get("retries", 0)
                    timeout_sec = step_config.get("timeout")
                else:
                    step_name = step_config
                    overrides = {}
                    condition = None
                    retries = 0
                    timeout_sec = None

                handler = self._step_handlers.get(step_name)
                if not handler:
                    yield ("warning", f"Unknown step: {step_name}")
                    step_index += 1
                    continue

                yield (
                    "step_start_detail",
                    {"step": step_name, "iteration": iteration, "index": step_index},
                )

                if condition is not None and not self._evaluate_condition(
                    condition, state
                ):
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

                # =============================================================
                # Configure tools for this specific step
                # =============================================================
                if hasattr(self.agent, "manager"):
                    # Only call configure_for_step for steps that explicitly want it via decorator
                    # Sub-agents rely more on assigned_tools + manual loading in start_complex_deployment
                    if not (
                        getattr(self.agent, "sub_agent", False)
                        and self.current_workflow_name == "sub_agent_execute"
                    ):
                        self.agent.manager.configure_for_step(step_name, self)
                    else:
                        # For sub-agents: very light touch
                        if step_name == "llm_stream":
                            logger.debug(
                                "Sub-agent llm_stream: preserving tools (assigned_tools take precedence)"
                            )
                            self.agent.manager.unload_tools(
                                ["BrowseTools"]
                            )  # only remove discovery tool
                        # Do NOT call full configure_for_step — it would unload assigned tools
                else:
                    logger.warning(
                        f"Agent has no manager. Skipping tool config for '{step_name}'."
                    )

                # Keep backward compatibility with old "toolset" override in step config
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

                if overrides:
                    yield (
                        "step_overrides_applied",
                        {"step": step_name, "overrides": dict(overrides)},
                    )

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

                go_to_target: Optional[Union[str, int]] = None
                go_to_data: Optional[dict] = None
                attempt = 0
                step_success = False

                while attempt <= retries:
                    try:
                        async for event, payload in self._run_step_handler(
                            handler,
                            iteration=iteration,
                            state=state,
                            timeout_sec=timeout_sec,
                        ):
                            yield (event, payload)

                            result = self._handle_runtime_event(
                                event,
                                payload,
                                step_name=step_name,
                                iteration=iteration,
                                step_index=step_index,
                                steps=steps,
                            )

                            for emitted in result["yield_events"]:
                                yield emitted

                            if result["switch_workflow"]:
                                pass

                            if result["go_to_target"] is not None:
                                go_to_target = result["go_to_target"]
                                go_to_data = result["go_to_data"]

                            if result["stop_iteration"]:
                                return

                            if result["break_iteration"]:
                                step_index = result["step_index"]
                                break

                            if result["finish_iteration"]:
                                step_index = result["step_index"]
                                break

                            if result["restart_iteration"]:
                                step_index = result["step_index"]
                                break

                            if event == "insert_steps" and isinstance(payload, dict):
                                self._apply_insert_steps(steps, payload)

                        else:
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
                        await asyncio.sleep(1.0 * (2**attempt))

                    attempt += 1

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

                # Handle control messages
                control = await self._drain_control(control_queue)
                if control:
                    event, payload = control
                    result = self._handle_runtime_event(
                        event,
                        payload,
                        step_name=step_name,
                        iteration=iteration,
                        step_index=step_index,
                        steps=steps,
                        from_control_queue=True,
                    )

                    for emitted in result["yield_events"]:
                        yield emitted

                    if result["stop_iteration"]:
                        return

                    if result["break_iteration"]:
                        step_index = result["step_index"]
                        break

                    if result["finish_iteration"]:
                        step_index = result["step_index"]
                        break

                    if result["restart_iteration"]:
                        step_index = result["step_index"]
                        break

                    if result["go_to_target"] is not None:
                        go_to_target = result["go_to_target"]
                        go_to_data = result["go_to_data"]

                    if result["switch_workflow"]:
                        pass

                next_step_index, emitted = self._resolve_next_step(
                    go_to_target,
                    go_to_data,
                    step_name=step_name,
                    iteration=iteration,
                    step_index=step_index,
                    steps=steps,
                )
                for item in emitted:
                    yield item

                step_index = next_step_index

            self._log_iteration_state(iteration, state)

        # Final summary (commented as in original)
        # async for ev, pl in self._generate_final_summary(state):
        #     yield (ev, pl)
