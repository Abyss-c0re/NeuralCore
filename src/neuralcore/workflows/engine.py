import asyncio
import json
from inspect import signature
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Callable, Union

from neuralcore.agents.state import AgentState
from neuralcore.utils.logger import Logger

from neuralcore.actions.sequence import SequenceRegistry
from neuralcore.utils.config import get_loader


logger = Logger.get_logger()


class WorkflowEngine:
    """
    Cleaned-up WorkflowEngine (single-config.yaml mode).
    - All workflows come from top-level 'workflows:' in config.yaml
    - No folder scanning / duplicate loading
    - AgentFlow only supplies the _wf_* step handlers
    """

    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    def __init__(self, agent, workflow_registry=None):
        self.agent = agent

        # === REGISTRIES ===
        self.registered_workflows: Dict[str, Dict[str, Any]] = {}
        self._step_handlers: Dict[str, Optional[Callable]] = {}
        self.workflow = workflow_registry  # kept for metadata / loop support

        self.current_workflow_name: str = "default"
        self.workflow_steps: List[Union[str, Dict[str, Any]]] = []
        self.workflow_description: str = ""

        self._custom_conditions: Dict[str, Callable] = {}

        self.sequence_registry = SequenceRegistry(self)

        # Load everything from config (single source of truth)
        self.load_workflow_from_config()

        logger.info("✅ WorkflowEngine ready — workflows loaded from config.yaml")

    # ===================================================================
    # REGISTRATION
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
            "hidden_toolsets": hidden_toolsets,
        }
        logger.info(f"✅ Registered workflow '{name}': {description}")
        if hidden_toolsets:
            logger.info(f"   → Hidden toolsets: {hidden_toolsets}")

    def register_step(self, name: str, handler: Optional[Callable]) -> None:
        if handler is None:
            logger.warning(f"Attempted to register step '{name}' with None handler")
            return
        self._step_handlers[name] = handler
        logger.info(f"Registered external step: {name}")

    def register_custom_condition(
        self, name: str, handler: Callable, description: str = ""
    ):
        self._custom_conditions[name] = handler
        logger.info(
            f"✅ Registered custom condition '{name}': {description or 'no desc'}"
        )

    # ===================================================================
    # WORKFLOW LOADING (single config.yaml mode)
    # ===================================================================
    def load_workflow_from_config(self):
        """Load all workflows defined under top-level 'workflows:' in config.yaml."""
        loader = get_loader()

        # Let the simplified loader handle the global workflows dict
        loader.load_workflow_sets(self)

        # Register every workflow from config into the engine
        global_workflows = loader.config.get("workflows", {})
        for name, wf_data in global_workflows.items():
            if name not in self.registered_workflows:
                self.register_workflow(
                    name=name,
                    description=wf_data.get("description", f"Workflow {name}"),
                    steps=wf_data.get("steps", []),
                    hidden_toolsets=wf_data.get("hidden_toolsets"),
                )

        # Apply the primary workflow for this agent
        agent_workflow_cfg = getattr(self.agent, "config", {}).get("workflow")
        primary_name = None

        if isinstance(agent_workflow_cfg, str):
            primary_name = agent_workflow_cfg.strip()
        elif isinstance(agent_workflow_cfg, dict):
            primary_name = next(iter(agent_workflow_cfg.keys()), None)

        if primary_name and primary_name in self.registered_workflows:
            self.switch_workflow(primary_name)
        else:
            # Ultimate fallback
            self.current_workflow_name = "default"
            self.workflow_description = "Default workflow"
            logger.warning("No primary workflow found in agent config — using default")

        logger.info(
            f"Workflow loaded: {self.current_workflow_name} — {self.workflow_description}"
        )

    def switch_workflow(self, name: str) -> bool:
        if name not in self.registered_workflows:
            logger.warning(f"Workflow '{name}' not found, attempting reload")
            self.load_workflow_from_config()

            if name not in self.registered_workflows and hasattr(
                self.agent, "get_parent_agent"
            ):
                parent = self.agent.get_parent_agent()
                if parent and hasattr(parent, "workflow"):
                    self.inherit_workflows_from_parent(parent.workflow)

            if name not in self.registered_workflows:
                logger.error(f"Workflow '{name}' still not found after reload")
                return False

        wf = self.registered_workflows[name]
        self.workflow_steps = self._resolve_steps(wf["steps"])
        self.workflow_description = wf["description"]
        self.current_workflow_name = name

        logger.info(f"🔄 Switched to workflow '{name}' → {self.workflow_description}")
        return True

    # ===================================================================
    # CONDITIONS
    # ===================================================================
    def _get_state_value(self, key: str, state: AgentState) -> Any:
        """Core state accessor with built-in computed conditions.

        Supports direct attributes/properties from AgentState and derived values
        (iteration, error_rate, sub-tasks, goal achievement, etc.).
        """
        if not key:
            return None

        # Normalize key for flexible condition writing in YAML configs
        key = key.lower().replace(" ", "_").replace("-", "_").replace(":", "_")

        # 1. Direct attribute or property access (most common case)
        if hasattr(state, key):
            value = getattr(state, key)
            if isinstance(value, property):
                try:
                    return value.__get__(state, type(state))
                except Exception:
                    return None
            return value

        # 2. Built-in computed / derived values
        history = getattr(state, "iteration_history", [])

        if key == "iteration":
            return len(history)

        if key == "no_progress_last_n":
            n = 3
            recent = history[-n:] if history else []
            return all(
                not h.get("executed_functions") and not h.get("tool_calls")
                for h in recent
            )

        if key == "error_rate_high":
            results = getattr(state, "tool_results", [])[-10:]
            errors = sum(
                1
                for r in results
                if "error" in str(r.get("result", "")).lower()
                or "failed" in str(r.get("result", "")).lower()
            )
            return errors >= 3

        if key == "last_step_was":
            if history:
                last_steps = history[-1].get("workflow_steps_run", [])
                return last_steps[-1] if last_steps else None
            return None

        # 3. Human-in-the-loop (support multiple common names used in configs)
        if key in ("needs_human_approval", "pending_approval", "needs_approval"):
            return getattr(state, "needs_approval", False)

        # 4. Goal achievement conditions (clean & reliable)
        if key in ("goal_achieved", "goal_reached", "goal_complete"):
            return getattr(state, "goal_achieved", False) or getattr(
                state, "is_complete", False
            )

        # 5. Sub-task / multi-agent conditions (delegated to agent)
        if key.startswith("sub_") and hasattr(self.agent, "get_sub_tasks"):
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

        # Unknown key → safe default (prevents crashes in conditions)
        return None

    def _compare(self, actual: Any, op: str, target: Any) -> bool:
        """Compare actual value against target using the given operator.

        Supports numeric coercion and common operators used in workflow conditions.
        """
        op = op.lower().strip()

        # Automatic type coercion for numeric comparisons (very useful in YAML configs)
        if isinstance(actual, (int, float)) and isinstance(target, str):
            try:
                target = int(target) if isinstance(actual, int) else float(target)
            except ValueError:
                logger.debug(
                    f"Cannot compare numeric actual={actual} with non-numeric target={target!r}"
                )
                return False

        # Standard comparison logic
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
        """Evaluate a single condition (bool, string shorthand, dict, or logical combinators)."""
        if cond is None:
            return True
        if isinstance(cond, bool):
            return cond

        # Simple string condition (e.g. "goal_achieved" or "error_rate_high")
        if isinstance(cond, str):
            value = self._get_state_value(cond, state)
            return bool(value) if value is not None else False

        if not isinstance(cond, dict):
            return bool(cond)

        # Custom registered conditions
        if "custom" in cond:
            custom = cond["custom"]
            if isinstance(custom, str):
                handler = self._custom_conditions.get(custom)
                return bool(handler(state, None)) if handler else False

            if isinstance(custom, dict):
                name = custom.get("name")
                args = custom.get("args", {})
                if not isinstance(name, str):
                    return False
                handler = self._custom_conditions.get(name)
                return bool(handler(state, args)) if handler else False

            return False

        # Logical combinators: and / or / not
        if "and" in cond and isinstance(cond["and"], (list, tuple)):
            return all(self._evaluate_condition(item, state) for item in cond["and"])
        if "or" in cond and isinstance(cond["or"], (list, tuple)):
            return any(self._evaluate_condition(item, state) for item in cond["or"])
        if "not" in cond:
            return not self._evaluate_condition(cond["not"], state)

        # Standard key: {operator: value} syntax
        for key, val in cond.items():
            actual = self._get_state_value(key, state)
            if isinstance(val, dict):
                for op_name, target in val.items():
                    if not self._compare(actual, op_name, target):
                        return False
            else:
                # Simple equality check
                if actual != val:
                    return False
        return True

    def evaluate_named_condition(
        self, condition_name: str, state: AgentState, args: Optional[dict] = None
    ) -> bool:
        """
        Evaluate a named custom condition registered via @workflow.condition decorator.
        Falls back to built-in evaluator if no custom handler exists.
        """
        handler = self._custom_conditions.get(condition_name)
        if handler is not None:
            try:
                sig = signature(handler)
                param_count = len(sig.parameters)

                if param_count == 0:
                    return bool(handler())
                elif param_count == 1:
                    return bool(handler(state))
                else:
                    return bool(handler(state, args))

            except Exception as e:
                logger.error(
                    f"Error executing custom condition '{condition_name}': {e}",
                    exc_info=True,
                )
                return False

        # Fallback to built-in condition evaluator
        return self._evaluate_condition(condition_name, state)

    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    def _build_objective_reminder(self) -> str:
        return (
            f"OBJECTIVE LOCKED:\nTask: {self.agent.state.task}\n"
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

    def inherit_workflows_from_parent(self, parent_engine: "WorkflowEngine"):
        """
        Sub-agents call this to copy all workflows and step handlers from the parent.
        """
        if not parent_engine or not hasattr(parent_engine, "registered_workflows"):
            logger.warning("Cannot inherit workflows: invalid parent engine")
            return

        inherited_wf = 0
        inherited_steps = 0

        # Inherit workflows
        for name, wf_data in parent_engine.registered_workflows.items():
            if name not in self.registered_workflows:
                self.register_workflow(
                    name=name,
                    description=wf_data.get("description", f"Inherited: {name}"),
                    steps=wf_data.get("steps", []).copy(),
                    hidden_toolsets=wf_data.get("hidden_toolsets"),
                )
                inherited_wf += 1
                logger.debug(f"Inherited workflow '{name}' from parent")

        # Inherit step handlers
        for step_name, handler in parent_engine._step_handlers.items():
            if step_name not in self._step_handlers:
                self.register_step(step_name, handler)
                inherited_steps += 1
                logger.debug(f"Inherited step handler '{step_name}' from parent")

        if inherited_wf > 0 or inherited_steps > 0:
            logger.info(
                f"✅ Sub-agent inherited {inherited_wf} workflows and "
                f"{inherited_steps} steps from parent engine"
            )
        else:
            logger.warning("No new workflows/steps were inherited from parent")

    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Delegate step metadata lookup to the underlying Workflow registry."""
        if self.workflow is None:
            logger.warning(
                f"No workflow registry attached when requesting metadata for '{step_name}'"
            )
            return None
        return self.workflow.get_step_metadata(step_name)

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
        """Centralized event handling."""
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

    async def execute_loop(
        self, loop_name: str, initial_state: Optional[dict] = None, **kwargs
    ) -> AsyncIterator[Tuple[str, Any]]:
        if self.workflow is None:
            raise RuntimeError("Workflow registry not attached")

        loop_meta = self.workflow.get_loop_metadata(loop_name)
        if not loop_meta:
            raise ValueError(f"Loop '{loop_name}' not found.")

        state: AgentState = (
            initial_state if isinstance(initial_state, AgentState) else AgentState()
        )

        max_iter = loop_meta.get("max_iterations", 10)
        break_condition = loop_meta.get("break_condition")

        logger.info(f"🔄 Starting loop '{loop_name}' (max {max_iter} iterations)")

        for iteration in range(1, max_iter + 1):
            logger.debug(f"[{loop_name}] iteration {iteration}/{max_iter}")

            handler = loop_meta["handler"]

            async for event, payload in handler(self.agent, state, **kwargs):
                yield event, payload

                if event == "llm_response" and isinstance(payload, dict):
                    state.full_reply = payload.get("full_reply", state.full_reply)
                    state.tool_calls = payload.get("tool_calls", state.tool_calls)
                    state.is_complete = payload.get("is_complete", state.is_complete)

                if break_condition and self.evaluate_named_condition(
                    break_condition, state
                ):
                    logger.info(
                        f"Loop '{loop_name}' broke on condition '{break_condition}'"
                    )
                    yield (
                        "loop_broken",
                        {"condition": break_condition, "loop_name": loop_name},
                    )
                    return

        else:
            logger.warning(f"Loop '{loop_name}' reached max iterations ({max_iter})")

        yield ("loop_completed", {"loop_name": loop_name})

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
        self.agent.state.task = user_prompt
        self.agent.state.goal = user_prompt
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
                    if not (
                        getattr(self.agent, "sub_agent", False)
                        and self.current_workflow_name == "sub_agent_execute"
                    ):
                        self.agent.manager.configure_for_step(step_name, self)
                    else:
                        if step_name == "llm_stream":
                            logger.debug(
                                "Sub-agent llm_stream: preserving tools (assigned_tools take precedence)"
                            )
                            self.agent.manager.unload_tools(["FindTool"])
                else:
                    logger.warning(
                        f"Agent has no manager. Skipping tool config for '{step_name}'."
                    )

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

        # Final summary can be re-enabled here if needed
        # async for ev, pl in self._generate_final_summary(state):
        #     yield (ev, pl)
