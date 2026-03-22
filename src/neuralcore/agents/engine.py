import asyncio
import re
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Callable, Union

from neuralcore.agents.state import AgentState, Phase
from neuralcore.utils.logger import Logger
from neuralcore.utils.exceptions_handler import ConfirmationRequired

logger = Logger.get_logger()


class WorkflowEngine:
    DEFAULT_WORKFLOW: List[Union[str, Dict[str, Any]]] = [
        "plan_tasks",
        "llm_stream",
        "execute_if_tools",
        "verify_goal_completion",
        "check_complete",
        "reflect_if_stuck",
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

        # Auto-register all built-in _wf_* methods
        self._register_builtin_handlers()
        self._register_builtin_workflow()

        # Load from config (now supports everything)
        self.load_workflow_from_config()

    # ===================================================================
    # USER-FRIENDLY REGISTRATION
    # ===================================================================
    def register_workflow(
        self,
        name: str,
        description: str,
        steps: List[Union[str, Dict[str, Any]]],
    ):
        """Register a workflow. Steps can contain 'include', 'if', 'overrides'."""
        self.registered_workflows[name] = {
            "description": description,
            "steps": steps.copy(),
        }
        logger.info(f"✅ Registered workflow '{name}': {description}")

    def register_step(self, name: str, handler: Callable):
        self._step_handlers[name] = handler
        logger.info(f"Registered external step: {name}")

    # ===================================================================
    # BUILT-IN REGISTRATION
    # ===================================================================
    def _register_builtin_handlers(self):
        count = 0
        for attr_name in dir(self):
            if attr_name.startswith("_wf_"):
                step_name = attr_name[4:]
                self._step_handlers[step_name] = getattr(self, attr_name)
                count += 1
        logger.info(f"Loaded {count} built-in workflow step handlers")

    def _register_builtin_workflow(self):
        self.register_workflow(
            name="default",
            description="Default ReAct loop + persistent goal + efficient ContextManager",
            steps=self.DEFAULT_WORKFLOW,
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

    # ===================================================================
    # WORKFLOW RESOLUTION (include support)
    # ===================================================================
    def _resolve_steps(self, steps: List, depth: int = 0) -> List:
        """Recursively resolve 'include' and return a flat list of steps."""
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

    # ===================================================================
    # OPTION 3: STRUCTURED CONDITIONS (if:)
    # ===================================================================
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
        return False

    def _evaluate_condition(self, cond: Any, state: AgentState) -> bool:
        """Full Option 3 structured condition evaluator."""
        if cond is None:
            return True
        if isinstance(cond, bool):
            return cond
        if not isinstance(cond, dict):
            return bool(cond)

        # Explicit logical operators (highest priority)
        if "and" in cond and isinstance(cond["and"], (list, tuple)):
            return all(self._evaluate_condition(item, state) for item in cond["and"])
        if "or" in cond and isinstance(cond["or"], (list, tuple)):
            return any(self._evaluate_condition(item, state) for item in cond["or"])
        if "not" in cond:
            return not self._evaluate_condition(cond["not"], state)

        # Implicit AND of all conditions in the dict
        for key, val in cond.items():
            actual = self._get_state_value(key, state)

            if isinstance(val, dict):
                # Operator style: iteration: { gt: 6 }
                for op_name, target in val.items():
                    if not self._compare(actual, op_name, target):
                        return False
            else:
                # Direct equality: has_tools: true
                if actual != val:
                    return False
        return True

    # ===================================================================
    # RUNTIME WORKFLOW SWITCHING
    # ===================================================================
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

    # ===================================================================
    # CONFIG LOADING
    # ===================================================================
    def load_workflow_from_config(self):
        wf_config = getattr(self.agent, "config", {}).get("workflow", {})

        # Named workflow (recommended)
        if isinstance(wf_config, dict) and "name" in wf_config:
            name = wf_config["name"]
            if name in self.registered_workflows:
                wf = self.registered_workflows[name]
                self.workflow_steps = self._resolve_steps(wf["steps"])
                self.workflow_description = wf["description"]
                self.current_workflow_name = name
                logger.info(f"Loaded named workflow → {name}")
                return

        # Fallback to raw steps (still supports if:/include)
        raw_steps = wf_config.get("steps", self.DEFAULT_WORKFLOW)
        self.workflow_steps = self._resolve_steps(raw_steps)
        self.workflow_description = wf_config.get("description", "Custom workflow")
        self.current_workflow_name = "custom"

        logger.info(
            f"Workflow loaded: {self.current_workflow_name} — {self.workflow_description}"
        )

    # ===================================================================
    # MAIN RUNNER (with full Option 3 + switch support)
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

        if hasattr(self.agent, "context_manager"):
            self.agent.context_manager.set_goal(user_prompt)
            await self.agent.context_manager.add_message("user", user_prompt)

        iteration = 0
        state = AgentState()

        while iteration < getattr(self.agent, "max_iterations", 20):
            iteration += 1
            yield (
                "step_start",
                {
                    "iteration": iteration,
                    "workflow": self.workflow_description,
                    "workflow_name": self.current_workflow_name,
                },
            )

            if stop_event and stop_event.is_set():
                yield ("cancelled", "User stop")
                return

            for step_config in self.workflow_steps.copy():
                if isinstance(step_config, dict):
                    step_name = step_config.get("name", "")
                    overrides = step_config.get(
                        "overrides", {}
                    ).copy()  # copy so we can pop safely
                else:
                    step_name = step_config
                    overrides = {}

                handler = self._step_handlers.get(step_name)
                if not handler:
                    yield ("warning", f"Unknown step: {step_name}")
                    continue

                # === OPTION 3: CONDITION ===
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
                            continue

                # === NEW: DYNAMIC TOOLSET SWITCHING PER STEP ===
                if "toolset" in overrides:
                    toolset_value = overrides.pop("toolset")  # consume it
                    if toolset_value:
                        # True switch: unload everything dynamic first, then load requested set(s)
                        self.agent.manager.unload_all()  # keeps browse_tools
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
                # ================================================

                # Apply remaining normal overrides (client, temperature, etc.)
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

                try:
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

                        if event in ("needs_confirmation", "cancelled", "finish"):
                            return

                    yield (
                        "step_completed",
                        {"step": step_name, "iteration": iteration},
                    )

                except Exception as e:
                    yield ("error", str(e))
                finally:
                    for k, v in original_params.items():
                        setattr(self.agent, k, v)

            self._log_iteration_state(iteration, state)

        async for ev, pl in self._generate_final_summary(state):
            yield (ev, pl)

    # ---------------- WORKFLOW STEP HANDLERS (can be overridden via register_step) ----------------
    async def _wf_chat(self, iteration: int, state: AgentState):
        """
        Persistent chat loop using ContextManager to maintain user/assistant history.
        Keeps chatting until a message contains [TASK].
        """
        state.phase = Phase.IDLE
        yield ("phase_changed", {"phase": state.phase.value})

        last_processed_msg: str = ""
        current_chat_task: Optional[asyncio.Task] = None

        while True:
            last_user_msg = self.agent.context_manager.get_last_user_message()
            if not last_user_msg or last_user_msg == last_processed_msg:
                await asyncio.sleep(0.1)
                continue

            last_processed_msg = last_user_msg

            messages = await self.agent.context_manager.provide_context(
                query=last_user_msg,
                system_prompt=self.agent.system_prompt,
            )

            async def chat_task_fn():
                reply = await self.agent.client.chat(
                    messages=messages,
                    temperature=self.agent.temperature,
                    max_tokens=self.agent.max_tokens,
                )
                return reply

            if current_chat_task is None or current_chat_task.done():
                current_chat_task = asyncio.create_task(chat_task_fn())
            else:
                await current_chat_task
                current_chat_task = asyncio.create_task(chat_task_fn())

            try:
                reply = await current_chat_task
            except asyncio.CancelledError:
                yield ("info", "Chat task externally cancelled")
                return

            for line in reply.split("\n"):
                if line.strip():
                    yield ("content_delta", line + "\n")
                    await asyncio.sleep(0.01)

            await self.agent.context_manager.add_message("assistant", reply)

            if "[TASK]" in last_user_msg:
                self.agent.task = last_user_msg.replace("[TASK]", "").strip()
                yield ("info", {"message": f"Task detected: {self.agent.task}"})
                state.phase = Phase.PLAN
                break

    async def _wf_llm_stream(self, iteration: int, state: AgentState):
        async for ev, pl in self._llm_stream_with_tools(iteration, state):
            if ev == "llm_response" and isinstance(pl, dict):
                state.full_reply = pl.get("full_reply", "")
                state.tool_calls = pl.get("tool_calls", [])
                state.is_complete = pl.get("is_complete", False)
            yield (ev, pl)
        self._log_iteration_state(iteration, state)

    async def _wf_plan_tasks(self, iteration: int, state: AgentState):
        # Only plan if not already
        if not state.planned_tasks:
            state.phase = Phase.PLAN
            yield ("phase_changed", {"phase": state.phase.value})
            logger.debug(f"Phase changed to PLAN for iteration {iteration}")

            # --- Stage 1: LLM generates registry queries ---
            query_prompt = f"""
            Agent must generate queries to identify tools from the registry
            that are relevant to accomplishing the following TASK:

            TASK:
            {self.agent.task}

            REQUIREMENTS:
            - Respond ONLY with a JSON array of short search queries.
            - NO explanation, NO extra text.
            JSON format: {{ "queries": ["query1", "query2", ...] }}
            """
            logger.debug("Sending query generation prompt to LLM")
            raw_queries = await self.agent.client.ask(query_prompt)

            queries_json = re.search(r"\{.*\}", str(raw_queries), re.DOTALL)
            queries_list = []
            if queries_json:
                try:
                    queries_list = json.loads(queries_json.group()).get("queries", [])
                    logger.debug(f"LLM returned queries: {queries_list}")
                except Exception as e:
                    yield ("warning", f"Failed to parse queries JSON: {e}")
                    logger.debug(f"Failed to parse queries JSON: {e}")

            # --- Stage 2: Search registry and load suggested tools ---
            suggested_tools = []
            for q in queries_list:
                results = self.agent.registry.search(q, limit=3)
                suggested_tools.extend([a.name for a, _ in results])
            suggested_tools = list(dict.fromkeys(suggested_tools))  # remove duplicates
            logger.debug(f"Suggested tools after registry search: {suggested_tools}")

            # Load them into DynamicActionManager
            if suggested_tools:
                self.agent.registry.manager.load_tools(suggested_tools)
                logger.debug(
                    f"Loaded tools into dynamic manager: {', '.join(suggested_tools)}"
                )
                yield ("info", f"Loaded tools: {', '.join(suggested_tools)}")

            # --- Stage 3: Ask LLM to generate ordered actionable steps ---
            plan_prompt = f"""
            Agent must plan a sequence of actionable steps to accomplish the TASK below.
            Only use tools available in the agent's dynamic toolset: {", ".join(self.agent.registry.manager._loaded_tools) or "none"}.
            Respond ONLY with JSON. NO explanations.

            TASK:
            {self.agent.task}

            REQUIREMENTS:
            - Ordered list of concise, actionable steps.
            - Strict JSON format: {{ "tasks": ["step1", "step2", ...] }}
            """
            logger.debug("Sending task planning prompt to LLM")
            raw_plan = await self.agent.client.ask(plan_prompt)

            plan_json = re.search(r"\{.*\}", str(raw_plan), re.DOTALL)
            if plan_json:
                try:
                    plan = json.loads(plan_json.group())
                    state.planned_tasks = plan.get("tasks", [])
                    state.current_task_index = 0
                    logger.debug(f"Planned tasks generated: {state.planned_tasks}")
                except Exception as e:
                    state.planned_tasks = []
                    yield ("warning", f"Failed to parse plan JSON: {e}")
                    logger.debug(f"Failed to parse plan JSON: {e}")
            else:
                state.planned_tasks = []
                yield ("warning", "No JSON found in planning response")
                logger.debug("No JSON found in planning response")

        # Yield next task
        if state.planned_tasks and state.current_task_index < len(state.planned_tasks):
            next_task = state.planned_tasks[state.current_task_index]
            await self.agent.context_manager.add_message(
                "system", f"[NEXT TASK REMINDER] {next_task}"
            )
            state.current_task_index += 1
            logger.debug(f"Yielding next planned task: {next_task}")
            yield ("planned_task", {"task": next_task})

    async def _wf_execute_if_tools(self, iteration: int, state: AgentState):
        """
        Executes any queued tool calls, clears them afterward,
        and switches the agent phase to DECISION.
        """
        if not state.tool_calls:
            return

        yield ("tool_calls", state.tool_calls)

        async for ev, pl in self._execute_tools(state.tool_calls, iteration, state):
            yield (ev, pl)

        # --- Clear executed tool calls to avoid repeat ---
        state.tool_calls = []
        logger.debug(f"Cleared tool_calls after execution at iteration {iteration}")

        # --- Track executed signatures to prevent double execution ---
        for r in self.agent.tool_results:
            sig = f"{r['name']}:{json.dumps(r['args'], sort_keys=True)}"
            self.agent.executed_signatures.add(sig)

        # --- Switch phase automatically ---
        state.phase = Phase.DECISION
        yield ("phase_changed", {"phase": state.phase.value})

        self._log_iteration_state(iteration, state)

    async def _wf_verify_goal_completion(self, iteration: int, state: AgentState):
        if not state.is_complete:
            return
        state.phase = Phase.DECISION
        yield ("phase_changed", {"phase": state.phase.value})
        yield ("goal_verification_start", {"iteration": iteration})

        summary = self.agent.context_manager.get_context_summary()
        prompt = f"""
            You are a strict goal auditor.

            GOAL:
            {self.agent.goal}

            LIVE CONTEXT SUMMARY:
            {summary}

            INVESTIGATION STATE:
            {json.dumps(self.agent.context_manager.investigation_state, indent=2)}

            Has the agent FULLY completed the goal? Reply STRICT JSON ONLY:

            {{
                "verified": true or false,
                "reason": "one-sentence explanation",
                "missing_steps": ["list"] or []
            }}
            """
        try:
            raw = await self.agent.client.ask(prompt)
            verification = json.loads(str(raw).strip())
        except Exception:
            verification = {
                "verified": False,
                "reason": "parse failed",
                "missing_steps": [],
            }

        yield ("goal_verification_result", verification)

        if not verification.get("verified", False):
            state.is_complete = False
            await self.agent.context_manager.add_message(
                "system", f"[GOAL VERIFICATION FAILED] {verification.get('reason')}"
            )
            yield ("info", {"message": "Goal verification FAILED — continuing"})
        else:
            yield ("info", {"message": "Goal verification PASSED"})

    async def _wf_check_complete(self, iteration: int, state: AgentState):
        if iteration == 1 and not state.tool_calls and not state.planned_tasks:
            state.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            yield (
                "llm_response",
                {"full_reply": state.full_reply, "tool_calls": [], "is_complete": True},
            )
            yield ("finish", {"reason": "casual_complete"})
            self._log_iteration_state(iteration, state)
            return

        if state.is_complete:
            state.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": state.phase.value})
            async for ev, pl in self._generate_final_summary(state):
                yield (ev, pl)
            yield ("finish", {"reason": "complete"})
            self._log_iteration_state(iteration, state)

    async def _wf_reflect_if_stuck(self, iteration: int, state: AgentState):
        # --- Cooldown: reflect only every 5 iterations ---
        REFLECT_INTERVAL = 5
        last_reflect = getattr(state, "last_reflection_iteration", 0)
        if iteration <= 3 or (iteration - last_reflect) < REFLECT_INTERVAL:
            return

        # --- Check for progress ---
        # No tool calls and no change in planned tasks/steps
        last_snapshot = getattr(state, "last_progress_snapshot", None)
        current_snapshot = {
            "planned_tasks": state.planned_tasks.copy(),
            "current_task_index": state.current_task_index,
            "tool_calls": state.tool_calls.copy() if state.tool_calls else [],
        }

        if last_snapshot and last_snapshot == current_snapshot:
            stuck = True
        else:
            stuck = False

        # Update last snapshot for next iteration
        state.last_progress_snapshot = current_snapshot

        if not stuck:
            return

        # --- Mark reflection iteration ---
        state.last_reflection_iteration = iteration

        async for ev, decision in self._force_reflection(iteration, state):
            yield (ev, decision)

            if ev != "reflection_decision":
                continue

            next_step = decision.get("next_step")
            tool_name = decision.get("tool_name")
            args = decision.get("arguments", {})

            if next_step == "tool" and tool_name:
                state.tool_calls = [
                    {"function": {"name": tool_name, "arguments": json.dumps(args)}}
                ]
                yield ("info", {"message": f"[REFLECTION] Enqueued tool: {tool_name}"})
                return
            elif next_step == "llm":
                await self.agent.context_manager.add_message(
                    "system", f"[REFLECTION GUIDANCE]\n{json.dumps(decision, indent=2)}"
                )
                return
            elif next_step == "finish":
                yield (
                    "finish",
                    {"reason": "reflection_finish", "details": decision.get("reason")},
                )
                return
            else:
                yield ("warning", f"Unknown reflection step: {next_step}")

    async def _wf_safety_fallback(self, iteration: int, state: AgentState):
        if iteration >= getattr(self.agent, "max_iterations", 20):
            async for ev, pl in self._generate_final_summary(state):
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})
            self._log_iteration_state(iteration, state)

    # ---------------- CORE METHODS (shared, used by step handlers) ----------------
    async def _llm_stream_with_tools(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.EXECUTE
        messages = await self.agent.context_manager.provide_context(
            query="",
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
        completed_tool_calls = []

        try:
            async for kind, payload in self.agent.client._drain_queue(queue):
                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)
                elif kind == "tool_delta":
                    yield ("tool_delta", payload)
                elif kind == "tool_complete":
                    completed_tool_calls.append(payload)
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
            "tool_calls": completed_tool_calls,
            "is_complete": self.FINAL_ANSWER_MARKER in text_buffer,
        }
        yield ("llm_response", response_state)

    async def _execute_tools(
        self, tool_calls: List[Dict], iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Execute each tool call in sequence. Skips already executed tools.
        Records outcomes and handles confirmations/errors.
        """
        state.phase = Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        for call in tool_calls or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}

            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in self.agent.executed_signatures:
                logger.debug(f"Skipping already executed tool: {sig}")
                continue

            self.agent.executed_signatures.add(sig)

            executor = self.agent.manager.get_executor(name)
            if not executor:
                yield ("tool_skipped", {"name": name, "reason": "no_executor"})
                continue

            yield ("tool_start", {"name": name, "args": args})

            task_id = f"{name}:{hash(json.dumps(args, sort_keys=True))}"
            if hasattr(self.agent.context_manager, "add_subtask"):
                self.agent.context_manager.add_subtask(task_id)

            try:
                if name in ("run", "self_run"):
                    result = "Self-run tool skipped"
                    yield (
                        "tool_skipped",
                        {"name": name, "reason": "recursion prevented"},
                    )
                else:
                    maybe = executor(**args)
                    result = await maybe if asyncio.iscoroutine(maybe) else maybe

                # --- Record results ---
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

            # Stop if client requested
            stop_event = getattr(self.agent.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", f"Stop after {name}")
                return

    async def _force_reflection(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        state.phase = Phase.REFLECT
        yield ("phase_changed", {"phase": state.phase.value})

        # Lightweight context summary
        summary = self.agent.context_manager.get_context_summary()

        # Minimal prompt to reduce LLM load
        prompt = f"""
        Agent is stuck after {iteration} iterations.

        TASK:
        {self.agent.task}

        LIVE CONTEXT SUMMARY (truncated):
        {summary}

        Return valid JSON ONLY with keys:
        - reason: why the agent is stuck
        - next_step: "tool", "llm", or "finish"
        - optional: tool_name, arguments, new_subtask, workflow_adjustments
        """

        raw_response = await self.agent.client.ask(prompt)

        # Safe JSON parsing
        try:
            decision = json.loads(str(raw_response).strip())
        except Exception:
            decision = {"reason": "parse failed", "next_step": "llm"}

        # --- Handle new subtask(s) ---
        if decision.get("new_subtask"):
            new_task = decision["new_subtask"]

            # Remove already completed tasks
            remaining_tasks = state.planned_tasks[state.current_task_index :]

            # Prepend the new subtask
            state.planned_tasks = [new_task] + remaining_tasks
            state.current_task_index = 0  # reset index to run new task next

            # Always record in ContextManager
            self.agent.context_manager.add_subtask(new_task)
            logger.info(
                f"[REFLECTION] Removed completed tasks and inserted new subtask: {new_task}"
            )

        # Record reason in ContextManager
        if decision.get("reason"):
            self.agent.context_manager.add_finding(f"Reflection: {decision['reason']}")

        # Log reflection for history
        await self.agent.context_manager.add_message(
            "system", f"[REFLECTION]\n{json.dumps(decision, indent=2)}"
        )
        await self.agent.context_manager.add_external_content(
            source_type="reflection",
            content=json.dumps(decision, indent=2),
            metadata={"iteration": iteration},
        )

        # Increment reflection count and check for stuck condition
        state.reflection_count += 1
        if state.reflection_count > getattr(
            self.agent, "max_reflections", 5
        ) and self._has_no_tools_recently(state, 5):
            yield ("warning", f"Stuck after {state.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            self._log_iteration_state(iteration, state)
            return

        # Apply optional workflow adjustments
        adjustments = decision.get("workflow_adjustments", {})
        if isinstance(adjustments, dict) and isinstance(
            adjustments.get("reorder_steps"), list
        ):
            self.workflow_steps = [
                s for s in adjustments["reorder_steps"] if s in self.workflow_steps
            ]

        state.last_reflection_decision = decision
        self._log_iteration_state(iteration, state)

        # Yield reflection events
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
