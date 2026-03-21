import asyncio
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

        # === NAMED WORKFLOW REGISTRY (new user-friendly system) ===
        self.registered_workflows: Dict[str, Dict[str, Any]] = {}
        self.current_workflow_name: str = "default"
        self.workflow_steps: List[Union[str, Dict[str, Any]]] = []
        self.workflow_description: str = ""

        self._step_handlers: Dict[str, Optional[Callable]] = {}

        # 1. Separate & auto-load all existing _wf_* methods
        self._register_builtin_handlers()

        # 2. Register the default workflow with proper name + description
        self._register_builtin_workflow()

        # 3. Load from agent config (supports both old "steps" style and new "name" style)
        self.load_workflow_from_config()

    # ===================================================================
    # USER-FRIENDLY WORKFLOW REGISTRATION
    # ===================================================================
    def register_workflow(
        self, name: str, description: str, steps: List[Union[str, Dict[str, Any]]]
    ):
        """Recommended way to register any workflow (built-in, plugin, or custom).

        Every workflow now **must** have a clear name and description.
        """
        self.registered_workflows[name] = {
            "description": description,
            "steps": steps.copy(),
        }

        # Safety check
        unknown_steps = []
        for step in steps:
            step_name = step.get("name") if isinstance(step, dict) else step
            if step_name not in self._step_handlers:
                unknown_steps.append(step_name)

        if unknown_steps:
            logger.warning(
                f"Workflow '{name}' references unknown steps: {unknown_steps}. "
                "Register them with register_step() first."
            )

        logger.info(f"✅ Registered workflow '{name}': {description}")

    def register_step(self, name: str, handler: Callable):
        """Register a single external/custom step (for plugins)."""
        self._step_handlers[name] = handler
        logger.info(f"Registered external workflow step: {name}")

    # ===================================================================
    # BUILT-IN REGISTRATION (separation of methods)
    # ===================================================================
    def _register_builtin_handlers(self):
        """Automatically discover and register every _wf_* method on the engine.
        This cleanly separates the step logic from the engine core while still
        making the built-in methods available for any workflow."""
        count = 0
        for attr_name in dir(self):
            if attr_name.startswith("_wf_"):
                step_name = attr_name[4:]  # _wf_plan_tasks → plan_tasks
                handler = getattr(self, attr_name)
                self._step_handlers[step_name] = handler
                count += 1

        logger.info(f"Loaded {count} built-in workflow step handlers")

    def _register_builtin_workflow(self):
        """Load the original default workflow as a named workflow (name + description)."""
        self.register_workflow(
            name="default",
            description="Default ReAct loop + persistent goal + efficient ContextManager",
            steps=self.DEFAULT_WORKFLOW,
        )

    # ===================================================================
    # CONFIG LOADING (now supports named workflows)
    # ===================================================================
    def load_workflow_from_config(self):
        wf_config = getattr(self.agent, "config", {}).get("workflow", {})

        # NEW: Support named workflow from config (cleanest for plugins)
        if isinstance(wf_config, dict) and "name" in wf_config:
            wf_name = wf_config["name"]
            if wf_name in self.registered_workflows:
                wf = self.registered_workflows[wf_name]
                self.workflow_steps = wf["steps"]
                self.workflow_description = wf["description"]
                self.current_workflow_name = wf_name
                logger.info(f"Loaded named workflow from config → {wf_name}")
                return
            else:
                logger.warning(
                    f"Workflow name '{wf_name}' not found. Falling back to steps."
                )

        # Backward-compatible fallback (old config style)
        self.workflow_steps = wf_config.get("steps", self.DEFAULT_WORKFLOW)
        self.workflow_description = wf_config.get(
            "description", "Default ReAct loop + persistent goal + efficient CM"
        )
        self.current_workflow_name = "custom"

        # Final filtering (only keep steps that actually have handlers)
        valid = set(self._step_handlers.keys())
        filtered = []
        for s in self.workflow_steps:
            name = s.get("name") if isinstance(s, dict) else s
            if name in valid:
                filtered.append(s)
            else:
                logger.warning(f"Unknown step '{name}' removed from workflow")

        self.workflow_steps = filtered
        logger.info(
            f"Workflow loaded: {self.current_workflow_name} — {self.workflow_description}"
        )

    # ===================================================================
    # (rest of the class is unchanged from previous version)
    # ===================================================================
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

    # ---------------- MAIN RUNNER ----------------
    async def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1212,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        # (exactly the same as previous version — no changes needed)
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

            workflow_steps = self.workflow_steps.copy()
            for step_config in workflow_steps:
                if isinstance(step_config, dict):
                    step_name: str = step_config.get("name", "")
                    overrides: Dict[str, Any] = step_config.get("overrides", {})
                else:
                    step_name = step_config
                    overrides = {}

                handler = self._step_handlers.get(step_name)
                if not handler:
                    yield ("warning", f"Unknown workflow step: {step_name}")
                    continue

                original_params = {
                    "client": self.agent.client,
                    "temperature": self.agent.temperature,
                    "max_tokens": self.agent.max_tokens,
                    "system_prompt": self.agent.system_prompt,
                }

                # Apply overrides
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

                        if event == "llm_response":
                            state.full_reply = payload.get("full_reply", "")
                            state.tool_calls = payload.get("tool_calls", [])
                            state.is_complete = payload.get("is_complete", False)
                            if state.full_reply.strip():
                                await self.agent.context_manager.add_message(
                                    "assistant", state.full_reply
                                )

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
            yield (ev, pl)
        self._log_iteration_state(iteration, state)

    async def _wf_plan_tasks(self, iteration: int, state: AgentState):
        state.phase = Phase.PLAN
        yield ("phase_changed", {"phase": state.phase.value})

        if not state.planned_tasks:
            prompt = f"""
Agent must plan a sequence of steps to accomplish the task below. Return STRICT JSON:

TASK:
{self.agent.task}

REQUIREMENTS:
- Provide an ordered list of concise steps.
- Each step should be actionable by the agent.
- Only return JSON: {{ "tasks": ["step1", "step2", ...] }}
"""
            raw_response = await self.agent.client.ask(prompt)
            try:
                plan = json.loads(str(raw_response).strip())
                state.planned_tasks = plan.get("tasks", [])
                state.current_task_index = 0
            except Exception:
                state.planned_tasks = []
                yield ("warning", "Failed to parse planning response")

        if state.planned_tasks and state.current_task_index < len(state.planned_tasks):
            next_task = state.planned_tasks[state.current_task_index]
            await self.agent.context_manager.add_message(
                "system", f"[NEXT TASK REMINDER] {next_task}"
            )
            yield ("planned_task", {"task": next_task})

    async def _wf_execute_if_tools(self, iteration: int, state: AgentState):
        if state.tool_calls:
            yield ("tool_calls", state.tool_calls)
            async for ev, pl in self._execute_tools(state.tool_calls, iteration, state):
                yield (ev, pl)
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
        if iteration == 1 and not state.tool_calls:
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
        if iteration <= 10 or not self._has_no_tools_recently(state, 5):
            return
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
            reserved_for_output=2048,
            system_prompt=self._build_objective_reminder(),
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
        state.phase = Phase.EXECUTE
        yield ("phase_changed", {"phase": state.phase.value})

        for call in tool_calls or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}

            sig = (name, json.dumps(args, sort_keys=True))
            if sig in self.agent.executed_signatures:
                continue
            self.agent.executed_signatures.add(sig)

            executor = self.agent.manager.get_executor(name)
            if not executor and name != "browse_tools":
                yield ("tool_not_found", {"name": name, "action": "auto_browse"})
                browse_executor = self.agent.manager.get_executor("browse_tools")
                if browse_executor:
                    try:
                        search_result = await browse_executor(query=name, limit=8)
                        yield (
                            "tool_search_result",
                            {
                                "query": name,
                                "result": search_result,
                                "loaded_tools": search_result.get("loaded_tools", []),
                            },
                        )
                        executor = self.agent.manager.get_executor(name)
                        if executor:
                            yield (
                                "info",
                                {"message": f"Tool '{name}' auto-loaded and ready"},
                            )
                        else:
                            yield (
                                "tool_skipped",
                                {
                                    "name": name,
                                    "reason": "still_not_found_after_search",
                                },
                            )
                            continue
                    except Exception as exc:
                        yield ("tool_search_failed", {"query": name, "error": str(exc)})
                        yield (
                            "tool_skipped",
                            {"name": name, "reason": "search_failed"},
                        )
                        continue
                else:
                    yield (
                        "tool_skipped",
                        {"name": name, "reason": "no_browser_available"},
                    )
                    continue

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

                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                logger.info(f"ConfirmationRequired: {name}")
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

            self.agent.tool_results.append(
                {"name": name, "result": result, "args": args}
            )

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
        inv_state = self.agent.context_manager.investigation_state

        prompt = f"""
Agent is stuck after {iteration} iterations.

TASK:
{self.agent.task}

LIVE CONTEXT SUMMARY:
{summary}

INVESTIGATION STATE:
{json.dumps(inv_state, indent=2)}

CURRENT WORKFLOW STEPS:
{json.dumps(self.workflow_steps, indent=2)}

Return valid JSON ONLY:
{{
    "reason": "why stuck",
    "next_step": "tool" | "llm" | "finish",
    "tool_name": "optional",
    "arguments": {{}},
    "new_subtask": "optional",
    "workflow_adjustments": {{...}}
}}
"""
        raw_response = await self.agent.client.ask(prompt)
        try:
            decision = json.loads(str(raw_response).strip())
        except Exception:
            decision = {"reason": "parse failed", "next_step": "llm"}

        if decision.get("new_subtask"):
            self.agent.context_manager.add_subtask(decision["new_subtask"])
        if decision.get("reason"):
            self.agent.context_manager.add_finding(f"Reflection: {decision['reason']}")

        await self.agent.context_manager.add_message(
            "system", f"[REFLECTION]\n{json.dumps(decision, indent=2)}"
        )
        await self.agent.context_manager.add_external_content(
            source_type="reflection",
            content=json.dumps(decision, indent=2),
            metadata={"iteration": iteration},
        )

        state.reflection_count += 1
        if state.reflection_count > getattr(
            self.agent, "max_reflections", 5
        ) and self._has_no_tools_recently(state, 5):
            yield ("warning", f"Stuck after {state.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            self._log_iteration_state(iteration, state)
            return

        adjustments = decision.get("workflow_adjustments", {})
        if isinstance(adjustments, dict):
            if isinstance(adjustments.get("reorder_steps"), list):
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
