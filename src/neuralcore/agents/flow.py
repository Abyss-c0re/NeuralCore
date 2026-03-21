import asyncio
import json
from enum import Enum

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field

from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


# ---------------------------- ENUMS & STATE ----------------------------
class Phase(str, Enum):
    IDLE = "idle"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    DECISION = "decision"
    FINALIZE = "finalize"


@dataclass
class AgentState:
    tool_calls: Optional[List[Dict]] = None
    full_reply: str = ""
    is_complete: bool = False
    last_reflection_decision: Dict = field(default_factory=dict)
    executed_functions: List[Dict] = field(default_factory=list)
    iteration_history: List[Dict] = field(default_factory=list)
    phase: Phase = Phase.IDLE
    reflection_count: int = 0
    planned_tasks: List[str] = field(default_factory=list)
    current_task_index: int = 0


# =======================================================================
# WORKFLOW ENGINE – all workflow logic
# =======================================================================
class WorkflowEngine:
    DEFAULT_WORKFLOW = [
        "plan_tasks",
        "llm_stream_with_tools",
        "execute_tools_if_present",
        "verify_goal_completion",
        "check_if_complete_or_casual_chat",
        "force_reflection_if_stuck",
        "safety_max_iterations",
    ]

    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    def __init__(self, agent):
        self.agent = agent
        self.workflow_steps: list[Union[str, Dict[str, Any]]] = []
        self.workflow_description: str = ""
        self._step_handlers: Dict[str, Optional[Callable]] = {}
        self._load_workflow()

    def _load_workflow(self):
        wf_config = self.agent.config.get("workflow", {})
        self.workflow_steps = wf_config.get("steps", self.DEFAULT_WORKFLOW)
        self.workflow_description = wf_config.get(
            "description", "Default ReAct loop + persistent goal + efficient CM"
        )

        # Convert dict steps to names for handlers
        self._step_handlers = {}
        valid_step_names = {name[4:] for name in dir(self) if name.startswith("_wf_")}

        for step in self.workflow_steps:
            if isinstance(step, dict):
                step_name = step.get("name", "")
            else:
                step_name = step

            if step_name in valid_step_names:
                self._step_handlers[step_name] = getattr(self, f"_wf_{step_name}", None)
            else:
                self._step_handlers[step_name] = None
                logger.warning(
                    f"Unknown workflow step for agent {self.agent.agent_id}: {step_name}"
                )

        # Filter out invalid steps
        self.workflow_steps = [
            s
            for s in self.workflow_steps
            if (s.get("name") if isinstance(s, dict) else s) in valid_step_names
        ]

    # ---------------- HELPERS ----------------
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
                "workflow_steps_run": self.agent.steps.copy(),
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

    # ---------------- MAIN EXECUTION LOOP ----------------
    async def execute(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1212,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        self.agent._reset_state()
        self.agent.task = user_prompt
        self.agent.goal = user_prompt
        self.agent.system_prompt = system_prompt
        self.agent.temperature = temperature
        self.agent.max_tokens = max_tokens

        self.agent.context_manager.set_goal(user_prompt)
        await self.agent.context_manager.add_message("user", user_prompt)

        iteration = 0
        state = AgentState()

        while iteration < self.agent.max_iterations:
            iteration += 1
            yield (
                "step_start",
                {"iteration": iteration, "workflow": self.workflow_description},
            )

            if stop_event and stop_event.is_set():
                yield ("cancelled", "User stop")
                return

            workflow_steps = self.workflow_steps.copy()
            for step_config in workflow_steps:
                # Determine step name and overrides
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

                # Save original parameters
                original_params = {
                    "client": self.agent.client,
                    "temperature": self.agent.temperature,
                    "max_tokens": self.agent.max_tokens,
                    "system_prompt": self.agent.system_prompt,
                }

                # Apply step-level overrides
                # Apply step-level overrides
                if "client" in overrides:
                    client_name = overrides["client"]
                    if client_name in self.agent.clients:
                        self.agent.client = self.agent.clients[client_name]
                    else:
                        yield (
                            "warning",
                            f"Step override client '{client_name}' not found, using default",
                        )

                if "temperature" in overrides:
                    self.agent.temperature = overrides["temperature"]
                if "max_tokens" in overrides:
                    self.agent.max_tokens = overrides["max_tokens"]
                if "system_prompt" in overrides:
                    self.agent.system_prompt = overrides["system_prompt"]

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
                    # Restore original parameters
                    for k, v in original_params.items():
                        setattr(self.agent, k, v)

        async for ev, pl in self._generate_final_summary(state):
            yield (ev, pl)

    # ---------------- WORKFLOW STEP HANDLERS ----------------
    async def _wf_chat(self, iteration: int, state: AgentState):
        """
        Persistent chat loop using ContextManager to maintain user/assistant history.
        Keeps chatting until a message contains [TASK].
        Prevents cancellation of streaming tasks when new messages arrive.
        """
        state.phase = Phase.IDLE
        yield ("phase_changed", {"phase": state.phase.value})

        last_processed_msg: str = ""
        current_chat_task: Optional[asyncio.Task] = None

        while True:
            # Get the latest user message from context
            last_user_msg = self.agent.context_manager.get_last_user_message()
            if not last_user_msg or last_user_msg == last_processed_msg:
                await asyncio.sleep(0.1)
                continue

            last_processed_msg = last_user_msg  # mark as processed

            # Prepare the conversation context
            messages = await self.agent.context_manager.provide_context(
                query=last_user_msg,
                system_prompt=self.agent.system_prompt,
            )

            # Start the chat task if none running
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
                # If a previous task is running, optionally wait for it to finish
                await current_chat_task
                current_chat_task = asyncio.create_task(chat_task_fn())

            # Await the task and handle its reply
            try:
                reply = await current_chat_task
            except asyncio.CancelledError:
                # Only happens if the user truly cancels the workflow externally
                yield ("info", "Chat task externally cancelled")
                return

            # Stream reply line-by-line for TUI
            for line in reply.split("\n"):
                if line.strip():
                    yield ("content_delta", line + "\n")
                    await asyncio.sleep(0.01)

            # Save assistant reply in context
            await self.agent.context_manager.add_message("assistant", reply)

            # Detect task marker
            if "[TASK]" in last_user_msg:
                self.agent.task = last_user_msg.replace("[TASK]", "").strip()
                yield ("info", {"message": f"Task detected: {self.agent.task}"})
                state.phase = Phase.PLAN
                break



    async def _wf_llm_stream(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        async for ev, pl in self._llm_stream_with_tools(iteration, state):
            yield (ev, pl)
        self._log_iteration_state(iteration, state)

    async def _wf_plan_tasks(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Plan the tasks to accomplish the goal, store them in state.planned_tasks,
        and gently remind about the next task in the workflow.
        """
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

        # Gently remind agent of the next task
        if state.planned_tasks and state.current_task_index < len(state.planned_tasks):
            next_task = state.planned_tasks[state.current_task_index]
            await self.agent.context_manager.add_message(
                "system", f"[NEXT TASK REMINDER] {next_task}"
            )
            yield ("planned_task", {"task": next_task})

    async def _wf_execute_if_tools(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        if state.tool_calls:
            yield ("tool_calls", state.tool_calls)
            async for ev, pl in self._execute_tools(state.tool_calls, iteration, state):
                yield (ev, pl)
        self._log_iteration_state(iteration, state)

    async def _wf_verify_goal_completion(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
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

    async def _wf_check_complete(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
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

    async def _wf_safety_fallback(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        if iteration >= self.agent.max_iterations:
            async for ev, pl in self._generate_final_summary(state):
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})
            self._log_iteration_state(iteration, state)

    # ---------------- CORE METHODS ----------------
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
        if (
            state.reflection_count > self.agent.max_reflections
            and self._has_no_tools_recently(state, 5)
        ):
            yield ("warning", f"Stuck after {state.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            self._log_iteration_state(iteration, state)
            return

        # Apply workflow adjustments
        adjustments = decision.get("workflow_adjustments", {})
        if isinstance(adjustments, dict):
            if isinstance(adjustments.get("reorder_steps"), list):
                self.workflow_steps = [
                    s for s in adjustments["reorder_steps"] if s in self.workflow_steps
                ]
            # (insert/remove/max_iterations logic omitted for brevity – same as before)

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
