import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml


from neuralcore.actions.manager import registry
from neuralcore.actions.decisions import add_decisions
from neuralcore.cognition.memory import ContextManager
from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger
from neuralcore.utils.tool_loader import load_tool_sets
from neuralcore.core.client_factory import get_clients

logger = Logger.get_logger()


# ---------------------------- ENUMS ----------------------------
class Phase(str, Enum):
    IDLE = "idle"
    EXECUTE = "execute"
    REFLECT = "reflect"
    FINALIZE = "finalize"
    DECISION = "decision"


# ---------------------------- AGENT STATE ----------------------------
@dataclass
class AgentState:
    tool_calls: Optional[List[Dict]] = None
    full_reply: str = ""
    is_complete: bool = False


# ---------------------------- AGENT CLASS ----------------------------
class Agent:
    """
    Production-ready Agent with full workflow engine:
    - streaming LLM output
    - tool execution + self-run protection
    - reflection if stuck
    - decision points after failure/reflection
    - human-editable workflow per agent YAML
    """

    DEFAULT_WORKFLOW = [
        "llm_stream_with_tools",
        "execute_tools_if_present",
        "check_if_complete_or_casual_chat",
        "force_reflection_if_stuck",
        "decide_after_reflection",
        "safety_max_iterations",
    ]

    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    # ---------------- INIT ----------------
    def __init__(
        self, agent_id: str, loader, app_root: Path, config_file: Optional[Path] = None
    ):
        self.agent_id = agent_id
        self.loader = loader
        self.app_root = app_root
        self.config = self._load_agent_config(agent_id, config_file)

        # Load client
        client_name = self.config.get("client", "main")
        clients = get_clients()
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        self.client = clients[client_name]

        # Basic info
        self.name = self.config.get("name", f"Agent-{agent_id}")
        self.description = self.config.get("description", "")
        self.max_iterations = self.config.get("max_iterations", 25)
        self.max_reflections = self.config.get("max_reflections", 2)
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 12048)

        # Registry & context
        self.registry = registry
        self.manager = registry.manager
        self.context_manager = ContextManager()

        # Register client + run methods as internal tools
        self._register_internal_tools()

        # Load agent tool sets
        self._load_agent_tools()

        # Agent state
        self._reset_state()

        # Load workflow
        self._load_workflow()

    # ---------------- CONFIG LOADING ----------------
    def _load_agent_config(self, agent_id: str, config_file: Optional[Path]) -> dict:
        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                full_cfg = yaml.safe_load(f)
            cfg = full_cfg.get("agents", {}).get(agent_id, {})
        else:
            cfg = getattr(self.loader, "config", {}).get("agents", {}).get(agent_id, {})
        if not cfg:
            raise ValueError(f"Agent '{agent_id}' not found in config")
        return cfg

    # ---------------- INTERNAL TOOLS ----------------
    def _register_internal_tools(self):
        from neuralcore.utils.llm_tools import InternalTools

        methods = [
            getattr(self.client, m)
            for m in dir(self.client)
            if callable(getattr(self.client, m)) and not m.startswith("_")
        ]
        methods.append(self.run)

        internal_tools = InternalTools(
            client=self.client,
            description="Agent internal methods + run workflow",
            methods=methods,
        )
        # self.registry.register_set("InternalTools", internal_tools.as_action_set())
        add_decisions(self)

    # ---------------- LOAD TOOLS ----------------
    def _load_agent_tools(self):
        tool_sets = self.config.get("tool_sets", [])
        load_tool_sets(self.loader, app_root=self.app_root, sets_to_load=tool_sets)
        for action_name in self.registry.all_actions:
            self.registry.manager.load_tools([action_name])
        self.registry.debug_print_all_tools()

    # ---------------- WORKFLOW ----------------
    def _load_workflow(self):
        agents_cfg = getattr(self.loader, "config", {}).get("agents", {})

        wf_config = None
        for agent_id, agent_cfg in agents_cfg.items():
            if agent_cfg.get("name") == self.name:
                wf_config = agent_cfg.get("workflow", {})
                break

        if wf_config is None:
            wf_config = self.config.get("workflow", {})

        self.workflow_steps = wf_config.get("steps", self.DEFAULT_WORKFLOW)
        self.workflow_description = wf_config.get("description", "Default ReAct loop")

        # Map step name to handler if exists
        self._step_handlers = {
            name: getattr(self, f"_wf_{name}", None) for name in self.workflow_steps
        }

    # ---------------- STATE RESET ----------------
    def _reset_state(self):
        self.task = ""
        self.goal = ""
        self.phase: Phase = Phase.IDLE
        self.steps: List[str] = []
        self.current_step: int = 0
        self.executed_signatures: set[tuple] = set()
        self.reflection_count = 0
        self.tool_results: List[Dict] = []
        self.messages: List[Dict] = []
        self._stop_event: Optional[asyncio.Event] = None

    # ---------------- HELPER METHODS ----------------
    def _set_phase(self, phase: Phase) -> dict:
        self.phase = phase
        return {"phase": phase.value}

    def _build_objective_reminder(self) -> str:
        return (
            f"OBJECTIVE LOCKED:\nTask: {self.task}\n"
            f"Only finish when you can append exactly {self.FINAL_ANSWER_MARKER}."
        )

    def _prepare_tool_call(self, call: Dict) -> Tuple[str, dict, tuple]:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception:
            args = {}
        sig = (name, json.dumps(args, sort_keys=True))
        return name, args, sig

    # ---------------- RUN ORCHESTRATOR ----------------

    # ---------------- WORKFLOW STEP HANDLERS ----------------
    async def _wf_llm_stream(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        async for ev, pl in self._llm_stream_with_tools(iteration):
            yield (ev, pl)

    async def _wf_execute_if_tools(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        if state.tool_calls:
            yield ("tool_calls", state.tool_calls)
            async for ev, pl in self._execute_tools(state.tool_calls, iteration):
                yield (ev, pl)

    async def _wf_check_complete(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        If no tools and first iteration, treat as casual mode and provide full reply immediately.
        """
        if iteration == 1 and not state.tool_calls:
            # Full casual reply mode
            self.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": self.phase.value})
            yield (
                "llm_response",
                {"full_reply": state.full_reply, "tool_calls": [], "is_complete": True},
            )
            yield ("finish", {"reason": "casual_complete"})
            return

        if state.is_complete:
            self.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": self.phase.value})
            async for ev, pl in self._generate_final_summary():
                yield (ev, pl)
            yield ("finish", {"reason": "complete"})

    async def _wf_reflect_if_stuck(self, iteration, state):
        if iteration > 3 and not state.tool_calls:
            async for ev, pl in self._force_reflection(iteration):
                yield (ev, pl)

                if ev == "reflection_decision":
                    next_step = pl.get("next_step")

                    if next_step == "tool":
                        tool_name = pl.get("tool_name")

                        if not tool_name:
                            yield ("warning", "Reflection chose tool but none provided")
                            return

                        args = pl.get("arguments", {})

                        state.tool_calls = [
                            {
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(args),
                                }
                            }
                        ]

                        yield (
                            "info",
                            {"message": f"[REFLECTION] Enqueued tool: {tool_name}"},
                        )
                        return

                    elif next_step == "finish":
                        yield (
                            "finish",
                            {
                                "reason": "reflection_finish",
                                "details": pl.get("reason"),
                            },
                        )
                        return

                    elif next_step == "llm":
                        await self.context_manager.add_message(
                            "system",
                            f"[REFLECTION GUIDANCE]\n{json.dumps(pl, indent=2)}",
                        )
                        return

                    else:
                        yield ("warning", f"Unknown reflection step: {next_step}")
                        return

    async def _wf_safety_fallback(
        self, iteration: int, state: AgentState
    ) -> AsyncIterator[Tuple[str, Any]]:
        if iteration >= self.max_iterations:
            async for ev, pl in self._generate_final_summary():
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})

    # ---------------- STREAMING, TOOLS, REFLECTION, SUMMARY ----------------
    # (Same as your original implementations, can be copied over with small improvements)

    async def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1212,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        self._stop_event = stop_event or asyncio.Event()
        self.task = user_prompt
        self.goal = user_prompt
        self.phase = Phase.IDLE
        self.steps.clear()
        self.current_step = 0
        self.executed_signatures.clear()
        self.reflection_count = 0
        self.tool_results.clear()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Add user message to ContextManager only
        await self.context_manager.add_message("user", user_prompt)

        iteration = 0
        state = AgentState()

        while iteration < self.max_iterations:
            iteration += 1
            yield (
                "step_start",
                {"iteration": iteration, "workflow": self.workflow_description},
            )

            if self._stop_event.is_set():
                yield ("cancelled", "User stop")
                return

            for step_name in self.workflow_steps:
                handler = self._step_handlers.get(step_name)
                if not handler:
                    yield ("warning", f"Unknown workflow step: {step_name}")
                    continue

                try:
                    async for event, payload in handler(iteration, state):
                        yield (event, payload)

                        # Update state from LLM responses
                        if event == "llm_response":
                            state.full_reply = payload.get("full_reply", "")
                            state.tool_calls = payload.get("tool_calls", [])
                            state.is_complete = payload.get("is_complete", False)

                        if event in ("needs_confirmation", "cancelled", "finish"):
                            return
                except Exception as e:
                    yield ("error", str(e))

        # Final safety net
        async for ev, pl in self._generate_final_summary():
            yield (ev, pl)

    # ---------------- Original streaming, tools, reflection, summary ----------------
    async def _llm_stream_with_tools(
        self, iteration: int
    ) -> AsyncIterator[Tuple[str, Any]]:
        self.phase = Phase.EXECUTE
        messages = await self.context_manager.provide_context(
            query="",
            max_input_tokens=self.max_tokens,
            reserved_for_output=2048,
            system_prompt=self._build_objective_reminder(),
        )

        queue = await self.client.stream_with_tools(
            messages=messages,
            tools=self.manager.get_llm_tools(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_choice="auto",
        )

        text_buffer = ""
        completed_tool_calls = []

        try:
            async for kind, payload in self.client._drain_queue(queue):
                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)

                elif kind == "tool_delta":
                    # just for UI display, do NOT execute yet
                    yield ("tool_delta", payload)

                elif kind == "tool_complete":
                    # only append fully completed tool calls
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

        state = {
            "full_reply": text_buffer.strip(),
            "tool_calls": completed_tool_calls,  # only fully complete calls here
            "is_complete": "[FINAL_ANSWER_COMPLETE]" in text_buffer,
        }

        yield ("llm_response", state)

    async def _execute_tools(
        self, tool_calls: List[Dict], iteration: int
    ) -> AsyncIterator[Tuple[str, Any]]:
        self.phase = Phase.EXECUTE
        yield ("phase_changed", {"phase": self.phase.value})

        for call in tool_calls or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}

            sig = (name, json.dumps(args, sort_keys=True))
            if sig in self.executed_signatures:
                continue
            self.executed_signatures.add(sig)

            executor = self.manager.get_executor(name)
            if not executor:
                yield ("tool_skipped", {"name": name, "reason": "no executor"})
                continue

            yield ("tool_start", {"name": name, "args": args})

            # Always define task_id
            task_id = f"{name}:{hash(json.dumps(args, sort_keys=True))}"

            # Track subtask in context manager if available
            if hasattr(self.context_manager, "add_subtask"):
                self.context_manager.add_subtask(task_id)

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

                # Record outcome
                await self.context_manager.record_tool_outcome(name, str(result), args)
                await self.context_manager.add_message("tool", str(result))

                # Mark subtask complete and add findings
                if hasattr(self.context_manager, "complete_subtask"):
                    self.context_manager.complete_subtask(task_id)
                if hasattr(self.context_manager, "add_finding"):
                    self.context_manager.add_finding(f"{name} → {str(result)[:200]}")

                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                logger.info(f"ConfirmationRequired: {name}")
                yield ("needs_confirmation", {**exc.__dict__, "tool_calls": tool_calls})
                return
            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                await self.context_manager.record_tool_outcome(name, result, args)

                # Track unknown in context
                if hasattr(self.context_manager, "add_unknown"):
                    self.context_manager.add_unknown(f"{name} failed: {str(exc)[:200]}")

                yield ("tool_result", {"name": name, "result": result, "error": True})

            self.tool_results.append({"name": name, "result": result, "args": args})

            # Handle manual stop
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", f"Stop after {name}")
                return

    async def _force_reflection(self, iteration: int) -> AsyncIterator[Tuple[str, Any]]:
        self.phase = Phase.REFLECT
        yield ("phase_changed", {"phase": self.phase.value})

        ctx = await self.context_manager.provide_context(
            query="",
            max_input_tokens=self.max_tokens,
            reserved_for_output=1024,
            system_prompt=self._build_objective_reminder(),
        )

        context_text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')[:300]}"
            for m in ctx
            if m.get("content")
        )[:6000]

        inv_state = getattr(self.context_manager, "investigation_state", {})

        raw_response = await self.client.ask(
            f"""
    Agent is stuck after {iteration} iterations.

    TASK:
    {self.task}

    INVESTIGATION STATE:
    {json.dumps(inv_state, indent=2)}

    RECENT CONTEXT:
    {context_text}

    You MUST return valid JSON ONLY:

    {{
    "reason": "why stuck",
    "next_step": "tool" | "llm" | "finish",
    "tool_name": "optional",
    "arguments": {{}},
    "new_subtask": "optional"
    }}
    """
        )

        try:
            decision = json.loads(str(raw_response).strip())
        except Exception:
            decision = {"reason": "failed_to_parse", "next_step": "llm"}

        yield ("reflection_decision", decision)

        if decision.get("new_subtask"):
            self.context_manager.add_subtask(decision["new_subtask"])

        if decision.get("reason"):
            self.context_manager.add_finding(f"Reflection: {decision['reason']}")

        await self.context_manager.add_message(
            "system", f"[REFLECTION]\n{json.dumps(decision, indent=2)}"
        )

        await self.context_manager.add_external_content(
            source_type="reflection",
            content=json.dumps(decision, indent=2),
            metadata={"iteration": iteration},
        )

        self.reflection_count += 1
        if self.reflection_count > self.max_reflections:
            yield ("warning", f"Stuck after {self.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            return

        yield ("reflection_triggered", decision)

    async def _generate_final_summary(self) -> AsyncIterator[Tuple[str, Any]]:
        self.phase = Phase.FINALIZE
        yield ("phase_changed", {"phase": self.phase.value})

        lines = [
            "# 🏁 Agent Execution Report",
            f"**Task:** {self.task[:200]}...",
            f"**Goal:** {self.goal}",
            "",
            "## 📊 ContextManager Stats",
            f"- KB items: {len(self.context_manager.knowledge_base)}",
            f"- Files checked: {len(self.context_manager.files_checked)}",
            f"- Tools executed: {len(self.context_manager.tools_executed)}",
            f"- Archived turns: {len(self.context_manager.current_topic.archived_history)}",
            "",
            "## 🛠️ Tool Results (last 10)",
        ]
        for r in self.tool_results[-10:]:
            lines.append(f"- {r['name']}: {str(r.get('result', ''))[:120]}...")

        final_text = "\n".join(lines)
        yield ("final_summary", final_text)
        yield ("finish", {"reason": "task_complete", "summary": final_text})
