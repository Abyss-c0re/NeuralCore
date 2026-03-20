import asyncio
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from pathlib import Path
import yaml

from neuralcore.actions.manager import registry
from neuralcore.cognition.memory import ContextManager
from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger
from neuralcore.utils.tool_loader import load_tool_sets
from neuralcore.core.client_factory import get_clients

logger = Logger.get_logger()


class Phase(str, Enum):
    IDLE = "idle"
    EXECUTE = "execute"
    REFLECT = "reflect"
    FINALIZE = "finalize"
    DECISION = "decision"


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

    def __init__(
        self, agent_id: str, loader, app_root: Path, config_file: Optional[Path] = None
    ):
        self.agent_id = agent_id
        self.loader = loader
        self.app_root = app_root

        # Load YAML config if provided, fallback to loader
        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                full_cfg = yaml.safe_load(f)
            self.config = full_cfg.get("agents", {}).get(agent_id, {})
        else:
            self.config = loader.config.get("agents", {}).get(agent_id, {})

        if not self.config:
            raise ValueError(f"Agent '{agent_id}' not found in config")

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
        self.max_reflections = self.config.get("max_reflections", 4)
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 12048)

        # Registry & context
        self.registry = registry
        self.manager = registry.manager
        self.context_manager = ContextManager()

        # Wrap client methods + run
        methods = [
            getattr(self.client, m)
            for m in dir(self.client)
            if callable(getattr(self.client, m)) and not m.startswith("_")
        ]
        methods.append(self.run)

        from neuralcore.utils.llm_tools import InternalTools

        internal_tools = InternalTools(
            client=self.client,
            description="Agent internal methods + run workflow",
            methods=methods,
        )
        self.registry.register_set("InternalTools", internal_tools.as_action_set())

        # Load agent tool sets
        tool_sets = self.config.get("tool_sets", [])
        load_tool_sets(loader, app_root=app_root, sets_to_load=tool_sets)
        for action_name in self.registry.all_actions:
            self.registry.manager.load_tools([action_name])
        self.registry.debug_print_all_tools()

        # Agent state
        self.task: str = ""
        self.goal: str = ""
        self.phase: Phase = Phase.IDLE
        self.steps: List[str] = []
        self.current_step: int = 0
        self.executed_signatures: set[tuple] = set()
        self.reflection_count = 0
        self.tool_results: List[Dict] = []
        self.messages: List[Dict] = []
        self._stop_event: Optional[asyncio.Event] = None

        # Load workflow
        self._load_workflow()

    # ---------------- Human-readable workflow ----------------
    def _load_workflow(self):
        # Ensure the workflow comes from the YAML entry matching self.name
        agents_cfg = getattr(self.loader, "config", {}).get("agents", {})

        # Find agent config by name
        wf_config = None
        for agent_id, agent_cfg in agents_cfg.items():
            if agent_cfg.get("name") == self.name:
                wf_config = agent_cfg.get("workflow", {})
                break

        # Fallback if not found (should rarely happen)
        if wf_config is None:
            wf_config = self.config.get("workflow", {})

        # Load workflow steps and description
        self.workflow_steps = wf_config.get("steps", self.DEFAULT_WORKFLOW)
        self.workflow_description = wf_config.get("description", "Default ReAct loop")

        # Step handlers mapping
        self._step_handlers = {
            "llm_stream_with_tools": self._wf_llm_stream,
            "execute_tools_if_present": self._wf_execute_if_tools,
            "check_if_complete_or_casual_chat": self._wf_check_complete,
            "force_reflection_if_stuck": self._wf_reflect_if_stuck,
            "decide_after_reflection": self._wf_decide_after_reflection,
            "safety_max_iterations": self._wf_safety_fallback,
        }

    def _build_objective_reminder(self) -> str:
        return f"OBJECTIVE LOCKED:\nTask: {self.task}\nOnly finish when you can append exactly [FINAL_ANSWER_COMPLETE]."

    # ---------------- Step methods ----------------
    async def _wf_llm_stream(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        async for ev, pl in self._llm_stream_with_tools(iteration):
            yield (ev, pl)
            if ev == "llm_response":
                yield ("llm_response", pl)

    async def _wf_execute_if_tools(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        if state.get("tool_calls"):
            yield ("tool_calls", state["tool_calls"])
            async for ev, pl in self._execute_tools(state["tool_calls"], iteration):
                yield (ev, pl)

    async def _wf_check_complete(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        if state.get("is_complete") or (iteration == 1 and len(self.tool_results) == 0):
            self.phase = Phase.FINALIZE
            yield ("phase_changed", {"phase": self.phase.value})
            async for ev, pl in self._generate_final_summary():
                yield (ev, pl)
            yield ("finish", {"reason": "complete"})

    async def _wf_reflect_if_stuck(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        if iteration > 1 and not state.get("tool_calls"):
            async for ev, pl in self._force_reflection(iteration):
                yield (ev, pl)

    async def _wf_decide_after_reflection(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        if self.reflection_count > self.max_reflections:
            # Example decision logic: configurable in YAML or prompt
            # Stop or continue – here we stop by default
            yield ("decision_point", {"message": "Reflection limit reached, stopping"})
            yield ("finish", {"reason": "reflection_failed"})

    async def _wf_safety_fallback(
        self, iteration: int, state: dict
    ) -> AsyncIterator[Tuple[str, Any]]:
        if iteration >= self.max_iterations:
            async for ev, pl in self._generate_final_summary():
                yield (ev, pl)
            yield ("finish", {"reason": "max_iterations"})

    # ---------------- Main run orchestrator ----------------
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
        self.messages.clear()
        self.executed_signatures.clear()
        self.reflection_count = 0
        self.tool_results.clear()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        await self.context_manager.add_message("user", user_prompt)
        self.messages.append({"role": "user", "content": user_prompt})

        iteration = 0
        state = {"tool_calls": None, "full_reply": "", "is_complete": False}

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

                async for event, payload in handler(iteration, state):
                    yield (event, payload)
                    if event == "llm_response":
                        state = payload
                    if event in ("needs_confirmation", "cancelled", "finish"):
                        return

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
        tool_calls = None
        try:
            async for kind, payload in self.client._drain_queue(queue):
                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)
                elif kind == "tool_delta":
                    yield ("tool_call_delta", payload)
                elif kind == "finish":
                    tool_calls = payload.get("tool_calls")
                    break
                elif kind == "error":
                    yield ("error", payload)
                    return
        except asyncio.CancelledError:
            yield ("cancelled", "Task cancelled")
            return

        state = {
            "full_reply": text_buffer.strip(),
            "tool_calls": tool_calls,
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

                await self.context_manager.record_tool_outcome(name, str(result), args)
                await self.context_manager.add_message("tool", str(result))
                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                logger.info(f"ConfirmationRequired: {name}")
                yield ("needs_confirmation", {**exc.__dict__, "tool_calls": tool_calls})
                return
            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                await self.context_manager.record_tool_outcome(name, result, args)
                yield ("tool_result", {"name": name, "result": result, "error": True})

            self.tool_results.append({"name": name, "result": result, "args": args})

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
            f"{m.get('role')}: {m.get('content', '')[:400]}"
            for m in ctx
            if m.get("content")
        )[:7000]

        raw_response = await self.client.ask(
            f"Agent ran {iteration} iterations.\n"
            f"Task: {self.task}\n"
            f"Goal: {self.goal}\n\n"
            f"Recent context:\n{context_text}\n\n"
            "No tools last turn. Suggest EXACT next action."
        )
        summary = str(raw_response).strip()

        new_query = f"SELF-REFLECTION (iter {iteration}):\n{summary}\nContinue."
        await self.context_manager.add_message("user", new_query)
        await self.context_manager.add_external_content(
            source_type="reflection",
            content=summary,
            metadata={"iteration": iteration, "type": "stuck_detection"},
        )

        self.reflection_count += 1
        if self.reflection_count > self.max_reflections:
            yield ("warning", f"Stuck after {self.reflection_count} reflections")
            yield ("finish", {"reason": "reflection_stuck"})
            return

        yield ("reflection_triggered", summary)

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
