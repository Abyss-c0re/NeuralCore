import asyncio
import json
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator

from pathlib import Path
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


class Agent:
    """
    IMPROVED PRODUCTION VERSION – dramatically faster + self-run safe.
    - Removed classify/plan/per-step tool discovery (the main slowdown).
    - Casual chat = zero extra calls (first turn, no tools → instant finish).
    - Complex tasks use the fast stream_with_tools loop directly.
    - Self-run / run tool is now safely skipped when called internally (prevents recursion/state corruption).
    - Deployment-as-tool works perfectly: casual prompts = normal chat, tasks = full agent.
    - All yields/streaming behaviour preserved exactly.
    """

    def __init__(self, agent_id: str, loader, app_root: Path):
        self.agent_id = agent_id
        self.config = loader.config.get("agents", {}).get(agent_id)
        if not self.config:
            raise ValueError(f"Agent '{agent_id}' not found in config")

        # Load client
        client_name = self.config.get("client")
        clients = get_clients()
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        self.client = clients[client_name]

        # Basic agent info
        self.name = self.config.get("name", f"Agent-{agent_id}")
        self.description = self.config.get("description", "")
        self.max_iterations = self.config.get("max_iterations", 25)
        self.max_reflections = self.config.get("max_reflections", 4)
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 12048)

        self.registry = registry
        self.manager = registry.manager
        self.context_manager = ContextManager()

        # Wrap client methods + self.run
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
        #self.registry.register_set("InternalTools", internal_tools.as_action_set())

        # --- Load tool sets for this agent from config ---
        tool_sets = self.config.get("tool_sets", [])
        load_tool_sets(loader, app_root=app_root, sets_to_load=tool_sets)
        for action_name in self.registry.all_actions:
            self.registry.manager.load_tools([action_name])

        self.registry.debug_print_all_tools()

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

    # ====================== OBJECTIVE REMINDER (simplified, no steps) ======================
    def _build_objective_reminder(self) -> str:
        return (
            f"OBJECTIVE LOCKED:\nTask: {self.task}\n"
            "Only finish when you can append exactly [FINAL_ANSWER_COMPLETE]."
        )

    # ====================== TOOL EXECUTION (self-run protection added) ======================
    async def _execute_tools(
        self, tool_calls: List[Dict], iteration: int
    ) -> AsyncIterator[Tuple[str, Any]]:
        self.phase = Phase.EXECUTE
        yield ("phase_changed", {"phase": self.phase.value})

        for call in tool_calls:
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
                # === SELF-RUN PROTECTION (deployment safety) ===
                if name in ("run", "self_run"):
                    result = "Self-run tool skipped to prevent recursion and state corruption."
                    yield (
                        "tool_skipped",
                        {"name": name, "reason": "recursion prevented"},
                    )
                else:
                    maybe = executor(**args)
                    result = await maybe if asyncio.iscoroutine(maybe) else maybe

                # FULL CONTEXT MANAGER INTEGRATION + CONFIRMATION
                await self.context_manager.record_tool_outcome(name, str(result), args)
                await self.context_manager.add_message("tool", str(result))

                yield ("tool_result", {"name": name, "result": result})

            except ConfirmationRequired as exc:
                logger.info(f"ConfirmationRequired for dangerous tool: {name}")
                yield ("needs_confirmation", {**exc.__dict__, "tool_calls": tool_calls})
                return

            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
                await self.context_manager.record_tool_outcome(name, result, args)
                yield ("tool_result", {"name": name, "result": result, "error": True})

            self.tool_results.append({"name": name, "result": result, "args": args})

            # stop guard
            stop_event = getattr(self.client, "_current_stop_event", None)
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                yield ("cancelled", f"Stop after {name}")
                return

    # ====================== FORCE REFLECTION ======================
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

    # ====================== FINAL SUMMARY ======================
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

    # ====================== PUBLIC RUN (STREAMLINED + FAST) ======================
    async def run(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        self._stop_event = stop_event or asyncio.Event()

        # Reset
        self.task = user_prompt
        self.goal = user_prompt
        self.phase = Phase.IDLE
        self.steps.clear()
        self.current_step = 0
        self.messages.clear()
        self.executed_signatures.clear()
        self.reflection_count = 0
        self.tool_results.clear()

        await self.context_manager.add_message("user", user_prompt)
        self.messages.append({"role": "user", "content": user_prompt})

        # ── MAIN EXECUTION LOOP (no extra classify/plan/browse calls) ─────────────
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            yield ("step_start", {"iteration": iteration, "phase": self.phase.value})

            if self._stop_event.is_set():
                yield ("cancelled", "User stop")
                return

            dynamic_system = (
                (system_prompt or "You are a powerful terminal AI assistant.")
                + "\n\n"
                + self._build_objective_reminder()
            )

            messages = await self.context_manager.provide_context(
                query="",
                max_input_tokens=self.max_tokens,
                reserved_for_output=2048,
                system_prompt=dynamic_system,
            )

            queue = await self.client.stream_with_tools(
                messages=messages,
                tools=self.manager.get_llm_tools(),
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
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

            full_reply = text_buffer.strip()
            is_complete = "[FINAL_ANSWER_COMPLETE]" in full_reply
            if is_complete:
                full_reply = full_reply.replace("[FINAL_ANSWER_COMPLETE]", "").strip()

            if not tool_calls:
                await self.context_manager.add_external_content(
                    "assistant_output", full_reply, {"iteration": iteration}
                )
                await self.context_manager.add_message("assistant", full_reply)
                self.messages.append({"role": "assistant", "content": full_reply})

                # Casual chat = instant finish (zero extra calls)
                if is_complete or (iteration == 1 and len(self.tool_results) == 0):
                    self.phase = Phase.FINALIZE
                    yield ("phase_changed", {"phase": self.phase.value})
                    async for ev, pl in self._generate_final_summary():
                        yield (ev, pl)
                    return

                # Stuck detection
                if iteration > 1:
                    async for ev, pl in self._force_reflection(iteration):
                        yield (ev, pl)
                continue

            # Tool path
            yield ("tool_calls", tool_calls)
            async for ev, pl in self._execute_tools(tool_calls, iteration):
                yield (ev, pl)
                if ev in ("needs_confirmation", "cancelled"):
                    return

        # Max iterations fallback
        async for ev, pl in self._generate_final_summary():
            yield (ev, pl)
