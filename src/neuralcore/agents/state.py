from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time
from neuralcore.utils.prompt_builder import PromptBuilder
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


@dataclass
class AgentState:
    # ==================== Core Identity & Goal ====================
    agent_id: str = ""
    task: str = ""
    goal: str = ""
    current_role: str = "general_assistant"
    current_task: str = ""
    current_workflow: str = "default"

    # ==================== Execution State ====================
    phase: Any = None
    status: str = "idle"
    is_complete: bool = False
    goal_achieved: bool = False

    # ==================== ReAct / Tool Execution ====================
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_reply: str = ""
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    executed_functions: List[Dict[str, Any]] = field(default_factory=list)
    last_tool_success: Optional[Dict[str, Any]] = None

    # ==================== Loaded Tools Tracking ====================
    loaded_tools: List[str] = field(default_factory=list)

    # ==================== NEW: FindTool Tracking ====================
    findtool_call_count: int = 0
    last_findtool_loop: int = -1

    # ==================== Planning & Orchestration ====================
    planned_tasks: List[str] = field(default_factory=list)
    task_expected_outcomes: List[str] = field(default_factory=list)
    current_task_index: int = 0
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)
    task_dependencies: Dict[int, List[int]] = field(default_factory=dict)

    # ==================== Sub-agent & Hub Coordination ====================
    sub_task_ids: List[str] = field(default_factory=list)
    task_id_map: Dict[int, str] = field(default_factory=dict)
    sub_agent_results: Dict[str, Any] = field(default_factory=dict)
    active_sub_agents: List[str] = field(default_factory=list)

    # ==================== Messaging & Inter-agent Communication ====================
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    message_count: int = 0
    last_message_time: float = 0.0

    # ==================== Waiting & Pausing ====================
    waiting: bool = False
    wait_type: Optional[str] = None
    wait_start_time: Optional[float] = None
    wait_target: Optional[str] = None
    wait_prompt: str = ""
    wait_timeout: Optional[float] = None
    wait_completed: bool = False
    last_wait_event: Optional[Dict[str, Any]] = None

    # ==================== Human-in-the-Loop (Confirmation) ====================
    needs_approval: bool = False
    pending_approval_prompt: str = ""
    last_confirmation_request: Optional[Dict[str, Any]] = (
        None  # ← NEW: required for re-execution
    )

    # ==================== History & Debugging ====================
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    complex_reason: str = ""

    # ==================== Control & Safety ====================
    last_error: Optional[str] = None
    error_count: int = 0

    # ==================== Timing & Metrics ====================
    start_time: float = field(default_factory=time.time)
    loop_count: int = 0
    total_tool_calls: int = 0
    empty_loops: int = 0
    action_restarts: int = 0

    # ==================== Loop Control Signals (NEW - Option 3) ====================
    # Generic inbox for engine-level loop commands (restart / pause / wait / resume / break).
    # 100% abstract — used by every @workflow.loop via WorkflowEngine.
    pending_loop_signals: List[Dict[str, Any]] = field(default_factory=list)

    # ==================== Properties ====================
    @property
    def current_task_name(self) -> Optional[str]:
        tasks: List[str] = self.planned_tasks
        idx: int = self.current_task_index
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            return tasks[idx]
        return None

    @property
    def has_sub_tasks(self) -> bool:
        return len(self.planned_tasks) > 0

    @property
    def goal_reached(self) -> bool:
        return self.goal_achieved or self.is_complete

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def wait_elapsed(self) -> Optional[float]:
        if self.wait_start_time is None:
            return None
        return round(time.time() - self.wait_start_time, 1)

    # ==================== FindTool Helpers ====================
    def record_findtool_call(self) -> None:
        self.findtool_call_count += 1
        self.last_findtool_loop = self.loop_count
        logger.debug(
            f"AgentState → FindTool recorded (total={self.findtool_call_count}, loop={self.loop_count})"
        )

    def clear_findtool_tracking(self) -> None:
        self.findtool_call_count = 0
        self.last_findtool_loop = -1
        logger.debug("AgentState → FindTool tracking cleared")

    # ==================== Loaded Tools Helpers ====================
    def update_loaded_tools(self, tools: List[str]) -> None:
        self.loaded_tools = [t for t in tools if t]
        logger.debug(f"AgentState → loaded_tools updated: {self.loaded_tools}")

    def clear_loaded_tools(self) -> None:
        self.loaded_tools.clear()
        logger.debug("AgentState → loaded_tools cleared")

    # ==================== NEW: Loop Signal Helpers (abstract) ====================
    def add_loop_signal(
        self,
        signal: str,
        target_loop: Optional[str] = None,
        reason: str = "",
        wait_config: Optional[dict] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Add a generic loop-control signal (restart / pause / wait / resume / break)."""
        sig = {
            "signal": signal,
            "target_loop": target_loop,
            "reason": reason or f"Signal '{signal}' issued",
            "wait_config": wait_config,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        self.pending_loop_signals.append(sig)
        logger.debug(
            f"AgentState → loop signal added: {signal} (target={target_loop or 'current'})"
        )

    def clear_pending_loop_signals(self) -> None:
        """Clear all pending signals (called after processing)."""
        count = len(self.pending_loop_signals)
        self.pending_loop_signals.clear()
        if count:
            logger.debug(f"AgentState → cleared {count} pending loop signal(s)")

    # ==================== NEW: State Validation ====================
    def validate_state_integrity(self) -> List[str]:
        """Validate that state has all required structures for multi-step execution."""
        warnings = []

        if len(self.planned_tasks) != len(self.task_expected_outcomes):
            warnings.append(
                f"planned_tasks ({len(self.planned_tasks)}) and task_expected_outcomes ({len(self.task_expected_outcomes)}) length mismatch"
            )

        if not isinstance(self.task_dependencies, dict):
            warnings.append("task_dependencies is not a dict")
            self.ensure_dependencies_structure()

        if not isinstance(self.task_tool_assignments, dict):
            warnings.append("task_tool_assignments is not a dict")
            self.task_tool_assignments = {}

        if self.current_task_index < 0 or (
            self.planned_tasks and self.current_task_index >= len(self.planned_tasks)
        ):
            warnings.append(
                f"current_task_index {self.current_task_index} out of bounds (0-{len(self.planned_tasks) - 1 if self.planned_tasks else 0})"
            )
            if self.planned_tasks:
                self.current_task_index = min(
                    max(0, self.current_task_index), len(self.planned_tasks) - 1
                )

        if self.planned_tasks and not self.task_expected_outcomes:
            warnings.append(
                "planned_tasks exist but task_expected_outcomes is empty — planning may be incomplete"
            )

        return warnings

    # ==================== Core Methods ====================
    def reset_for_new_task(self, new_task: str = "", new_goal: str = "") -> None:
        logger.info(f"Resetting AgentState for new task: '{new_task[:100]}...'")

        self.planned_tasks.clear()
        self.task_expected_outcomes.clear()
        self.task_tool_assignments.clear()
        self.task_dependencies.clear()
        self.sub_task_ids.clear()
        self.task_id_map.clear()
        self.sub_agent_results.clear()
        self.active_sub_agents.clear()
        self.pending_messages.clear()

        self.phase = None
        self.status = "idle"
        self.current_task_index = 0
        self.is_complete = False
        self.goal_achieved = False
        self.tool_calls = None
        self.full_reply = ""
        self.tool_results.clear()
        self.executed_functions.clear()
        self.last_tool_success = None
        self.iteration_history.clear()

        self.complex_reason = ""
        self.waiting = False
        self.wait_type = None
        self.wait_start_time = None
        self.wait_target = None
        self.wait_prompt = ""
        self.wait_timeout = None
        self.wait_completed = False
        self.last_wait_event = None

        # ==================== Reset Human-in-the-Loop ====================
        self.needs_approval = False
        self.pending_approval_prompt = ""
        self.last_confirmation_request = None

        self.last_error = None
        self.error_count = 0
        self.loop_count = 0
        self.total_tool_calls = 0
        self.empty_loops = 0
        self.action_restarts = 0
        self.message_count = 0
        self.last_message_time = 0.0

        self.clear_loaded_tools()
        self.clear_findtool_tracking()
        self.clear_pending_loop_signals()  # ← NEW: clear loop signals on reset

        self.start_time = time.time()

        self.task = new_task
        self.goal = new_goal or new_task
        self.current_task = ""
        self.current_workflow = "default"

        logger.debug(f"AgentState reset complete. Goal: '{self.goal[:80]}...'")

    def add_tool_result(
        self, tool_name: str, result: Any, success: bool = True
    ) -> None:
        entry: Dict[str, Any] = {
            "name": tool_name,
            "result": result,
            "success": success,
            "timestamp": time.time(),
        }
        self.tool_results.append(entry)
        self.executed_functions.append({"name": tool_name, "success": success})

        level = logger.info if success else logger.warning
        level(f"Tool '{tool_name}' executed → success={success}")

        if not success:
            logger.debug(f"Tool failure details: {result}")

    def add_executed_function(
        self, function_name: str, args: Optional[Dict[str, Any]] = None
    ) -> None:
        entry: Dict[str, Any] = {
            "name": function_name,
            "args": args or {},
            "timestamp": time.time(),
        }
        self.executed_functions.append(entry)
        logger.debug(f"Function recorded: {function_name}")

    def increment_loop(self) -> None:
        self.loop_count += 1
        if self.loop_count % 5 == 0:
            logger.info(
                f"Agent loop count: {self.loop_count} | duration: {self.duration:.1f}s"
            )

    def increment_tool_call(self) -> None:
        self.total_tool_calls += 1
        logger.debug(f"Total tool calls: {self.total_tool_calls}")

    def increment_empty_loop(self) -> None:
        self.empty_loops += 1
        if self.empty_loops >= 3:
            logger.warning(
                f"Empty loop detected ({self.empty_loops} times). Possible stall."
            )

    def increment_action_restart(self) -> None:
        self.action_restarts += 1
        logger.warning(
            f"Action restart triggered. Total restarts: {self.action_restarts}"
        )

    def ensure_dependencies_structure(self) -> None:
        if not isinstance(self.task_dependencies, dict):
            self.task_dependencies = {}
        for k in list(self.task_dependencies.keys()):
            v = self.task_dependencies[k]
            if not isinstance(v, list):
                self.task_dependencies[k] = (
                    []
                    if v in (None, "null", "")
                    else [int(v)]
                    if str(v).isdigit()
                    else []
                )

    def mark_goal_achieved(self, reason: str = "") -> None:
        self.goal_achieved = True
        self.is_complete = True
        msg = f"Goal achieved: {reason}" if reason else "Goal achieved."
        logger.info(f"✅ {msg}")

    def record_error(self, error_msg: str) -> None:
        self.last_error = error_msg
        self.error_count += 1
        self.status = "error"
        logger.error(f"Agent error recorded: {error_msg}")

    def add_message(self, message: Dict[str, Any]) -> None:
        self.pending_messages.append(message)
        self.message_count += 1
        self.last_message_time = time.time()

    def start_wait(
        self,
        wait_type: str,
        prompt: str = "",
        target: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.waiting = True
        self.wait_type = wait_type
        self.wait_prompt = prompt
        self.wait_target = target
        self.wait_timeout = timeout
        self.wait_start_time = time.time()
        self.wait_completed = False
        self.last_wait_event = None
        self.status = "waiting"

        logger.info(
            f"Agent entered waiting state: {wait_type} | prompt={prompt[:100]}..."
        )

    def complete_wait(self, reason: str = "") -> None:
        self.waiting = False
        self.wait_completed = True
        self.status = "idle"
        if self.wait_start_time:
            duration = time.time() - self.wait_start_time
            logger.info(
                f"Wait completed ({self.wait_type}) after {duration:.1f}s | reason={reason}"
            )

    def get_objective_reminder(self) -> str:
        """Builds dynamic state reminder and delegates formatting to PromptBuilder."""
        parts: List[str] = []

        goal_text = self.goal or self.task or "No goal set"
        parts.append(f"Current goal: {goal_text}")

        if self.planned_tasks:
            total = len(self.planned_tasks)
            current = self.current_task_index + 1
            if total > 1:
                parts.append(f"Progress: Sub-task {current}/{total}")
                if self.current_task_name:
                    parts.append(f"Current sub-task: {self.current_task_name[:120]}...")

            if self.task_expected_outcomes and 0 <= self.current_task_index < len(
                self.task_expected_outcomes
            ):
                expected = self.task_expected_outcomes[self.current_task_index]
                if expected:
                    parts.append(f"Expected outcome for current step: {expected}")

        if self.tool_results:
            parts.append(f"Tool results available: {len(self.tool_results)}")

        if self.loaded_tools:
            parts.append(f"Currently loaded tools: {', '.join(self.loaded_tools[:8])}")

        if self.empty_loops > 0:
            parts.append(f"Empty loops: {self.empty_loops}/3")

        if self.action_restarts > 0:
            parts.append(f"Action restarts: {self.action_restarts}")

        if self.phase:
            parts.append(f"Current phase: {self.phase}")

        if self.status != "idle":
            parts.append(f"Status: {self.status}")

        if self.duration > 60:
            parts.append(f"Running for {self.duration:.0f}s")

        if self.task_dependencies:
            dep_parts: List[str] = []
            for idx, dep_list in self.task_dependencies.items():
                if isinstance(idx, int) and 0 <= idx < len(self.planned_tasks):
                    dep_names: List[str] = []
                    if isinstance(dep_list, list):
                        for d in dep_list:
                            if isinstance(d, int) and 0 <= d < len(self.planned_tasks):
                                dep_names.append(self.planned_tasks[d])
                    if dep_names:
                        dep_parts.append(
                            f"Step {idx + 1} depends on: {', '.join(dep_names[:2])}"
                        )
            if dep_parts:
                parts.append("Dependencies:\n" + "\n".join(dep_parts[:4]))

        reminder_body = "\n".join(parts)

        warnings = self.validate_state_integrity()
        if warnings:
            logger.warning(f"AgentState integrity warnings: {warnings}")

        return PromptBuilder.objective_reminder(reminder_body)

    def to_dict(self) -> Dict[str, Any]:
        """Generic, serializable snapshot — safe for NeuralHub, WebSocket, VR, etc."""
        exclude = {
            "message_queue",
            "_input_event",
            "_stop_event",
            "_background_task",
            "_task_contexts",
        }

        data: Dict[str, Any] = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in exclude
        }

        data["tool_results"] = self.tool_results[-20:]
        data["executed_functions"] = self.executed_functions[-15:]
        data["iteration_history"] = self.iteration_history[-10:]
        data["pending_messages"] = self.pending_messages[-20:]
        data["findtool_call_count"] = self.findtool_call_count
        data["last_findtool_loop"] = self.last_findtool_loop
        data["tool_calls"] = self.tool_calls or []
        data["loaded_tools"] = self.loaded_tools[:]

        data["duration"] = round(self.duration, 2)
        data["current_task_name"] = self.current_task_name
        data["has_sub_tasks"] = self.has_sub_tasks
        data["goal_reached"] = self.goal_reached

        # Waiting state
        data["waiting"] = getattr(self, "waiting", False)
        data["wait_type"] = getattr(self, "wait_type", None)
        data["wait_prompt"] = getattr(self, "wait_prompt", "")
        data["wait_target"] = getattr(self, "wait_target", None)
        data["wait_timeout"] = getattr(self, "wait_timeout", None)
        data["wait_completed"] = getattr(self, "wait_completed", False)
        data["wait_start_time"] = getattr(self, "wait_start_time", None)
        data["last_wait_event"] = getattr(self, "last_wait_event", None)

        # Human-in-the-Loop (full confirmation support)
        data["needs_approval"] = getattr(self, "needs_approval", False)
        data["pending_approval_prompt"] = getattr(self, "pending_approval_prompt", "")
        data["last_confirmation_request"] = getattr(
            self, "last_confirmation_request", None
        )

        # Loop Control Signals (NEW)
        data["pending_loop_signals"] = self.pending_loop_signals[:]

        if data.get("waiting"):
            data["wait_summary"] = {
                "type": data["wait_type"],
                "prompt": data["wait_prompt"][:200] + "..."
                if len(data.get("wait_prompt", "")) > 200
                else data.get("wait_prompt", ""),
                "elapsed": round(
                    time.time() - (data.get("wait_start_time") or time.time()), 1
                ),
                "target": data["wait_target"],
            }

        return data
