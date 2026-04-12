from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time
import asyncio

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
    phase: str = "idle"  # idle | planning | execution | reflection | verification | waiting | complete
    status: str = (
        "idle"  # idle | thinking | tool_call | waiting_approval | error | paused
    )
    is_complete: bool = False
    goal_achieved: bool = False

    # ==================== ReAct / Tool Execution ====================
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_reply: str = ""
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    executed_functions: List[Dict[str, Any]] = field(default_factory=list)

    # ==================== Planning & Orchestration ====================
    planned_tasks: List[str] = field(default_factory=list)
    current_task_index: int = 0
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)
    task_dependencies: Dict[int, Optional[str]] = field(default_factory=dict)

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
    wait_type: Optional[str] = None  # time | human | subtask | agent | condition
    wait_start_time: Optional[float] = None
    wait_target: Optional[str] = None  # task_id, agent_id(s), condition name, etc.
    wait_prompt: str = ""
    wait_timeout: Optional[float] = None
    wait_completed: bool = False
    last_wait_event: Optional[Dict[str, Any]] = None

    # ==================== Human-in-the-Loop ====================
    needs_approval: bool = False
    pending_approval_prompt: str = ""

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

    # ==================== Properties ====================
    @property
    def current_task_name(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None

    @property
    def has_sub_tasks(self) -> bool:
        return len(self.sub_task_ids) > 0

    @property
    def goal_reached(self) -> bool:
        return self.goal_achieved or self.is_complete

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def wait_elapsed(self) -> Optional[float]:
        """Safe elapsed seconds since wait started."""
        if self.wait_start_time is None:
            return None
        return round(time.time() - self.wait_start_time, 1)

    # ==================== Core Methods ====================
    def reset_for_new_task(self, new_task: str = "", new_goal: str = "") -> None:
        """Reset state for a new task or iteration cycle."""
        logger.info(f"Resetting AgentState for new task: '{new_task[:100]}...'")

        self.planned_tasks.clear()
        self.task_tool_assignments.clear()
        self.task_dependencies.clear()
        self.sub_task_ids.clear()
        self.task_id_map.clear()
        self.sub_agent_results.clear()
        self.active_sub_agents.clear()
        self.pending_messages.clear()

        self.phase = "idle"
        self.status = "idle"
        self.current_task_index = 0
        self.is_complete = False
        self.goal_achieved = False
        self.tool_calls = None
        self.full_reply = ""
        self.tool_results.clear()
        self.executed_functions.clear()
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

        self.needs_approval = False
        self.pending_approval_prompt = ""
        self.last_error = None
        self.error_count = 0
        self.loop_count = 0
        self.total_tool_calls = 0
        self.empty_loops = 0
        self.action_restarts = 0
        self.message_count = 0
        self.last_message_time = 0.0

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
        parts = []

        goal_text = self.goal or self.task or "No goal set"
        parts.append(f"Current goal: {goal_text}")

        if self.planned_tasks:
            total = len(self.planned_tasks)
            current = self.current_task_index + 1
            if total > 1:
                parts.append(f"Progress: Sub-task {current}/{total}")
                if self.current_task_name:
                    parts.append(f"Current sub-task: {self.current_task_name[:120]}...")

        if self.tool_results:
            parts.append(f"Tool results available: {len(self.tool_results)}")

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

        reminder = "\n".join(parts)

        return f"""OBJECTIVE REMINDER:
        {reminder}

        CRITICAL INSTRUCTIONS:
        - Stay focused on the current goal and sub-task.
        - If the required tool for the current action is missing, FIRST use the FindTool tool to discover and load it.
        - Only after the needed tool has been successfully loaded via FindTool should you call the actual tool.
        - When a sub-task is complete, output exactly: [FINAL_ANSWER_COMPLETE]
        - Use only verified information from tool_results when summarizing."""

    def to_dict(self) -> Dict[str, Any]:
        """Generic, serializable snapshot — safe for any transport layer (NeuralHub, WebSocket, VR, etc.)."""
        # Base exclusion list
        exclude = {
            "message_queue",
            "_input_event",
            "_stop_event",
            "_background_task",
            "_task_contexts",  # if you have any internal private dicts
        }

        # Core data from __dict__
        data: Dict[str, Any] = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in exclude
        }

        # Explicit handling of complex / list fields (limit size where needed)
        data["tool_results"] = self.tool_results[-20:]  # keep recent only
        data["executed_functions"] = self.executed_functions[-15:]
        data["iteration_history"] = self.iteration_history[-10:]
        data["pending_messages"] = self.pending_messages[-20:]

        # Waiting state - explicitly included and cleaned
        data["waiting"] = getattr(self, "waiting", False)
        data["wait_type"] = getattr(self, "wait_type", None)
        data["wait_prompt"] = getattr(self, "wait_prompt", "")
        data["wait_target"] = getattr(self, "wait_target", None)
        data["wait_timeout"] = getattr(self, "wait_timeout", None)
        data["wait_completed"] = getattr(self, "wait_completed", False)
        data["wait_start_time"] = getattr(self, "wait_start_time", None)
        data["last_wait_event"] = getattr(self, "last_wait_event", None)

        # Human-in-the-loop
        data["needs_approval"] = getattr(self, "needs_approval", False)
        data["pending_approval_prompt"] = getattr(self, "pending_approval_prompt", "")

        # Ensure tool_calls is always present and serializable
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        else:
            data["tool_calls"] = []

        # Computed fields
        data["duration"] = round(self.duration, 2)
        data["current_task_name"] = self.current_task_name
        data["has_sub_tasks"] = self.has_sub_tasks
        data["goal_reached"] = self.goal_reached

        # Optional: add a clean wait summary for external consumers
        if data["waiting"]:
            data["wait_summary"] = {
                "type": data["wait_type"],
                "prompt": data["wait_prompt"][:200] + "..."
                if len(data["wait_prompt"]) > 200
                else data["wait_prompt"],
                "elapsed": round(
                    time.time() - (data["wait_start_time"] or time.time()), 1
                ),
                "target": data["wait_target"],
            }

        return data
