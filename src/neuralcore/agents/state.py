from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


@dataclass
class AgentState:
    # ==================== Core ReAct / Execution State ====================
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_reply: str = ""
    is_complete: bool = False          # Final answer / termination flag
    phase: Any = None        # "planning", "execution", "verification", "reflection", etc.

    # ==================== Mode ====================
    mode: str = "task"                 # "task" or "casual"

    # ==================== Tool Execution History ====================
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # ==================== Human-in-the-Loop ====================
    needs_approval: bool = False

    # ==================== Planning & Orchestration ====================
    planned_tasks: List[str] = field(default_factory=list)
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)
    task_dependencies: Dict[int, Optional[str]] = field(default_factory=dict)
    current_task_index: int = 0

    # ==================== Sub-agent / Deployment Tracking ====================
    sub_task_ids: List[str] = field(default_factory=list)
    task_id_map: Dict[int, str] = field(default_factory=dict)
    complex_reason: str = ""

    # ==================== History & Debugging ====================
    executed_functions: List[Dict[str, Any]] = field(default_factory=list)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # ==================== Timing ====================
    start_time: float = field(default_factory=time.time)

    # ==================== Task & Goal Tracking ====================
    task: str = ""                     # Original user prompt / main task
    goal: str = ""                     # Refined or clarified goal
    loop_count: int = 0
    total_tool_calls: int = 0
    empty_loops: int = 0
    action_restarts: int = 0

    # Goal achievement (preferred over just is_complete in most workflows)
    goal_achieved: bool = False

    # ==================== Properties ====================
    @property
    def current_task(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None

    @property
    def has_sub_tasks(self) -> bool:
        return len(self.sub_task_ids) > 0

    @property
    def goal_reached(self) -> bool:
        """Convenient alias used in many workflows"""
        return self.goal_achieved or self.is_complete

    @property
    def duration(self) -> float:
        """How long this agent state has been running (in seconds)"""
        return time.time() - self.start_time

    # ==================== Methods with Logging ====================
    def reset_for_new_task(self, new_task: str = "", new_goal: str = ""):
        """Reset state when starting a new complex task or iteration cycle."""
        logger.info(f"Resetting AgentState for new task. Task: '{new_task[:100]}...'")

        self.planned_tasks.clear()
        self.task_tool_assignments.clear()
        self.task_dependencies.clear()
        self.sub_task_ids.clear()
        self.task_id_map.clear()
        self.phase = None
        self.current_task_index = 0
        self.is_complete = False
        self.goal_achieved = False
        self.tool_calls = None
        self.full_reply = ""
        self.tool_results.clear()
        self.needs_approval = False
        self.executed_functions.clear()
        self.iteration_history.clear()
        self.complex_reason = ""
        self.loop_count = 0
        self.empty_loops = 0
        self.action_restarts = 0
        self.total_tool_calls = 0
        self.start_time = time.time()
        self.mode = "task"

        self.task = new_task
        self.goal = new_goal or new_task

        logger.debug(f"AgentState reset complete. Goal set to: '{self.goal[:80]}...'")

    def add_tool_result(self, tool_name: str, result: Any, success: bool = True):
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
    ):
        entry: Dict[str, Any] = {
            "name": function_name,
            "args": args or {},
            "timestamp": time.time(),
        }
        self.executed_functions.append(entry)

        logger.debug(f"Function recorded: {function_name} (args: {bool(args)})")

    def increment_loop(self):
        self.loop_count += 1
        if self.loop_count % 5 == 0:  # Log every 5 loops to avoid spam
            logger.info(f"Agent loop count: {self.loop_count} | duration: {self.duration:.1f}s")

    def increment_tool_call(self):
        self.total_tool_calls += 1
        logger.debug(f"Total tool calls: {self.total_tool_calls}")

    def increment_empty_loop(self):
        self.empty_loops += 1
        if self.empty_loops >= 3:
            logger.warning(f"Empty loop detected ({self.empty_loops} times). Possible stall.")

    def increment_action_restart(self):
        self.action_restarts += 1
        logger.warning(f"Action restart triggered. Total restarts: {self.action_restarts}")

    def mark_goal_achieved(self, reason: str = ""):
        """Explicitly mark that the main goal has been reached."""
        self.goal_achieved = True
        self.is_complete = True
        msg = f"Goal achieved: {reason}" if reason else "Goal achieved."
        logger.info(f"✅ {msg}")