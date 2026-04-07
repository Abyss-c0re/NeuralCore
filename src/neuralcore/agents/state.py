from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time


@dataclass
class AgentState:
    # Core ReAct / execution state
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_reply: str = ""
    is_complete: bool = False          # ← used for final answer
    phase: Any = None       # e.g. "planning", "execution", "verification"

    # Hybrid chat mode
    mode: str = "task"                 # "casual" or "task"

    # Tool execution results (used by error_rate_high, etc.)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Human-in-the-loop
    needs_approval: bool = False

    # Planning & orchestration
    planned_tasks: List[str] = field(default_factory=list)
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)
    task_dependencies: Dict[int, Optional[str]] = field(default_factory=dict)
    current_task_index: int = 0

    # Sub-agent / deployment tracking (moved from Agent)
    sub_task_ids: List[str] = field(default_factory=list)
    task_id_map: Dict[int, str] = field(default_factory=dict)
    complex_reason: str = ""

    # History & debugging
    executed_functions: List[Dict[str, Any]] = field(default_factory=list)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    start_time: float = field(default_factory=time.time)

    # === NEW / RELOCATED FIELDS ===
    task: str = ""                     # Main task / user prompt
    goal: str = ""                     # Often same as task, but can be refined
    loop_count: int = 0

    # Goal tracking (clean replacement for reflection)
    goal_achieved: bool = False        # This is the key condition you want

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

    def reset_for_new_task(self):
        """Reset state when starting a new complex task or iteration cycle."""
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
        self.start_time = time.time()
        self.mode = "task"
        self.task = ""
        self.goal = ""

    def add_tool_result(self, tool_name: str, result: Any, success: bool = True):
        entry: Dict[str, Any] = {
            "name": tool_name,
            "result": result,
            "success": success,
            "timestamp": time.time(),
        }
        self.tool_results.append(entry)
        self.executed_functions.append({"name": tool_name, "success": success})

    def add_executed_function(
        self, function_name: str, args: Optional[Dict[str, Any]] = None
    ):
        entry: Dict[str, Any] = {
            "name": function_name,
            "args": args or {},
            "timestamp": time.time(),
        }
        self.executed_functions.append(entry)

    def increment_loop(self):
        self.loop_count += 1