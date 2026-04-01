from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time


@dataclass
class AgentState:
    # Core ReAct / execution state
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_reply: str = ""
    is_complete: bool = False
    phase: Any = None

    # Tool execution results (used by needs_reflection and error_rate_high)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Reflection & safety
    reflection_count: int = 0
    last_reflection_iteration: int = 0
    last_replan_iteration: int = 0
    last_progress_snapshot: Optional[Dict[str, Any]] = None
    last_reflection_decision: Dict[str, Any] = field(default_factory=dict)

    # Human-in-the-loop support
    needs_approval: bool = False

    # Planning & sub-agent orchestration
    planned_tasks: List[str] = field(default_factory=list)
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)
    current_task_index: int = 0

    # Sub-agent tracking
    sub_task_ids: List[str] = field(default_factory=list)
    task_id_map: Dict[int, str] = field(default_factory=dict)
    complex_reason: str = ""
    loop_count: int = 0

    # History / debugging
    executed_functions: List[Dict[str, Any]] = field(default_factory=list)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    start_time: float = field(default_factory=time.time)

    @property
    def current_task(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None

    @property
    def has_sub_tasks(self) -> bool:
        return len(self.sub_task_ids) > 0

    def reset_for_new_task(self):
        """Reset state when starting a new complex task or iteration cycle."""
        self.planned_tasks.clear()
        self.task_tool_assignments.clear()
        self.sub_task_ids.clear()
        self.task_id_map.clear()
        self.current_task_index = 0
        self.is_complete = False
        self.reflection_count = 0
        self.tool_calls = None
        self.full_reply = ""
        self.tool_results.clear()
        self.needs_approval = False
        self.executed_functions.clear()
        self.iteration_history.clear()
        self.last_reflection_decision.clear()
        self.last_progress_snapshot = None
        self.complex_reason = ""
        self.loop_count = 0
        self.start_time = time.time()

    def add_tool_result(self, tool_name: str, result: Any, success: bool = True):
        """Helper to record tool outcomes."""
        entry: Dict[str, Any] = {
            "name": tool_name,
            "result": result,
            "success": success,
            "timestamp": time.time(),
        }
        self.tool_results.append(entry)
        # Keep executed_functions in sync
        self.executed_functions.append({"name": tool_name, "success": success})

    def add_executed_function(
        self, function_name: str, args: Optional[Dict[str, Any]] = None
    ):
        """Helper for tracking executed actions."""
        entry: Dict[str, Any] = {
            "name": function_name,
            "args": args or {},
            "timestamp": time.time(),
        }
        self.executed_functions.append(entry)
