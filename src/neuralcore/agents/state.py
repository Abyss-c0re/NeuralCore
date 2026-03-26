from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class AgentState:
    # Core ReAct / execution state
    tool_calls: Optional[List[Dict]] = None
    full_reply: str = ""
    is_complete: bool = False
    phase: Any = None

    # Reflection & safety
    reflection_count: int = 0
    last_reflection_iteration: int = 0
    last_replan_iteration: int = 0
    last_progress_snapshot: Optional[Dict[str, Any]] = None
    last_reflection_decision: Dict[str, Any] = field(default_factory=dict)

    # Planning & sub-agent orchestration
    planned_tasks: List[str] = field(default_factory=list)
    task_tool_assignments: Dict[int, List[str]] = field(default_factory=dict)  # ← FIXED
    current_task_index: int = 0

    # Sub-agent tracking
    sub_task_ids: List[str] = field(default_factory=list)
    task_id_map: Dict[int, str] = field(default_factory=dict)  # step_index → task_id

    # History / debugging
    executed_functions: List[Dict] = field(default_factory=list)
    iteration_history: List[Dict] = field(default_factory=list)

    @property
    def current_task(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None

    @property
    def has_sub_tasks(self) -> bool:
        return len(self.sub_task_ids) > 0

    def reset_for_new_task(self):
        """Helpful utility to reset state when starting a new complex task"""
        self.planned_tasks.clear()
        self.task_tool_assignments.clear()
        self.sub_task_ids.clear()
        self.task_id_map.clear()
        self.current_task_index = 0
        self.is_complete = False
        self.reflection_count = 0
        self.tool_calls = None
        self.full_reply = ""
