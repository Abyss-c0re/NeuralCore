from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class AgentState:
    tool_calls: Optional[List[Dict]] = None
    full_reply: str = ""
    is_complete: bool = False
    last_reflection_decision: Dict = field(default_factory=dict)
    executed_functions: List[Dict] = field(default_factory=list)
    iteration_history: List[Dict] = field(default_factory=list)
    phase: Any = None

    reflection_count: int = 0
    planned_tasks: List[str] = field(default_factory=list)
    current_task_index: int = 0

    # ── NEW: Support for sub-agent task tracking ──
    sub_task_ids: List[str] = field(
        default_factory=list
    )  # List of task_ids (e.g. ["deploy_001", "deploy_002"])
    task_id_map: Dict[int, str] = field(
        default_factory=dict
    )  # Optional: step_index → task_id mapping
    last_progress_snapshot: Optional[Dict[str, Any]] = None
    last_reflection_iteration: int = 0
    last_replan_iteration: int = 0

    @property
    def current_task(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None

    # Helper properties for convenience
    @property
    def has_sub_tasks(self) -> bool:
        return len(self.sub_task_ids) > 0

    def get_sub_task_status(self, task_id: str) -> Optional[Dict]:
        """Optional helper - can be used in workflow steps if needed"""
        # This would normally be accessed via agent.sub_tasks, but we can expose it here if desired
        return None
