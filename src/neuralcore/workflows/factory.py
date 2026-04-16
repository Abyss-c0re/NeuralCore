from typing import Any, Callable, Dict, List, Optional, Union

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class WorkflowFactory:
    """
    Standalone factory for workflows, steps, conditions, loops.
    Completely abstract — used by ConfigLoader and decorators.
    """

    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Optional[Callable]] = {}
        self.conditions: Dict[str, Callable] = {}
        self.loops: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        logger.debug("WorkflowFactory initialized")

    # ===================================================================
    # REGISTRATION METHODS (moved from old Workflow class)
    # ===================================================================
    def register_workflow(
        self,
        name: str,
        description: str,
        steps: List[Union[str, Dict[str, Any]]],
        hidden_toolsets: Optional[Union[str, List[str]]] = None,
    ):
        self.workflows[name] = {
            "description": description,
            "steps": steps.copy(),
            "hidden_toolsets": hidden_toolsets,
        }
        logger.info(f"✅ Workflow registered: '{name}'")

    def register_step(self, name: str, handler: Callable):
        self.handlers[name] = handler
        logger.info(f"✅ Step registered: '{name}'")

    def register_custom_condition(
        self, name: str, handler: Callable, description: str = ""
    ):
        self.conditions[name] = handler
        logger.info(f"✅ Condition registered: '{name}'")

    def register_loop(
        self,
        name: str,
        handler: Callable,
        max_iterations: Optional[int] = None,
        break_condition: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.loops[name] = {
            "handler": handler,
            "max_iterations": max_iterations,
            "break_condition": break_condition,
            "description": description or f"Loop: {name}",
        }
        logger.info(f"✅ Loop registered: '{name}' (max={max_iterations or '∞'})")

    # Getters for engine
    def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        return self.workflows.get(name)

    def get_step_handler(self, name: str) -> Optional[Callable]:
        return self.handlers.get(name)

    def get_loop_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self.loops.get(name)

    def get_condition(self, name: str) -> Optional[Callable]:
        return self.conditions.get(name)

    def list_workflows(self) -> List[str]:
        return list(self.workflows.keys())
