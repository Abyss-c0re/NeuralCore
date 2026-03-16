from typing import Any, Dict, Optional, List
from neuralcore.actions.actions import Action, ActionSet
from neuralcore.actions.registry import ActionRegistry


class DynamicActionManager:
    def __init__(self, registry: ActionRegistry):
        self.registry = registry
        self.current_set = ActionSet("DynamicCore")
        self._loaded = set()

    # Delegate all important ActionSet methods
    def get_llm_tools(self) -> List[Dict[str, Any]]:
        return self.current_set.get_llm_tools()

    def get_executor(self, name: str) -> Optional[Action]:
        action = self.current_set.by_name.get(name)
        if action:
            return action  # always the Action object
        return None

    def add(self, action: Action) -> None:
        self.current_set.add(action)

    def load_tools(self, tool_names: List[str]):
        for name in tool_names:
            if name in self._loaded:
                continue
            if name not in self.registry.all_actions:
                continue
            action, _ = self.registry.all_actions[name]
            self.add(action)  # now works
            self._loaded.add(name)

    def unload_all(self):
        browser = self.current_set.by_name.get("browse_tools")
        self.current_set = ActionSet("DynamicCore")
        if browser:
            self.current_set.add(browser)

    # Optional: more delegation if needed
    @property
    def actions(self):
        return self.current_set.actions

    @property
    def by_name(self):
        return self.current_set.by_name
