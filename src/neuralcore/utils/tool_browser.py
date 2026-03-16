from src.actions.actions import Action
from src.actions.registry import ActionRegistry
from src.actions.manager import DynamicActionManager

class ToolBrowser(Action):
    def __init__(self, registry: ActionRegistry, manager: DynamicActionManager):

        super().__init__(
            name="browse_tools",
            description=(
                "Search for tools not currently loaded. "
                "Use when you need capabilities like web search, email, APIs, files, or terminal."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "Describe the tool you need",
                },
                "limit": {"type": "integer", "default": 8},
            },
            executor=self._search,
            required=["query"],
        )

        self.registry = registry
        self.manager = manager

        manager.current_set.add(self)

    async def _search(self, query: str, limit: int = 8):

        results = self.registry.search(query, limit)

        if not results:
            return {
                "status": "no_matches",
                "message": "No tools found. Try broader terms.",
            }

        # Load only the best few
        best = results[:3]
        self.manager.load_tools([a.name for a, _ in best])

        return {
            "status": "success",
            "loaded_tools": [a.name for a, _ in best],
            "matching_tools": [
                {
                    "name": a.name,
                    "description": a.description,
                    "category": set_name,
                    "parameters": a._raw_schema["parameters"],
                }
                for a, set_name in results
            ],
            "message": f"{len(best)} tools loaded and ready.",
        }