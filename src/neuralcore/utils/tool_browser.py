from neuralcore.actions.actions import Action
from neuralcore.actions.registry import ActionRegistry
from neuralcore.actions.manager import DynamicActionManager


class ToolBrowser(Action):
    def __init__(self, registry: ActionRegistry, manager: DynamicActionManager):
        # We'll override get_description() later to make it dynamic
        super().__init__(
            name="browse_tools",
            description="initial placeholder — will be dynamic",
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "Just the current action — short, no file names, no content, no details. "
                        "Examples: edit file  •  create folder  •  run command  •  send email  •  search web"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max tools to consider (default 6–8)",
                    "default": 8,
                },
            },
            executor=self._search,
            required=["query"],
        )

        self.registry = registry
        self.manager = manager
        self._attempt_count = 0
        self._max_attempts = 5

        manager.current_set.add(self)

    def get_description(self) -> str:
        """Override to provide dynamic description with current attempt count"""
        attempts_used = self._attempt_count
        attempts_left = max(0, self._max_attempts - attempts_used)
        
        remaining = (
            f" ({attempts_left} attempt{'s' if attempts_left != 1 else ''} left)"
            if attempts_left > 0
            else " (no more attempts left)"
        )

        return (
            f"Find tools for the current step. "
            f"Call multiple times with different short action phrases to build your toolkit. "
            f"Each call adds good matches.{remaining}\n\n"
            "Use only the action — never include file paths, URLs, content or explanations."
        )

    async def _search(self, query: str, limit: int = 8):
        self._attempt_count += 1
        attempts_left = max(0, self._max_attempts - self._attempt_count)

        results = self.registry.search(query, limit=limit * 3)

        if not results:
            msg = "No tools matched."
            if attempts_left > 0:
                msg += f" {attempts_left} attempt{'s' if attempts_left > 1 else ''} left."
            return {
                "status": "no_matches",
                "attempts_used": self._attempt_count,
                "attempts_left": attempts_left,
                "message": msg + " Try shorter action phrase.",
            }

        results.sort(key=lambda x: x[0], reverse=True)

        to_load = results[: min(4, len(results))]
        loaded_names = [action.name for _, action, _ in to_load]
        self.manager.load_tools(loaded_names)

        return {
            "status": "success",
            "attempts_used": self._attempt_count,
            "attempts_left": attempts_left,
            "loaded_tools": loaded_names,
            "matching_tools": [
                {
                    "name": action.name,
                    "description": action.description,
                    "category": set_name,
                    "confidence": round(score, 2),
                }
                for score, action, set_name in to_load
            ],
            "total_found": len(results),
            "message": (
                f"Added {len(loaded_names)} tool{'s' if len(loaded_names) != 1 else ''}. "
                f"{attempts_left} attempt{'s' if attempts_left != 1 else ''} left.\n"
                "Need better match? Call again with different action phrase."
            ),
        }