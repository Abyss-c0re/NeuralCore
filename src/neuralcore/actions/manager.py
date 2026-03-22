import math
from typing import Any, Dict, List, Optional, Set, Union
from neuralcore.utils.search import fuzzy_score, keyword_score
from neuralcore.actions.actions import Action, ActionSet

from inspect import signature, _empty
from typing import get_origin

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()

PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def map_type_to_json(param_annotation):
    """Convert Python type annotation to JSON Schema type."""
    if param_annotation is _empty:
        return "string"  # default type

    # Handle basic types
    if param_annotation in PYTHON_TO_JSON_TYPE:
        return PYTHON_TO_JSON_TYPE[param_annotation]

    # Handle typing generics like list[str], dict[str, int], etc.
    origin = get_origin(param_annotation)
    if origin in PYTHON_TO_JSON_TYPE:
        return PYTHON_TO_JSON_TYPE[origin]

    # Fallback
    print(f"[Warning] Unmapped annotation {param_annotation}, defaulting to 'string'")
    return "string"


# ─────────────────────────────────────────────────────────────
# Dynamic Action Manager
# ─────────────────────────────────────────────────────────────


class DynamicActionManager:
    """Manages dynamically loaded tools for the current session."""

    def __init__(self, registry: "ActionRegistry"):
        self.registry = registry
        self.current_set = ActionSet("DynamicCore")
        self._loaded_tools: Set[str] = set()  # tool names currently active
        self._tool_to_set: Dict[str, str] = {}  # tool_name → originating set_name
        self._set_to_tools: Dict[
            str, Set[str]
        ] = {}  # set_name → {tool_names loaded from it}

        # Always keep browse_tools available
        self._persistent_tools: Set[str] = {"browse_tools"}

        logger.debug("DynamicActionManager initialized")

    def add(self, action: Action, origin_set: Optional[str] = None):
        """Add a single action to the current dynamic set."""
        if action.name in self._loaded_tools:
            logger.debug(f"Tool '{action.name}' already loaded — skipping")
            return

        self.current_set.add(action)
        self._loaded_tools.add(action.name)

        if origin_set:
            self._tool_to_set[action.name] = origin_set
            self._set_to_tools.setdefault(origin_set, set()).add(action.name)

        logger.debug(
            f"Added tool '{action.name}' to DynamicCore (origin: {origin_set or 'manual'})"
        )

    def load_tools(self, tool_names: Union[str, List[str]]):
        """Load specific tools by name (manual / one-off loading)."""
        if isinstance(tool_names, str):
            tool_names = [tool_names]

        loaded = []
        for name in tool_names:
            if name in self._loaded_tools:
                continue
            if name not in self.registry.all_actions:
                logger.warning(f"Tool '{name}' not found in registry")
                continue

            action, origin_set = self.registry.all_actions[name]
            self.add(action, origin_set=origin_set)
            loaded.append(name)

        if loaded:
            logger.info(f"Loaded {len(loaded)} individual tools: {', '.join(loaded)}")

    def load_toolsets(self, toolset_names: Union[str, List[str]]) -> int:
        """Load all actions from one or more named ActionSets."""
        if isinstance(toolset_names, str):
            toolset_names = [n.strip() for n in toolset_names.split(",") if n.strip()]

        loaded_count = 0
        loaded_tools_list = []

        for set_name in toolset_names:
            if set_name not in self.registry.sets:
                logger.warning(f"Toolset '{set_name}' not found in registry")
                continue

            action_set = self.registry.sets[set_name]
            newly_loaded = []

            for action in action_set.actions:
                if action.name in self._loaded_tools:
                    continue
                # Skip persistent tools if already loaded
                if action.name in self._persistent_tools:
                    continue

                self.add(action, origin_set=set_name)
                newly_loaded.append(action.name)
                loaded_count += 1
                loaded_tools_list.append(action.name)

            if newly_loaded:
                logger.info(
                    f"Loaded {len(newly_loaded)} tools from set '{set_name}': {', '.join(newly_loaded)}"
                )

        if loaded_count > 0:
            logger.info(f"Total new tools loaded from toolsets: {loaded_count}")

        return loaded_count

    def unload_tools(self, tool_names: Union[str, List[str], None] = None):
        """Unload specific tools by name."""
        if tool_names is None:
            self.unload_all()
            return

        if isinstance(tool_names, str):
            tool_names = [tool_names]

        removed = []
        for name in tool_names:
            if name in self._persistent_tools:
                logger.debug(f"Skipping unload of persistent tool: {name}")
                continue
            if name not in self._loaded_tools:
                continue

            self.current_set.remove_by_name(
                name
            )  # assumes ActionSet has .remove(name) method
            self._loaded_tools.remove(name)
            self._tool_to_set.pop(name, None)

            # Clean up reverse mapping
            for s, tools in self._set_to_tools.items():
                if name in tools:
                    tools.remove(name)
                    if not tools:
                        self._set_to_tools.pop(s, None)
                    break

            removed.append(name)

        if removed:
            logger.info(f"Unloaded tools: {', '.join(removed)}")

    def unload_toolsets(self, toolset_names: Union[str, List[str], None] = None):
        """Unload all tools that originated from one or more toolsets."""
        if toolset_names is None:
            self.unload_all()
            return

        if isinstance(toolset_names, str):
            toolset_names = [n.strip() for n in toolset_names.split(",") if n.strip()]

        to_unload = set()
        for set_name in toolset_names:
            if set_name in self._set_to_tools:
                to_unload.update(self._set_to_tools[set_name])

        if to_unload:
            self.unload_tools(list(to_unload))
            logger.info(f"Unloaded all tools from sets: {', '.join(toolset_names)}")

    def unload_all(self):
        """Unload all non-persistent dynamic tools."""
        to_remove = self._loaded_tools - self._persistent_tools
        if not to_remove:
            return

        for name in list(to_remove):
            self.current_set.remove_by_name(name)
            self._loaded_tools.remove(name)
            self._tool_to_set.pop(name, None)

        self._set_to_tools.clear()  # all dynamic origins gone

        # Re-add persistent tools (e.g. browse_tools)
        for p_name in self._persistent_tools:
            if p_name in self.registry.all_actions:
                action, _ = self.registry.all_actions[p_name]
                if p_name not in self._loaded_tools:
                    self.current_set.add(action)
                    self._loaded_tools.add(p_name)

        logger.info(
            f"Unloaded all dynamic tools. Retained persistent: {', '.join(self._persistent_tools)}"
        )

    def is_loaded(self, tool_name: str) -> bool:
        return tool_name in self._loaded_tools

    def get_loaded_toolsets(self) -> List[str]:
        """Return list of toolsets that currently have loaded tools."""
        return list(self._set_to_tools.keys())

    def get_tool_origin(self, tool_name: str) -> Optional[str]:
        """Return which toolset originally loaded this tool."""
        return self._tool_to_set.get(tool_name)

    # Existing properties (unchanged)
    def get_llm_tools(self) -> List[Dict[str, Any]]:
        return self.current_set.get_llm_tools()

    def get_executor(self, name: str) -> Optional[Action]:
        return self.current_set.by_name.get(name)

    @property
    def actions(self):
        return self.current_set.actions

    @property
    def by_name(self):
        return self.current_set.by_name


# ─────────────────────────────────────────────────────────────
# Tool Browser
# ─────────────────────────────────────────────────────────────
class ToolBrowser(Action):
    def __init__(self, registry: "ActionRegistry", manager: DynamicActionManager):
        super().__init__(
            name="browse_tools",
            description=(
                "Find and load tools not currently available. "
                "Provide a short action query (e.g. 'open file', 'edit file')."
            ),
            parameters={
                "query": {"type": "string", "description": "Short action phrase"},
                "limit": {"type": "integer", "default": 8},
            },
            executor=self._search,
            required=["query"],
        )
        self.registry = registry
        self.manager = manager
        manager.current_set.add(self)
        logger.debug(" ToolBrowser added to DynamicCore")

    async def _search(self, query: str, limit: int = 8):
        logger.debug(f" ToolBrowser search: query='{query}', limit={limit}")
        results = self.registry.search(query, limit)
        if not results:
            logger.debug(" No matching tools found")
            return {
                "status": "no_matches",
                "message": "No tools found. Try broader terms.",
            }

        best = results[:3]
        self.manager.load_tools([a.name for a, _ in best])
        logger.debug(f" ToolBrowser loaded {len(best)} tools")

        return {
            "status": "success",
            "loaded_tools": [a.name for a, _ in best],
            "matching_tools": [
                {
                    "name": a.name,
                    "description": a.description,
                    "category": set_name,
                    "parameters": getattr(a, "_raw_schema", {}).get("parameters", {}),
                }
                for a, set_name in results
            ],
            "message": f"{len(best)} tools loaded and ready.",
        }


# ─────────────────────────────────────────────────────────────
# Main Action Registry
# ─────────────────────────────────────────────────────────────
class ActionRegistry:
    """Central searchable registry of all tools."""

    def __init__(self):
        self.sets: Dict[str, ActionSet] = {}
        self.all_actions: Dict[str, tuple[Action, str]] = {}
        self._index = []

        # Dynamic manager
        self.manager = DynamicActionManager(self)

        logger.debug(" ActionRegistry initialized")

        # Load ToolBrowser
        ToolBrowser(self, self.manager)
        logger.debug(" ToolBrowser registered")

    def register_set(self, name: str, action_set: ActionSet):
        logger.debug(f"Registering set '{name}' with {len(action_set.actions)} actions")
        if name in self.sets:
            raise ValueError(f"Set {name} already exists")
        self.sets[name] = action_set

        for action in action_set.actions:
            self._add_to_index(action, name)

    def register_action(self, action: Action, set_name: str):
        if set_name in self.sets:
            self.sets[set_name].add(action)
        else:
            aset = ActionSet(name=set_name)
            aset.add(action)
            self.register_set(set_name, aset)
        self._add_to_index(action, set_name)

    def _add_to_index(self, action: Action, set_name: str):
        self.all_actions[action.name] = (action, set_name)
        searchable = " ".join(
            [action.name, action.description]
            + getattr(action, "tags", [])
            + getattr(action, "aliases", [])
        ).lower()
        self._index.append({"action": action, "set": set_name, "text": searchable})
        logger.debug(f" Added action '{action.name}' to index under set '{set_name}'")

    def search(self, query: str, limit: int = 10):
        query = query.lower().strip()
        query_words = query.split()

        results = []

        for entry in self._index:
            action = entry["action"]
            text = entry["text"]
            set_name = entry["set"]

            # 1. Scoring
            k_score = keyword_score(query_words, text.split())
            f_score = fuzzy_score(query, text)

            usage_bonus = math.log1p(action.usage_count)
            set_bonus = 1.5 if query in set_name.lower() else 0

            # 2. Weighted Total (Increased weights to ensure matches)
            total = k_score * 5 + f_score * 3 + usage_bonus * 0.4 + set_bonus

            # 3. Debug Output (Optional: Remove if you want cleaner logs)
            if "ls" in action.name or "list" in query:
                print(
                    f"[SCORE] {action.name:12} | f={f_score:.2f} k={k_score:.2f} tot={total:.2f}"
                )

            # 4. Lower Threshold (0.10 is safe; 0.4 was too strict)
            if total > 0.10:
                results.append((total, action, set_name))

        # Sort by score (highest first)
        results.sort(key=lambda x: x[0], reverse=True)
        return [(a, s) for _, a, s in results[:limit]]

    def list_all_tools(self, limit: int = 8) -> List[Dict]:
        """
        Returns a list of all tools currently in the registry.
        """
        tools = []

        # Correct Unpacking:
        # 1. key (name_string)
        # 2. value (tuple containing Action and set_name)
        for name, (action, set_name) in self.all_actions.items():
            tools.append(
                {
                    "name": action.name,
                    "description": action.description,
                    "set_name": set_name,
                    "tags": getattr(action, "tags", []),
                }
            )

        # Sort by name for readability
        tools.sort(key=lambda x: x["name"])

        if limit:
            return tools[:limit]

        return tools

    # Optional: Quick debug print version
    def debug_print_all_tools(self, limit: int = 8):
        tools = self.list_all_tools(limit)
        print("\n" + "=" * 80)
        print("📂 GLOBAL TOOL REGISTRY CONTENTS")
        print("=" * 80)
        for tool in tools:
            print(f"  🛠️  [ {tool['name']:30} ] @ {tool['set_name']}")
            print(f"     📖 {tool['description']}")
            if tool.get("tags"):
                print(f"     🏷️  Tags: {', '.join(tool['tags'])}")
            print("-" * 80)
        print(f"✅ Total Tools: {len(tools)}")
        print("=" * 80 + "\n")

    def execute(self, name: str, **kwargs):
        if name not in self.all_actions:
            raise ValueError(f"Action '{name}' not found")
        action, _ = self.all_actions[name]
        logger.debug(f" Executing action '{name}' with args: {kwargs}")
        result = action.executor(**kwargs)
        action.usage_count = getattr(action, "usage_count", 0) + 1
        return result


# ─────────────────────────────────────────────────────────────
# Create global singleton
# ─────────────────────────────────────────────────────────────
registry = ActionRegistry()
logger.debug(f" Global registry created with sets: {list(registry.sets.keys())}")


def tool(set_name: str, **kwargs):
    """
    Wrap a function as an Action and auto-add it to a set.
    Converts all Python type annotations to JSON Schema types.
    """

    def wrapper(fn):
        name = kwargs.get("name", fn.__name__)
        description = kwargs.get("description", fn.__doc__ or "")
        tags = kwargs.get("tags", [])
        aliases = kwargs.get("aliases", [])

        # ---- Infer parameters from signature if not explicitly provided ----
        sig = signature(fn)
        parameters = kwargs.get("parameters", {})
        required = kwargs.get("required", [])

        for param_name, param in sig.parameters.items():
            if param_name not in parameters:
                param_type = map_type_to_json(param.annotation)
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter '{param_name}'",
                }

            # Determine if required
            if param.default is _empty and param_name not in required:
                required.append(param_name)

        # ---- Create the Action ----
        action = Action(
            name=name,
            description=description,
            tags=tags,
            parameters=parameters,
            required=required,
            executor=fn,
        )
        action.aliases = aliases
        action.usage_count = 0

        # ---- Add to or create ActionSet ----
        if set_name in registry.sets:
            aset = registry.sets[set_name]
        else:
            aset = ActionSet(name=set_name, description=f"{set_name} tools")
            registry.register_set(set_name, aset)
            logger.debug(f" Created new ActionSet '{set_name}' for decorator")

        # Add action to the set
        aset.add(action)

        # ---- Immediately add to search index ----
        registry._add_to_index(action, set_name)
        logger.debug(
            f"Registered action '{name}' under set '{set_name}' with required args: {required}"
        )

        return fn

    return wrapper
