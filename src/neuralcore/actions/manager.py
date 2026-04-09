import re
import inspect

from functools import wraps
from typing import Any, Dict, List, Optional, Set, Union, Callable
from neuralcore.utils.search import fuzzy_score, keyword_score
from neuralcore.actions.actions import Action, ActionSet
from neuralcore.workflows.registry import Workflow

from inspect import signature, _empty, iscoroutinefunction
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


TOKENIZER = re.compile(r"\b\w+(?:[-_]\w+)*\b")


def _tokenize(text: str) -> list[str]:
    return TOKENIZER.findall(text.lower())


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
# Main Action Registry
# ─────────────────────────────────────────────────────────────
class ActionRegistry:
    """Central searchable registry of all tools."""

    def __init__(self):
        self.sets: Dict[str, ActionSet] = {}
        self.all_actions: Dict[str, tuple[Action, str]] = {}
        self._index = []

        logger.debug("ActionRegistry initialized")

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
        """Improved indexing: separate name, description, tags, set_name for better scoring."""
        self.all_actions[action.name] = (action, set_name)

        name_tokens = _tokenize(action.name)
        desc_tokens = _tokenize(action.description)
        tag_tokens = [t.lower() for t in getattr(action, "tags", [])]
        alias_tokens = [a.lower() for a in getattr(action, "aliases", [])]
        set_tokens = _tokenize(set_name)

        searchable_text = " ".join(
            name_tokens + desc_tokens + tag_tokens + alias_tokens + set_tokens
        )

        self._index.append(
            {
                "action": action,
                "set": set_name,
                "name_tokens": name_tokens,  # ← new
                "text": searchable_text,  # full text for fuzzy/keyword
                "set_tokens": set_tokens,  # ← new
            }
        )
        logger.debug(f"Added action '{action.name}' to index under set '{set_name}'")

    def search(
        self,
        query: str,
        limit: int = 5,
        current_agent_id: Optional[str] = None,
        is_subagent: bool = False,
    ):
        """Pure dynamic lexical + semantic search with per-agent and sub-agent hiding."""
        query = query.lower().strip()
        if not query:
            return []

        query_words = _tokenize(query)
        results = []

        for entry in self._index:
            action = entry["action"]

            # ==================== HIDING LOGIC ====================
            if current_agent_id and getattr(action, "hidden_for_agents", None):
                if current_agent_id in action.hidden_for_agents:
                    continue

            if is_subagent and getattr(action, "hidden_for_subagents", False):
                continue
            # =====================================================

            text = entry["text"]
            name_tokens = entry["name_tokens"]

            # Core scoring (unchanged)
            k_score = keyword_score(query_words, text) * 5.0
            f_score = fuzzy_score(query, text) * 3.0

            name_match = len(set(query_words) & set(name_tokens))
            name_boost = name_match * 8.0

            bigram_bonus = 0.0
            if len(query_words) >= 2:
                query_bigrams = {
                    " ".join(query_words[i : i + 2])
                    for i in range(len(query_words) - 1)
                }
                text_bigrams = {
                    " ".join(_tokenize(text)[i : i + 2])
                    for i in range(len(_tokenize(text)) - 1)
                }
                bigram_bonus = len(query_bigrams & text_bigrams) * 4.0

            total_score = k_score + f_score + name_boost + bigram_bonus

            if total_score > 3.0:
                results.append((total_score, action, entry["set"]))

        results.sort(key=lambda x: x[0], reverse=True)
        return [(a, s) for _, a, s in results[:limit]]

    # list_all_tools, debug_print_all_tools, execute unchanged...
    def list_all_tools(self, limit: int = 100) -> List[Dict]:
        tools = []
        for name, (action, set_name) in self.all_actions.items():
            tools.append(
                {
                    "name": action.name,
                    "description": action.description,
                    "set_name": set_name,
                    "tags": getattr(action, "tags", []),
                }
            )
        tools.sort(key=lambda x: x["name"])
        return tools[:limit] if limit else tools

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
        logger.debug(f"Executing action '{name}' with args: {kwargs}")
        result = action.executor(**kwargs)
        action.usage_count = getattr(action, "usage_count", 0) + 1
        return result


# Global singleton
registry = ActionRegistry()
logger.debug(f"Global registry created with sets: {list(registry.sets.keys())}")


class DynamicActionManager:
    def __init__(self, registry: "ActionRegistry", agent: Optional[Any] = None):
        self.current_set = ActionSet("DynamicCore")
        self._agent: Optional[Any] = agent  # ← stored once

        self._loaded_tools: Set[str] = set()
        self._tool_to_set: Dict[str, str] = {}
        self._set_to_tools: Dict[str, Set[str]] = {}

        self._persistent_tools: Set[str] = {
            "FindTool",
        }
        self._discovered_tools: Set[str] = set()

        self.protect_persistent_tools = True

        logger.debug(
            f"DynamicActionManager initialized with agent={getattr(agent, 'agent_id', 'None')}"
        )

    def add(self, action: Action, origin_set: Optional[str] = None):
        if action.name in self._loaded_tools:
            return

        self.current_set.add(action)
        self._loaded_tools.add(action.name)

        if origin_set:
            self._tool_to_set[action.name] = origin_set
            self._set_to_tools.setdefault(origin_set, set()).add(action.name)

        if origin_set is None or "FindTool" in action.name:
            self._discovered_tools.add(action.name)

        # === KEY FIX: Bind agent immediately when the action is added to DynamicCore ===
        if self._agent is not None:
            try:
                action.bind_agent(self._agent)
                logger.debug(f"[BIND ON ADD] {action.name} → agent bound at load time")
            except Exception as e:
                logger.error(f"Failed to bind agent to '{action.name}' on add: {e}")

        logger.debug(
            f"Added tool '{action.name}' to DynamicCore (origin: {origin_set or 'manual'})"
        )

    def select_toolset(self, toolset_names: Union[str, List[str]]) -> int:
        """
        Select one or more toolsets as active.
        Unloads all other dynamic tools and loads only the specified toolset(s).

        Returns the number of tools newly loaded.
        """
        # Step 1: unload all non-persistent tools
        self.unload_all()

        # Step 2: load the desired toolset(s)
        return self.load_toolsets(toolset_names)

    def get_action_set(self, set_name: str) -> Optional[ActionSet]:
        if set_name == "DynamicCore":
            return self.current_set

        action_set = registry.sets.get(set_name)
        if not action_set:
            logger.warning(f"ActionSet '{set_name}' not found in registry")
        return action_set

    def load_tools(self, tool_names: Union[str, List[str], None] = None):
        """Load specific tools by name. Safely handles None or empty input."""
        if not tool_names:
            return

        if isinstance(tool_names, str):
            tool_names = [tool_names]

        if not isinstance(tool_names, list):
            logger.warning(f"Invalid tool_names type: {type(tool_names)}. Skipping.")
            return

        loaded = []
        for name in tool_names:
            if not name or not isinstance(name, str):
                continue
            if name in self._loaded_tools:
                continue
            if name not in registry.all_actions:
                logger.warning(f"Tool '{name}' not found in registry")
                continue

            action, origin_set = registry.all_actions[name]
            self.add(action, origin_set=origin_set)
            loaded.append(name)

        if loaded:
            logger.info(f"Loaded {len(loaded)} individual tools: {', '.join(loaded)}")

    def load_toolsets(self, toolset_names: Union[str, List[str], None] = None) -> int:
        """Load all actions from one or more named ActionSets. Safely handles None."""
        if not toolset_names:
            return 0

        if isinstance(toolset_names, str):
            toolset_names = [toolset_names]

        if not isinstance(toolset_names, list):
            logger.warning(
                f"Invalid toolset_names type: {type(toolset_names)}. Skipping."
            )
            return 0

        loaded_count = 0
        loaded_tools_list: List[str] = []

        for set_name in toolset_names:
            if not isinstance(set_name, str) or not set_name.strip():
                continue

            set_name = set_name.strip()
            if set_name not in registry.sets:
                logger.warning(f"Toolset '{set_name}' not found in registry")
                continue

            action_set = registry.sets[set_name]
            newly_loaded = []

            for action in action_set.actions:
                if action.name in self._loaded_tools:
                    continue
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

    def configure_for_step(
        self, step_name: str, engine, workflow: Optional["Workflow"] = None
    ) -> int:
        """
        Configure tools for a workflow step based on @workflow.set decorator.
        - By default protects persistent tools.
        - Respects hidden_toolsets and dynamic_allowed.
        - Always restores persistent tools at the end unless protection is disabled.
        """
        if workflow is None:
            from neuralcore.workflows.registry import workflow as global_workflow

            workflow = global_workflow

        if workflow is None:
            logger.warning(
                f"No Workflow instance for step '{step_name}'. Unloading all."
            )
            self.unload_all()
            return 0

        meta = workflow.get_step_metadata(step_name)
        if not meta:
            logger.warning(f"No metadata for step '{step_name}'. Unloading all.")
            self.unload_all()
            return 0

        logger.info(f"🔧 Configuring tools for workflow step: '{step_name}'")

        workflow_name = getattr(workflow, "current_workflow_name", "") or getattr(
            workflow, "name", ""
        )
        is_sub_workflow = workflow_name == "sub_agent_execute"

        # Decide unload strategy
        if is_sub_workflow and step_name == "llm_stream":
            logger.debug(
                "Sub-agent llm_stream: preserving assigned tools (flexible mode)"
            )
            self.unload_tools(["FindTool"])
        else:
            self.unload_all()  # respects protect_persistent_tools

        loaded_count = 0

        # 1. Load explicitly allowed toolsets
        toolsets = meta.get("toolsets")
        if toolsets:
            loaded_count += self.load_toolsets(toolsets)

        # 2. Load specific individual tools
        tools = meta.get("tools")
        if tools:
            self.load_tools(tools)
            loaded_count += len(tools) if isinstance(tools, (list, tuple)) else 1

        # 3. Handle hidden_toolsets
        hidden_toolsets: List[str] = []
        if engine:
            wf_meta = engine.registered_workflows.get(workflow_name)
            if wf_meta and wf_meta.get("hidden_toolsets"):
                hidden = wf_meta["hidden_toolsets"]
                if isinstance(hidden, str):
                    hidden_toolsets.append(hidden)
                elif isinstance(hidden, (list, tuple)):
                    hidden_toolsets.extend(hidden)

        step_hidden = meta.get("hidden_toolsets")
        if step_hidden:
            if isinstance(step_hidden, str):
                hidden_toolsets.append(step_hidden)
            elif isinstance(step_hidden, (list, tuple)):
                hidden_toolsets.extend(step_hidden)

        if hidden_toolsets:
            self.unload_toolsets(hidden_toolsets)
            logger.info(f"Hidden toolsets for step '{step_name}': {hidden_toolsets}")

        # 4. Dynamic browsing control
        if not meta.get("dynamic_allowed", True):
            self.unload_tools(["FindTool"])
            logger.debug(f"FindTool disabled for step '{step_name}'")

        # FINAL SAFETY: Restore persistent tools unless protection is off
        if self.protect_persistent_tools:
            for p_name in self._persistent_tools:
                if p_name in registry.all_actions and not self.is_loaded(p_name):
                    action, origin = registry.all_actions[p_name]
                    self.add(action, origin_set=origin)
                    loaded_count += 1

        current_tools = len(self.current_set.actions)
        logger.info(
            f"✅ Step '{step_name}' configured with {current_tools} tools "
            f"(toolsets={toolsets or '-'}, hidden={hidden_toolsets or '-'}, "
            f"tools={tools or '-'}, dynamic_allowed={meta.get('dynamic_allowed', True)})"
        )

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

    def unload_all(self, keep_discovered: bool = False):
        """Unload non-persistent tools.

        Args:
            keep_discovered: If True, protect tools that were dynamically loaded via FindTool.
                            This is a generic flag — the caller (NeuralVoid) decides when to use it.
        """
        to_remove = []
        for name in list(self._loaded_tools):
            if name in self._persistent_tools:
                continue
            if keep_discovered and name in self._discovered_tools:
                continue
            to_remove.append(name)

        for name in to_remove:
            self.current_set.remove_by_name(name)
            self._loaded_tools.remove(name)
            self._tool_to_set.pop(name, None)

        self._set_to_tools.clear()

        # Re-add persistent tools
        if self.protect_persistent_tools:
            for p_name in self._persistent_tools:
                if p_name in registry.all_actions and not self.is_loaded(p_name):
                    action, origin = registry.all_actions[p_name]
                    self.add(action, origin_set=origin)

        # Re-add discovered tools when requested
        if keep_discovered:
            for d_name in list(self._discovered_tools):
                if not self.is_loaded(d_name) and d_name in registry.all_actions:
                    action, origin = registry.all_actions[d_name]
                    self.add(action, origin_set=origin)

        logger.info(
            f"Unloaded dynamic tools. Persistent protected. "
            f"keep_discovered={keep_discovered} → {len(self._discovered_tools)} kept"
        )

    def reset_to_default_package(self, step_name: str, engine) -> int:
        logger.info(f"🔄 Resetting to default tool package for step '{step_name}'")

        # Generic: always unload non-persistent first (caller can control via keep_discovered later)
        self.unload_all(keep_discovered=False)

        # Load defaults from metadata (this part can stay)
        meta = engine.get_step_metadata(step_name) if engine else None
        loaded_count = 0
        if meta:
            if toolsets := meta.get("toolsets"):
                loaded_count += self.load_toolsets(toolsets)
            if tools := meta.get("tools"):
                self.load_tools(tools)
                loaded_count += len(tools) if isinstance(tools, (list, tuple)) else 1

        # Always ensure persistent tools
        for p_name in self._persistent_tools:
            if p_name in registry.all_actions and not self.is_loaded(p_name):
                action, origin = registry.all_actions[p_name]
                self.add(action, origin_set=origin)
                loaded_count += 1

        logger.info(f"✅ Default package restored for step '{step_name}'")
        return loaded_count

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

    def get_executor(self, name: str, agent: Optional[Any] = None) -> Optional[Action]:
        """Simple delegation. Binding already happened in .add()"""
        action = self.current_set.get_executor(name)
        if action is None:
            logger.warning(f"[DYNAMIC MANAGER] No executor for '{name}'")
            return None

        # Optional re-bind if a different agent is explicitly passed (rare)
        binding_agent = agent or self._agent
        if (
            binding_agent is not None
            and action._needs_agent
            and action._bound_agent is None
        ):
            try:
                action.bind_agent(binding_agent)
            except Exception as e:
                logger.error(f"Failed to bind agent to '{name}': {e}")
                raise

        return action

    @property
    def actions(self):
        return self.current_set.actions

    @property
    def by_name(self):
        return self.current_set.by_name


# ─────────────────────────────────────────────────────────────
# Tool Browser — PURE DYNAMIC SEARCH
# ─────────────────────────────────────────────────────────────
class ToolBrowser(Action):
    def __init__(self, registry: "ActionRegistry", manager: DynamicActionManager):
        super().__init__(
            name="FindTool",
            description=(
                "Call this when you need a specific tool or capability that is not currently available. "
                "Provide a short, clear description of the required action (3-10 words). "
                "Examples: 'web search', 'read pdf file', 'search code', 'list directory', 'fetch webpage'"
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "Short keyword-rich description of the needed capability",
                },
            },
            executor=self._search,
            required=["query"],
        )
        self.manager = manager
        manager.current_set.add(self)
        logger.info("ToolBrowser (FindTool) added to DynamicCore as persistent tool")

    async def _search(self, query: str):
        logger.info(f"ToolBrowser search executed: query='{query}'")

        # === Detect current agent context ===
        current_agent_id = None
        is_subagent = False
        if self.manager and getattr(self.manager, "_agent", None):
            agent = self.manager._agent
            current_agent_id = getattr(agent, "agent_id", None)
            is_subagent = getattr(
                agent, "sub_agent", False
            )  # ← uses the flag from your Agent class

        results = registry.search(
            query, limit=3, current_agent_id=current_agent_id, is_subagent=is_subagent
        )

        if not results:
            logger.info("ToolBrowser: no matches found")
            return {
                "status": "no_matches",
                "message": "No tools found matching your request.",
            }

        self.manager.load_tools([a.name for a, _ in results])
        logger.info(f"ToolBrowser: loaded {len(results)} tools")
        return {
            "status": "tools_loaded",
            "loaded_tools": [a.name for a, _ in results],
            "message": f"Found and loaded {len(results)} relevant tools.",
        }


class AgentActionHelper:
    """
    Helper to register actions bound to a specific Agent instance.

    - Automatically injects `agent` into executors when required
    - Builds clean JSON schema (hides internal args like `agent`)
    - Supports both sync and async functions
    """

    def __init__(self, agent):
        self.agent = agent
        self.manager = self.agent.manager

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def register_action(self, action: Action, set_name: Optional[str] = None):
        """Register an already created Action."""
        self._ensure_set(set_name)

        self.manager.add(action, origin_set=set_name)
        registry._add_to_index(action, set_name or "manual")

        logger.debug(f"Agent '{self.agent.agent_id}' registered action: {action.name}")

    def register_action_from_function(
        self,
        func: Callable,
        set_name: str,
        **kwargs,
    ) -> Action:
        """
        Wrap a function into an Action and register it.

        Supports:
        - def tool(agent, ...)
        - def tool(...)
        - async def tool(...)
        """

        name = kwargs.get("name", func.__name__)
        description = kwargs.get("description", func.__doc__ or "")
        tags = kwargs.get("tags", [])
        aliases = kwargs.get("aliases", [])

        parameters = dict(kwargs.get("parameters", {}))  # copy!
        required = list(kwargs.get("required", []))  # copy!

        sig = signature(func)

        # Build schema (skip internal params like 'agent')
        for param_name, param in sig.parameters.items():
            if param_name == "agent":
                continue

            if param_name not in parameters:
                param_type = map_type_to_json(param.annotation)
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter '{param_name}'",
                }

            if param.default is _empty and param_name not in required:
                required.append(param_name)

        # Bind agent if needed
        executor = self._bind_executor(func)

        action = Action(
            name=name,
            description=description,
            tags=tags,
            parameters=parameters,
            required=required,
            executor=executor,
            hidden_for_agents=kwargs.get("hidden_for_agents"),
            hidden_for_subagents=kwargs.get("hidden_for_subagents", False),
        )

        action.aliases = aliases
        action.usage_count = 0

        self.register_action(action, set_name=set_name)
        return action

    def register_agent_method(
        self,
        method_name: str,
        set_name: str,
        **kwargs,
    ) -> Action:
        """
        Register a method directly from the agent.
        """
        method = getattr(self.agent, method_name)

        return self.register_action_from_function(
            method,
            set_name=set_name,
            name=kwargs.get("name", method_name),
            description=kwargs.get("description", method.__doc__ or ""),
            **kwargs,
        )

    def get_agent_tools(self) -> List[Dict[str, Any]]:
        """Return currently available LLM tools."""
        return self.manager.get_llm_tools()

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _ensure_set(self, set_name: Optional[str]):
        if not set_name:
            return

        if set_name not in registry.sets:
            aset = ActionSet(name=set_name, description=f"{set_name} tools")
            registry.register_set(set_name, aset)

    def _needs_agent_arg(self, func: Callable) -> bool:
        sig = signature(func)
        params = list(sig.parameters.values())

        return len(params) > 0 and params[0].name == "agent"

    def _bind_executor(self, func: Callable) -> Callable:
        if not self._needs_agent_arg(func):
            return func

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(self.agent, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(self.agent, *args, **kwargs)

            return sync_wrapper


def tool(set_name: str, **kwargs):
    """
    Wrap a function as an Action and auto-add it to a set.
    - First argument can be 'agent' (or 'self') and will be injected automatically
    - Sync or async functions supported
    - 'agent' is hidden from the final parameters schema
    """

    def wrapper(fn: Callable):
        name = kwargs.get("name", fn.__name__)
        description = kwargs.get("description", fn.__doc__ or "")
        tags = kwargs.get("tags", [])
        aliases = kwargs.get("aliases", [])

        sig = signature(fn)
        parameters = dict(kwargs.get("parameters", {}))
        required = list(kwargs.get("required", []))

        # Skip first argument if it's 'agent' or 'self' (hide from schema)
        param_list = list(sig.parameters.items())
        skip_first = bool(param_list and param_list[0][0] in ("agent", "self"))

        for i, (param_name, param) in enumerate(param_list):
            if skip_first and i == 0:
                continue
            if param_name not in parameters:
                param_type = map_type_to_json(param.annotation)
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter '{param_name}'",
                }
            if param.default is _empty and param_name not in required:
                required.append(param_name)

        # Wrap executor to inject agent automatically if first param is agent/self
        if iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if skip_first and args and hasattr(args[0], "context_manager"):
                    return await fn(args[0], *args[1:], **kwargs)
                else:
                    return await fn(*args, **kwargs)

            executor = async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                if skip_first and args and hasattr(args[0], "context_manager"):
                    return fn(args[0], *args[1:], **kwargs)
                else:
                    return fn(*args, **kwargs)

            executor = sync_wrapper

        # Create Action (agent is hidden from schema)
        action = Action(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            executor=executor,
            tags=tags,
            aliases=aliases,
            hidden_for_agents=kwargs.get("hidden_for_agents"),
            hidden_for_subagents=kwargs.get("hidden_for_subagents", False),
        )

        # Add to registry
        if set_name in registry.sets:
            aset = registry.sets[set_name]
        else:
            aset = ActionSet(name=set_name, description=f"{set_name} tools")
            registry.register_set(set_name, aset)
            logger.debug(f"Created new ActionSet '{set_name}' for decorator")

        aset.add(action)
        registry._add_to_index(action, set_name)
        logger.debug(f"Registered action '{name}' under set '{set_name}'")

        return fn

    return wrapper
