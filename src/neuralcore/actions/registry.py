from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable
from neuralcore.utils.search import fuzzy_score, keyword_score
from neuralcore.utils.formatting import _tokenize, map_type_to_json
from neuralcore.actions.actions import Action, ActionSet
from neuralcore.actions.sequence import sequence


from inspect import signature, _empty, iscoroutinefunction


from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


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
                "name_tokens": name_tokens,
                "text": searchable_text,
                "set_tokens": set_tokens,
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


def sequenced(
    name: str,
    description: str,
    *,
    steps: List[str],  # tool names as strings (lazy resolved)
    propagate: bool = True,
    output_from: Union[int, str, None] = -1,
    confirm_predicate: Optional[Callable[[Any], bool]] = None,
    tags: Optional[List[str]] = None,
    set_name: str,  # required, same as @tool
    dependencies: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
) -> Callable[[Callable], Action]:
    """
    Decorator to register a multi-step sequence as a first-class Action.

    Example:
    @sequenced(
        name="find_and_read_file",
        description="Search and read first matching file",
        set_name="FileEditingTools",
        steps=["search_files", "read_file"],
        dependencies={
            "search_files": {"name": "input"},
            "read_file": {"file_path": "search_files"}
        }
    )
    def find_and_read_file():
        pass
    """

    def decorator(original_func: Callable) -> Action:
        if not steps:
            logger.warning(f"[SEQUENCED] Sequence '{name}' has no steps defined.")

        # Create the SequenceAction with lazy step resolution
        seq = sequence(
            name=f"{name}_internal",
            description=description,
            steps=[],  # temporary
            propagate=propagate,  # ← make sure it uses the public name
            output_from=output_from,
            confirm_predicate=confirm_predicate,
            dependencies=dependencies,
            tags=tags,
        )

        # Mark for lazy resolution at execution time (this is the cleanest way)
        seq._step_names_to_resolve = steps  # type: ignore[attr-defined]

        # Optional: store original steps for debugging
        seq._original_step_names = steps  # type: ignore[attr-defined]

        # Register into ActionSet (exactly like @tool decorator)
        if set_name not in registry.sets:
            aset = ActionSet(name=set_name, description=f"{set_name} tools")
            registry.register_set(set_name, aset)
        else:
            aset = registry.sets[set_name]

        aset.add(seq)
        registry._add_to_index(seq, set_name)

        logger.info(
            f"[SEQUENCED] Registered sequence '{name}' into set '{set_name}' "
            f"(steps: {steps})"
        )

        # Optional: attach back to original function for introspection
        try:
            original_func._registered_sequence_action = seq  # type: ignore[attr-defined]
        except Exception:
            pass  # not critical

        return seq

    return decorator
