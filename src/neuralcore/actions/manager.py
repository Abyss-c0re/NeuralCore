import time
from typing import Any, Dict, List, Optional, Set, Union

from neuralcore.actions.actions import Action, ActionSet
from neuralcore.actions.registry import ActionRegistry
from neuralcore.workflows.registry import Workflow

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


class DynamicActionManager:
    def __init__(self, registry: "ActionRegistry", agent):
        self.current_set = ActionSet("DynamicCore")
        self._agent = agent
        self._consolidator = agent.context_manager.consolidator

        self._loaded_tools: Set[str] = set()
        self._tool_to_set: Dict[str, str] = {}
        self._set_to_tools: Dict[str, Set[str]] = {}

        self._persistent_tools: Set[str] = {
            "FindTool",
        }
        self._discovered_tools: Set[str] = set()

        self.protect_persistent_tools = True
        self.registry = registry

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

        action_set = self.registry.sets.get(set_name)
        if not action_set:
            logger.warning(f"ActionSet '{set_name}' not found in registry")
        return action_set

    def load_tools(self, tool_names: Union[str, List[str], None] = None):
        """Load specific tools by name. Safely handles None or empty input."""
        if not tool_names:
            return

        if isinstance(tool_names, str):
            tool_names = [tool_names]

        loaded = []
        for name in tool_names:
            if not name or not isinstance(name, str):
                continue
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

    def load_toolsets(self, toolset_names: Union[str, List[str], None] = None) -> int:
        """Load all actions from one or more named ActionSets. Safely handles None."""
        if not toolset_names:
            return 0

        if isinstance(toolset_names, str):
            toolset_names = [toolset_names]

        loaded_count = 0
        loaded_tools_list: List[str] = []

        for set_name in toolset_names:
            if not isinstance(set_name, str) or not set_name.strip():
                continue

            set_name = set_name.strip()
            if set_name not in self.registry.sets:
                logger.warning(f"Toolset '{set_name}' not found in registry")
                continue

            action_set = self.registry.sets[set_name]
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
                if p_name in self.registry.all_actions and not self.is_loaded(p_name):
                    action, origin = self.registry.all_actions[p_name]
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
                if p_name in self.registry.all_actions and not self.is_loaded(p_name):
                    action, origin = self.registry.all_actions[p_name]
                    self.add(action, origin_set=origin)

        # Re-add discovered tools when requested
        if keep_discovered:
            for d_name in list(self._discovered_tools):
                if not self.is_loaded(d_name) and d_name in self.registry.all_actions:
                    action, origin = self.registry.all_actions[d_name]
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
            if p_name in self.registry.all_actions and not self.is_loaded(p_name):
                action, origin = self.registry.all_actions[p_name]
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

    async def execute_direct(
        self,
        action_name: str,
        **kwargs: Any,
    ) -> Any:
        """
        Direct execution using currently loaded tools in DynamicCore.
        - Auto-loads the action if missing (respects hidden_for_agents / subagents)
        - Falls back safely to registry.aexecute
        - Never returns None — raises clear error instead
        """
        if action_name not in self.current_set.by_name:
            logger.info(
                f"[DIRECT EXEC] Action '{action_name}' not in DynamicCore → attempting auto-load"
            )
            # Reuse existing search (respects all hiding rules)
            results = self.registry.search(action_name, limit=1)
            if results:
                self.load_tools([results[0][0].name])
                logger.debug(f"[DIRECT EXEC] Auto-loaded '{action_name}'")
            else:
                logger.warning(
                    f"[DIRECT EXEC] No match found for '{action_name}' → falling back to registry"
                )
                return await self.registry.aexecute(action_name, **kwargs)

        # Safe get
        action = self.current_set.get_executor(action_name)
        if action is None:
            raise RuntimeError(
                f"[DIRECT EXEC FAILED] get_executor returned None for '{action_name}' "
                f"(even after auto-load). Check registry or hidden_for_agents rules."
            )

        logger.debug(
            f"[DIRECT EXEC via DynamicManager] {action_name} | kwargs={kwargs}"
        )
        return await action(**kwargs)  # ← full Action.__call__

    async def search_and_load(
        self,
        query: str,
        limit: int = 5,
        use_reranker: bool = True,
    ) -> List[str]:
        """
        Enhanced tool search + load with full description + tags reranking.
        Completely safe against None, empty tuples, and mixed return types from registry.search().
        """
        logger.debug(
            f"[TOOL SEARCH] query='{query[:80]}...' | use_reranker={use_reranker} | limit={limit}"
        )

        # Step 1: Broad lexical search
        results = self.registry.search(
            query,
            limit=max(limit * 4, 20),
            current_agent_id=getattr(self._agent, "agent_id", None),
            is_subagent=getattr(self._agent, "sub_agent", False),
        )

        if not results:
            logger.info("ToolBrowser: no lexical matches found")
            return []

        if not use_reranker or self._consolidator is None or len(results) <= 1:
            # SAFE FALLBACK — handle empty tuples and malformed items
            tool_names: List[str] = []
            for item in results[:limit]:
                if item is None:
                    continue
                if isinstance(item, tuple):
                    if (
                        len(item) >= 1
                        and item[0] is not None
                        and hasattr(item[0], "name")
                    ):
                        tool_names.append(item[0].name)
                elif hasattr(item, "name"):
                    tool_names.append(item.name)
            self.load_tools(tool_names)
            return tool_names

        # Step 2: Rich adaptation for reranker (full description + tags)
        adapted_candidates = []
        for action, set_name in results:
            rich_text = f"{action.name}\n{action.description}\nTags: {', '.join(getattr(action, 'tags', []))}\nSet: {set_name}"

            adapted = type(
                "ToolCandidate",
                (object,),
                {
                    "key": action.name,
                    "content": rich_text,
                    "source_type": "tool",
                    "embedding": None,
                    "timestamp": getattr(action, "last_used", time.time()),
                    "usage_count": getattr(action, "usage_count", 0),
                    "category_path": set_name,
                },
            )()
            adapted_candidates.append(adapted)

        # Step 3: Rerank with LambdaMART
        try:
            reranked_items = await self._consolidator.rerank(
                query=query,
                candidates=adapted_candidates,
                k=limit,
            )

            name_to_action = {
                action.name: (action, set_name) for action, set_name in results
            }
            results = [
                name_to_action.get(item.key)
                for item in reranked_items
                if item.key in name_to_action
            ]
            results = [r for r in results if r is not None]

            logger.debug(f"[TOOL RERANK] LambdaMART reranked → {len(results)} tools")
        except Exception as e:
            logger.warning(
                f"[TOOL RERANK] Consolidator failed, falling back to lexical: {e}"
            )
            results = results[:limit]

        # Step 4: Safe tool name extraction (final guard)
        tool_names: List[str] = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, tuple):
                if len(item) >= 1 and item[0] is not None and hasattr(item[0], "name"):
                    tool_names.append(item[0].name)
            elif hasattr(item, "name"):
                tool_names.append(item.name)

        self.load_tools(tool_names)

        logger.info(
            f"ToolBrowser: loaded {len(tool_names)} tools (reranked) → {tool_names}"
        )
        return tool_names

    @property
    def actions(self):
        return self.current_set.actions

    @property
    def by_name(self):
        return self.current_set.by_name

    @property
    def loaded_tools(self) -> List[str]:
        """Public, read-only list of currently loaded tool names.
        Generic observability API — safe for any client bridge / dashboard.
        """
        return list(self._loaded_tools)

    @property
    def loaded_toolsets(self) -> List[str]:
        """Public list of toolsets that have active tools loaded."""
        return self.get_loaded_toolsets()


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
        self.registry = registry
        manager.current_set.add(self)
        logger.info("ToolBrowser (FindTool) added to DynamicCore as persistent tool")

    async def _search(self, query: str):
        """Simple thin wrapper around DynamicActionManager.search_and_load"""
        logger.info(f"ToolBrowser search executed: query='{query}'")

        loaded_names = await self.manager.search_and_load(
            query=query,
            limit=3,
            use_reranker=True,
        )

        if not loaded_names:
            logger.info("ToolBrowser: no matches found")
            return {
                "status": "no_matches",
                "message": "No tools found matching your request.",
            }

        logger.info(f"ToolBrowser: loaded {len(loaded_names)} tools → {loaded_names}")
        return {
            "status": "tools_loaded",
            "loaded_tools": loaded_names,
            "message": f"Found and loaded {len(loaded_names)} relevant tools (reranked with LambdaMART).",
        }
