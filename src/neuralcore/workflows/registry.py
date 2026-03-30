from functools import wraps
from typing import Any, Dict, Callable, List, Union, Optional
from neuralcore.workflows.engine import WorkflowEngine
from inspect import signature, iscoroutinefunction


from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


# ===================================================================
# WORKFLOW — Decorator-based registry (exactly like @tool)
# ===================================================================


logger = Logger.get_logger()


class Workflow:
    """
    Enhanced Workflow Registry with per-step tool control.

    Supports:
      - toolsets: which toolsets to load
      - tools: specific individual tools
      - hidden_toolsets: toolsets to explicitly hide/unload for this step
      - dynamic_allowed: whether BrowseTools is permitted
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}  # workflow_name → metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}  # step_name → step metadata

    def bind_to_engine(self, engine: "WorkflowEngine", instance=None):
        """
        Register all decorated steps to the engine's step handler map.
        If 'instance' is provided, bind all step handlers to it.
        """
        for step_name, handler in self.handlers.items():
            bound_handler = handler
            if instance is not None:
                bound_handler = handler.__get__(instance, instance.__class__)
            engine.register_step(step_name, bound_handler)
            logger.info(
                f"🔗 Bound step '{step_name}' to engine of agent '{engine.agent.name}'"
            )

    def set(
        self,
        workflow_name: str = "default",
        *,
        toolsets: Union[str, List[str], None] = None,
        tools: Union[str, List[str], None] = None,
        hidden_toolsets: Union[str, List[str], None] = None,  # ← NEW
        dynamic_allowed: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Decorator to register a workflow step with explicit tool configuration.
        """

        def decorator(fn: Callable):
            step_name = name or kwargs.get("name") or fn.__name__
            if step_name.startswith("_wf_"):
                step_name = step_name[4:]

            desc = (
                description
                or kwargs.get("description")
                or fn.__doc__
                or f"Step: {step_name}"
            )

            # Normalize all lists
            norm_toolsets = self._normalize_list(toolsets)
            norm_tools = self._normalize_list(tools)
            norm_hidden = self._normalize_list(hidden_toolsets)

            # ====================== AGENT BINDING LOGIC ======================
            sig = signature(fn)
            param_list = list(sig.parameters.items())

            # Check if first parameter is 'agent' or 'self'
            skip_first = bool(param_list and param_list[0][0] in ("agent", "self"))

            if skip_first:
                if iscoroutinefunction(fn):

                    @wraps(fn)
                    async def async_step_wrapper(*args, **kwargs):
                        # If agent is already passed as first positional arg, use it
                        if args and (
                            hasattr(args[0], "agent_id") or args[0] is not None
                        ):
                            return await fn(*args, **kwargs)

                        # Otherwise, try to get agent from workflow/engine context
                        agent = getattr(self, "_current_agent", None)
                        if agent is None:
                            # Fallback: look for agent in kwargs (for manual calls)
                            agent = kwargs.pop("agent", None)

                        if agent is None:
                            raise RuntimeError(
                                f"Step '{step_name}' expects 'agent' as first parameter "
                                f"but no agent instance was provided or bound."
                            )

                        return await fn(agent, *args, **kwargs)

                    executor = async_step_wrapper
                else:

                    @wraps(fn)
                    def sync_step_wrapper(*args, **kwargs):
                        if args and (
                            hasattr(args[0], "agent_id") or args[0] is not None
                        ):
                            return fn(*args, **kwargs)

                        agent = getattr(self, "_current_agent", None)
                        if agent is None:
                            agent = kwargs.pop("agent", None)

                        if agent is None:
                            raise RuntimeError(
                                f"Step '{step_name}' expects 'agent' as first parameter "
                                f"but no agent instance was provided or bound."
                            )

                        return fn(agent, *args, **kwargs)

                    executor = sync_step_wrapper
            else:
                executor = fn

            # Register the (possibly wrapped) handler
            self.handlers[step_name] = executor

            # Auto-create workflow entry
            if workflow_name not in self.workflows:
                self.workflows[workflow_name] = {
                    "description": kwargs.get(
                        "workflow_description", f"Workflow: {workflow_name}"
                    ),
                    "steps": [],
                }

            if step_name not in self.workflows[workflow_name]["steps"]:
                self.workflows[workflow_name]["steps"].append(step_name)

            # Store rich metadata
            self.metadata[step_name] = {
                "description": desc,
                "workflow": workflow_name,
                "toolsets": norm_toolsets,
                "tools": norm_tools,
                "hidden_toolsets": norm_hidden,
                "dynamic_allowed": dynamic_allowed,
                "full_name": fn.__name__,
                "requires_agent": skip_first,  # ← Useful for debugging / introspection
            }

            logger.info(
                f"✅ Step '{step_name}' registered to workflow '{workflow_name}' "
                f"(toolsets={norm_toolsets or '-'}, hidden={norm_hidden or '-'}, "
                f"tools={norm_tools or '-'}, dynamic={dynamic_allowed}, "
                f"agent_bound={skip_first})"
            )

            return fn  # Return original function (decorator transparency)

        return decorator

    def _normalize_list(self, value: Union[str, List[str], None]) -> List[str]:
        """Convert string, list, or None into a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if item and str(item).strip()]
        return []

    # ====================== Public Query Methods ======================

    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Return full metadata for a step (including hidden_toolsets)."""
        return self.metadata.get(step_name)

    def get_workflow_steps(self, workflow_name: str) -> List[str]:
        return self.workflows.get(workflow_name, {}).get("steps", [])

    def get_workflow_metadata(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        return self.workflows.get(workflow_name)

    def list_all_steps(self) -> List[Dict[str, Any]]:
        """Debug helper."""
        steps = []
        for step_name, meta in self.metadata.items():
            steps.append(
                {
                    "step": step_name,
                    "workflow": meta.get("workflow"),
                    "toolsets": meta.get("toolsets"),
                    "hidden_toolsets": meta.get("hidden_toolsets"),
                    "tools": meta.get("tools"),
                    "dynamic_allowed": meta.get("dynamic_allowed", True),
                }
            )
        return sorted(steps, key=lambda x: (x["workflow"], x["step"]))

    def debug_print(self):
        """Nice debug output."""
        print("\n" + "=" * 100)
        print("📋 WORKFLOW REGISTRY (with hidden_toolsets support)")
        print("=" * 100)

        for wf_name, wf_meta in self.workflows.items():
            print(f"\n🔹 Workflow: {wf_name}")
            print(f"   Description: {wf_meta.get('description', '')}")
            print(f"   Steps: {len(wf_meta['steps'])}")

            for step_name in wf_meta["steps"]:
                meta = self.metadata.get(step_name, {})
                ts = meta.get("toolsets", [])
                hidden = meta.get("hidden_toolsets", [])
                tl = meta.get("tools", [])
                dyn = "✓" if meta.get("dynamic_allowed", True) else "✗"

                print(
                    f"     • {step_name:30} | toolsets={ts or '-':25} | "
                    f"hidden={hidden or '-':25} | tools={tl or '-':20} | dynamic={dyn}"
                )

        print(
            f"\n✅ Total Workflows: {len(self.workflows)} | Total Steps: {len(self.metadata)}"
        )
        print("=" * 100 + "\n")


class Condition:
    """
    Global condition registry.
    Decorator: @condition.add("condition_name", description="...")
    """

    def __init__(self):
        # name -> handler(state, args_dict) -> bool
        self.handlers: Dict[str, Callable[[Any, Optional[dict]], bool]] = {}

    def add(self, name: str, description: str = ""):
        """
        Decorator to register a condition.
        Usage:
            @condition.add("has_tools_recently", description="True if agent used tools recently")
            def has_tools(state, args=None):
                return bool(state.tool_calls)
        """

        def decorator(fn: Callable[[Any, Optional[dict]], bool]):
            self.handlers[name] = fn
            logger.info(
                f"✅ Registered condition '{name}' → {description or fn.__doc__}"
            )
            return fn

        return decorator

    def register_to_engine(self, engine):
        """
        Push all registered conditions to a WorkflowEngine instance.
        """
        for name, handler in self.handlers.items():
            engine.register_custom_condition(name, handler)
            logger.info(f"✅ Condition '{name}' registered in engine")


# Singleton instance
condition = Condition()

workflow = Workflow()
