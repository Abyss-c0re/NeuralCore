from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from inspect import signature, iscoroutinefunction  # ← restored exact original imports

from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class Workflow:
    """
    Unified Workflow Registry with Steps, Loops, and Conditions.
    100% backward compatible with original @workflow.set agent binding.
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Merged Loops + Conditions
        self.loops: Dict[str, Dict[str, Any]] = {}
        self.conditions: Dict[str, Callable] = {}

        logger.debug("Workflow registry initialized (steps + loops + conditions)")

    # ===================================================================
    # CONDITIONS
    # ===================================================================
    def condition(self, name: str, description: str = ""):
        """Decorator: @workflow.condition('name')"""
        def decorator(fn: Callable):
            self.conditions[name] = fn
            logger.info(
                f"✅ Condition registered: '{name}' → {description or fn.__doc__ or 'No description'}"
            )
            return fn
        return decorator

    def evaluate_condition(self, condition_name: str, state: Any, args: Optional[dict] = None) -> bool:
        """Evaluate condition registered with @workflow.condition"""
        if condition_name not in self.conditions:
            logger.warning(f"Condition '{condition_name}' not found.")
            return False

        try:
            handler = self.conditions[condition_name]
            sig = signature(handler)
            param_count = len(sig.parameters)

            if param_count == 0:
                return bool(handler())
            elif param_count == 1:
                return bool(handler(state))
            else:
                return bool(handler(state, args))

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition_name}': {e}", exc_info=True)
            return False
    # ===================================================================
    # LOOPS
    # ===================================================================
    def loop(
        self,
        loop_name: str,
        *,
        max_iterations: int = 10,
        break_condition: Optional[str] = None,
        continue_condition: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """@workflow.loop('name', max_iterations=..., break_condition=...)"""

        def decorator(fn: Callable):
            self.loops[loop_name] = {
                "name": loop_name,
                "handler": fn,
                "max_iterations": max_iterations,
                "break_condition": break_condition,
                "continue_condition": continue_condition,
                "description": description or fn.__doc__ or f"Loop: {loop_name}",
            }

            # Register as hidden step for compatibility
            step_name = f"_loop_{loop_name}"
            self.handlers[step_name] = fn

            logger.info(
                f"✅ Loop '{loop_name}' registered (max={max_iterations}, "
                f"break={break_condition or continue_condition or 'none'})"
            )
            return fn

        return decorator

    # ===================================================================
    # STEPS — ORIGINAL AGENT BINDING LOGIC (100% unchanged)
    # ===================================================================
    def set(
        self,
        workflow_name: str = "default",
        *,
        toolsets: Union[str, List[str], None] = None,
        tools: Union[str, List[str], None] = None,
        hidden_toolsets: Union[str, List[str], None] = None,
        dynamic_allowed: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
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

            norm_toolsets = self._normalize_list(toolsets)
            norm_tools = self._normalize_list(tools)
            norm_hidden = self._normalize_list(hidden_toolsets)

            # ====================== ORIGINAL AGENT BINDING LOGIC ======================
            sig = signature(fn)  # ← now correctly imported
            param_list = list(sig.parameters.items())
            skip_first = bool(param_list and param_list[0][0] in ("agent", "self"))

            if skip_first:
                if iscoroutinefunction(fn):  # ← now correctly imported

                    @wraps(fn)
                    async def async_step_wrapper(*args, **kwargs):
                        if args and (
                            hasattr(args[0], "agent_id") or args[0] is not None
                        ):
                            return await fn(*args, **kwargs)

                        agent = getattr(self, "_current_agent", None)
                        if agent is None:
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

            # Register
            self.handlers[step_name] = executor

            if workflow_name not in self.workflows:
                self.workflows[workflow_name] = {
                    "description": kwargs.get(
                        "workflow_description", f"Workflow: {workflow_name}"
                    ),
                    "steps": [],
                }

            if step_name not in self.workflows[workflow_name]["steps"]:
                self.workflows[workflow_name]["steps"].append(step_name)

            self.metadata[step_name] = {
                "description": desc,
                "workflow": workflow_name,
                "toolsets": norm_toolsets,
                "tools": norm_tools,
                "hidden_toolsets": norm_hidden,
                "dynamic_allowed": dynamic_allowed,
                "full_name": fn.__name__,
                "requires_agent": skip_first,
            }

            logger.info(
                f"✅ Step '{step_name}' registered to workflow '{workflow_name}' "
                f"(toolsets={norm_toolsets or '-'}, hidden={norm_hidden or '-'}, "
                f"tools={norm_tools or '-'}, dynamic={dynamic_allowed}, agent_bound={skip_first})"
            )

            return fn

        return decorator

    def _normalize_list(self, value: Union[str, List[str], None]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if item and str(item).strip()]
        return []

    # ===================================================================
    # Engine binding + helpers (unchanged)
    # ===================================================================
    def bind_to_engine(self, engine: "WorkflowEngine", instance=None):
        for step_name, handler in self.handlers.items():
            bound_handler = handler
            if instance is not None:
                bound_handler = handler.__get__(instance, instance.__class__)
            engine.register_step(step_name, bound_handler)
            logger.info(f"🔗 Bound step '{step_name}' to engine")

        registered_count = 0
        for wf_name, wf_meta in self.workflows.items():
            if wf_name not in engine.registered_workflows:
                engine.register_workflow(
                    name=wf_name,
                    description=wf_meta.get("description", f"Workflow: {wf_name}"),
                    steps=wf_meta.get("steps", []),
                )
                registered_count += 1

        if registered_count > 0:
            logger.info(f"✅ Synced {registered_count} workflows from decorators")

    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(step_name)

    def get_loop_metadata(self, loop_name: str) -> Optional[Dict[str, Any]]:
        return self.loops.get(loop_name)

    def get_condition(self, name: str) -> Optional[Callable]:
        return self.conditions.get(name)

    def debug_print(self):
        print("\n" + "=" * 120)
        print("📋 WORKFLOW REGISTRY — Steps | Loops | Conditions")
        print("=" * 120)
        # ... (same debug output as before)
        for wf_name, wf_meta in self.workflows.items():
            print(f"\n🔹 Workflow: {wf_name} ({len(wf_meta['steps'])} steps)")
            for step in wf_meta["steps"]:
                meta = self.metadata.get(step, {})
                print(
                    f"   • {step:35} toolsets={meta.get('toolsets') or '-':20} hidden={meta.get('hidden_toolsets') or '-':15}"
                )

        print(f"\n🔄 Loops ({len(self.loops)}):")
        for name, meta in self.loops.items():
            bc = meta.get("break_condition") or meta.get("continue_condition") or "—"
            print(f"   • {name:30} max={meta['max_iterations']:2}  break={bc}")

        print(f"\n✅ Conditions ({len(self.conditions)}):")
        for name in sorted(self.conditions.keys()):
            print(f"   • {name}")

        print(
            f"\nTotal: {len(self.workflows)} workflows | {len(self.metadata)} steps | "
            f"{len(self.loops)} loops | {len(self.conditions)} conditions"
        )
        print("=" * 120)


# Global singleton (replaces both old workflow and condition)
workflow = Workflow()
