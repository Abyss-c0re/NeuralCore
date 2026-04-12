from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from inspect import signature, iscoroutinefunction

from neuralcore.workflows.engine import WorkflowEngine
from neuralcore.utils.config import get_loader
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class Workflow:
    """
    Unified Workflow Registry with Steps, Loops, Conditions, and Universal Waits.
    100% backward compatible.
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        self.loops: Dict[str, Dict[str, Any]] = {}
        self.conditions: Dict[str, Callable] = {}

        logger.debug(
            "Workflow registry initialized (steps + loops + conditions + waits)"
        )

    # ===================================================================
    # CONDITIONS
    # ===================================================================
    def condition(self, name: str, description: str = ""):
        def decorator(fn: Callable):
            self.conditions[name] = fn
            logger.info(
                f"✅ Condition registered: '{name}' → {description or fn.__doc__ or 'No description'}"
            )
            return fn

        return decorator

    def evaluate_condition(
        self, condition_name: str, state: Any, args: Optional[dict] = None
    ) -> bool:
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
            logger.error(
                f"Error evaluating condition '{condition_name}': {e}", exc_info=True
            )
            return False

    # ===================================================================
    # UNIVERSAL WAIT (clean + mypy-safe)
    # ===================================================================
    def wait(
        self,
        name: Optional[str] = None,
        *,
        wait_type: str = "time",
        description: Optional[str] = None,
        default_seconds: Optional[float] = 5.0,
        default_timeout: Optional[float] = 600.0,
        **default_config,
    ):
        """Decorator for reusable wait points."""

        def decorator(fn: Optional[Callable] = None):
            step_name = name or (fn.__name__ if fn else f"wait_{wait_type}")

            if fn is not None:
                self.handlers[step_name] = fn
                uses_universal = False
                logger.info(f"✅ Custom wait handler registered: '{step_name}'")
            else:
                # Dummy callable - will be replaced in bind_to_engine
                self.handlers[step_name] = lambda *args, **kwargs: None
                uses_universal = True

            if "wait" not in self.workflows:
                self.workflows["wait"] = {
                    "description": "Internal universal wait primitives",
                    "steps": [],
                }

            self.metadata[step_name] = {
                "type": "wait",
                "wait_type": wait_type,
                "description": description or f"Universal wait ({wait_type})",
                "default_config": {
                    "wait_type": wait_type,
                    "seconds": default_seconds,
                    "timeout": default_timeout,
                    **default_config,
                },
                "requires_agent": True,
                "uses_universal_wait": uses_universal,
            }

            logger.info(
                f"✅ Wait primitive '{step_name}' registered "
                f"(type={wait_type}, default_timeout={default_timeout}s)"
            )
            return fn

        return decorator

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
        def decorator(fn: Callable):
            self.loops[loop_name] = {
                "name": loop_name,
                "handler": fn,
                "max_iterations": max_iterations,
                "break_condition": break_condition,
                "continue_condition": continue_condition,
                "description": description or fn.__doc__ or f"Loop: {loop_name}",
            }

            step_name = f"_loop_{loop_name}"
            self.handlers[step_name] = fn

            logger.info(
                f"✅ Loop '{loop_name}' registered (max={max_iterations}, "
                f"break={break_condition or continue_condition or 'none'})"
            )
            return fn

        return decorator

    # ===================================================================
    # STEPS (unchanged)
    # ===================================================================
    def step(
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

            sig = signature(fn)
            param_list = list(sig.parameters.items())
            skip_first = bool(param_list and param_list[0][0] in ("agent", "self"))

            if skip_first:
                if iscoroutinefunction(fn):

                    @wraps(fn)
                    async def async_step_wrapper(*args, **kwargs):
                        if args and (
                            hasattr(args[0], "agent_id") or args[0] is not None
                        ):
                            return await fn(*args, **kwargs)
                        agent = getattr(self, "_current_agent", None) or kwargs.pop(
                            "agent", None
                        )
                        if agent is None:
                            raise RuntimeError(
                                f"Step '{step_name}' expects 'agent' as first parameter"
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
                        agent = getattr(self, "_current_agent", None) or kwargs.pop(
                            "agent", None
                        )
                        if agent is None:
                            raise RuntimeError(
                                f"Step '{step_name}' expects 'agent' as first parameter"
                            )
                        return fn(agent, *args, **kwargs)

                    executor = sync_step_wrapper
            else:
                executor = fn

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
    # BIND TO ENGINE
    # ===================================================================
    def bind_to_engine(self, engine: "WorkflowEngine", instance=None):
        """Bind everything to the engine. Universal waits are specially handled."""
        for step_name, handler in list(self.handlers.items()):
            meta = self.metadata.get(step_name, {})

            if meta.get("type") == "wait" and meta.get("uses_universal_wait", False):
                # Route to universal wait implementation
                if hasattr(engine, "_step_wait"):
                    bound_handler = engine._step_wait
                    logger.info(
                        f"🔗 Bound universal wait '{step_name}' (type={meta.get('wait_type')})"
                    )
                else:
                    logger.warning(f"Engine missing _step_wait for '{step_name}'")
                    bound_handler = handler
            else:
                # Normal step or custom handler
                bound_handler = handler
                if instance is not None:
                    bound_handler = handler.__get__(instance, instance.__class__)

            engine.register_step(step_name, bound_handler)
            logger.info(f"🔗 Bound step '{step_name}' to engine")

        # Register workflows
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

        # Bind conditions
        for cond_name, handler in self.conditions.items():
            engine.register_custom_condition(
                name=cond_name,
                handler=handler,
                description="Auto-bound from @workflow.condition",
            )
            logger.info(f"✅ Bound condition handler: '{cond_name}' to engine")

    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(step_name)

    def get_loop_metadata(self, loop_name: str) -> Optional[Dict[str, Any]]:
        return self.loops.get(loop_name)

    def get_condition(self, name: str) -> Optional[Callable]:
        return self.conditions.get(name)

    def merge_yaml_loop_steps(self, engine: "WorkflowEngine"):
        """Merge steps from YAML 'loops:' section into decorator-defined loops."""
        loader = get_loader()
        for loop_name, loop_cfg in loader.config.get("loops", {}).items():
            if loop_name not in self.loops:
                logger.warning(f"Loop '{loop_name}' defined in YAML but not registered via @workflow.loop")
                continue

            yaml_steps = loop_cfg.get("steps", [])
            if not yaml_steps:
                continue

            if "steps" not in self.loops[loop_name]:
                self.loops[loop_name]["steps"] = []

            self.loops[loop_name]["steps"].extend(yaml_steps)

            logger.info(f"✅ Merged {len(yaml_steps)} YAML steps into loop '{loop_name}' "
                       f"(waits and other steps added)")

    def debug_print(self):
        print("\n" + "=" * 120)
        print("📋 WORKFLOW REGISTRY — Steps | Loops | Conditions | Waits")
        print("=" * 120)

        for wf_name, wf_meta in self.workflows.items():
            print(f"\n🔹 Workflow: {wf_name} ({len(wf_meta['steps'])} steps)")
            for step in wf_meta["steps"]:
                meta = self.metadata.get(step, {})
                wait_info = (
                    f" [wait:{meta.get('wait_type')}]"
                    if meta.get("type") == "wait"
                    else ""
                )
                print(
                    f"   • {step:35} toolsets={meta.get('toolsets') or '-':20} hidden={meta.get('hidden_toolsets') or '-':15}{wait_info}"
                )

        print(f"\n🔄 Loops ({len(self.loops)}):")
        for name, meta in self.loops.items():
            bc = meta.get("break_condition") or meta.get("continue_condition") or "—"
            print(f"   • {name:30} max={meta['max_iterations']:2}  break={bc}")

        print(f"\n✅ Conditions ({len(self.conditions)}):")
        for name in sorted(self.conditions.keys()):
            print(f"   • {name}")

        wait_count = sum(1 for m in self.metadata.values() if m.get("type") == "wait")
        print(
            f"\nTotal: {len(self.workflows)} workflows | {len(self.metadata)} steps | "
            f"{len(self.loops)} loops | {len(self.conditions)} conditions | {wait_count} waits"
        )
        print("=" * 120)


# Global singleton
workflow = Workflow()
