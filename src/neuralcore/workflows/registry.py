from typing import Any, Dict, Callable, Type, Union, Optional


from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


# ===================================================================
# WORKFLOW — Decorator-based registry (exactly like @tool)
# ===================================================================
class Workflow:
    """
    Global workflow registry.
    Decorator: @workflow.set("WorkflowName", name="step_name", description="...")
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.workflows: Dict[
            str, Dict[str, Any]
        ] = {}  # workflow_name → {"description": , "steps": [...]}
        self.metadata: Dict[
            str, Dict
        ] = {}  # step_name → {"description": , "workflow": }

    def set(self, workflow_name: str = "default", **kwargs):
        """
        @workflow.set("MyWorkflow", name="plan_tasks", description="Generates ordered tasks...")
        """

        def decorator(fn: Callable):
            # Resolve step name
            step_name = kwargs.get("name") or fn.__name__
            if step_name.startswith("_wf_"):
                step_name = step_name[4:]

            # Resolve description
            description = kwargs.get(
                "description", fn.__doc__ or f"Workflow step: {step_name}"
            )

            # Register handler
            self.handlers[step_name] = fn

            # Auto-create workflow group
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
                "description": description,
                "workflow": workflow_name,
            }

            logger.info(
                f"✅ Step '{step_name}' registered to workflow '{workflow_name}' → {description[:80]}..."
            )
            return fn

        return decorator

    def register_to(self, agent, step_source: Union[Type, object]):
        """
        Register _wf_* steps from a class or instance into the agent's workflow engine.
        """
        engine = agent.workflow
        count = 0

        # Determine workflow name
        if isinstance(step_source, type):  # class
            workflow_name = getattr(
                step_source, "__workflow_name__", step_source.__name__.lower()
            )
            instance = step_source(agent)
        else:  # instance
            workflow_name = getattr(
                step_source.__class__,
                "__workflow_name__",
                step_source.__class__.__name__.lower(),
            )
            instance = step_source

        # Register all _wf_* methods
        for attr_name in dir(instance):
            if attr_name.startswith("_wf_"):
                method = getattr(instance, attr_name)
                if callable(method):
                    step_name = attr_name[4:]
                    engine._step_handlers[step_name] = method
                    count += 1

        # Collect step names for workflow registration
        steps = [attr[4:] for attr in dir(instance) if attr.startswith("_wf_")]
        engine.register_workflow(
            name=workflow_name,
            description=f"Workflow auto-registered from {instance.__class__.__name__}",
            steps=steps,
        )

        logger.info(
            f"✅ Registered {count} steps from {instance.__class__.__name__} into engine '{workflow_name}'"
        )


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
