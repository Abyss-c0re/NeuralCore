from typing import Any, Dict, Callable, List, Union, Optional


from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


# ===================================================================
# WORKFLOW — Decorator-based registry (exactly like @tool)
# ===================================================================


logger = Logger.get_logger()


class Workflow:
    """
    Enhanced Workflow Registry with per-step tool (action set) assignment.

    This is the recommended way to control which tools are available
    for each step in a workflow.
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.workflows: Dict[
            str, Dict[str, Any]
        ] = {}  # workflow_name → workflow metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}  # step_name → step metadata

    def set(
        self,
        workflow_name: str = "default",
        *,
        toolsets: Union[str, List[str], None] = None,
        tools: Union[str, List[str], None] = None,
        dynamic_allowed: bool = True,
        workflow_description: Optional[str] = None,
        **kwargs,
    ):
        """
        Decorator to register a workflow step with explicit tool configuration.

        Args:
            workflow_name: Name of the workflow this step belongs to
            toolsets: Toolsets (ActionSets) to load for this step (recommended)
            tools: Specific individual tool names to load
            dynamic_allowed: Whether to allow 'browse_tools' for dynamic discovery
            workflow_description: Optional description for the entire workflow

        Example:
            @workflow.set("ResearchWorkflow",
                          toolsets=["web_search", "browse_page", "knowledge"],
                          name="gather_information",
                          description="Search and collect relevant data")
            async def _wf_gather_information(agent, query: str):
                ...

            @workflow.set("WritingWorkflow",
                          toolsets=["writing", "editor"],
                          tools=["format_markdown"],
                          dynamic_allowed=False)  # strict mode
            def _wf_write_section(agent, content: str):
                ...
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

            # Normalize toolsets and tools
            normalized_toolsets = self._normalize_list(toolsets)
            normalized_tools = self._normalize_list(tools)

            # Register the handler
            self.handlers[step_name] = fn

            # Auto-create workflow group
            if workflow_name not in self.workflows:
                self.workflows[workflow_name] = {
                    "description": workflow_description or f"Workflow: {workflow_name}",
                    "steps": [],
                }

            if step_name not in self.workflows[workflow_name]["steps"]:
                self.workflows[workflow_name]["steps"].append(step_name)

            # Store rich metadata for this step
            self.metadata[step_name] = {
                "description": description,
                "workflow": workflow_name,
                "toolsets": normalized_toolsets,
                "tools": normalized_tools,
                "dynamic_allowed": dynamic_allowed,
                "full_name": fn.__name__,
            }

            logger.info(
                f"✅ Step '{step_name}' registered to workflow '{workflow_name}' "
                f"(toolsets={normalized_toolsets or 'None'}, "
                f"tools={normalized_tools or 'None'}, "
                f"dynamic_allowed={dynamic_allowed})"
            )

            return fn

        return decorator

    def _normalize_list(self, value: Union[str, List[str], None]) -> List[str]:
        """Convert string or list input into a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if item and str(item).strip()]
        return []

    # ====================== Public Query Methods ======================

    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific step (including tool configuration)."""
        return self.metadata.get(step_name)

    def get_workflow_steps(self, workflow_name: str) -> List[str]:
        """Return list of all step names in a given workflow."""
        return self.workflows.get(workflow_name, {}).get("steps", [])

    def get_workflow_metadata(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for an entire workflow."""
        return self.workflows.get(workflow_name)

    def list_all_steps(self) -> List[Dict[str, Any]]:
        """Debug helper: List all registered steps with their tool settings."""
        steps = []
        for step_name, meta in self.metadata.items():
            steps.append(
                {
                    "step": step_name,
                    "workflow": meta["workflow"],
                    "description": meta["description"],
                    "toolsets": meta["toolsets"],
                    "tools": meta["tools"],
                    "dynamic_allowed": meta["dynamic_allowed"],
                }
            )
        return sorted(steps, key=lambda x: (x["workflow"], x["step"]))

    def register_to(self, agent, step_source: Union[type, object]):
        """
        Register all _wf_* methods from a class or instance into the agent's workflow.
        Note: Tool configuration (toolsets/tools) must still be defined using @workflow.set
        """
        count = 0
        source_class = (
            step_source if isinstance(step_source, type) else step_source.__class__
        )
        workflow_name = getattr(
            source_class, "__workflow_name__", source_class.__name__.lower()
        )

        instance = step_source(agent) if isinstance(step_source, type) else step_source

        for attr_name in dir(instance):
            if attr_name.startswith("_wf_"):
                method = getattr(instance, attr_name)
                if callable(method):
                    step_name = attr_name[4:]
                    # Assuming your engine has _step_handlers
                    if hasattr(agent, "workflow") and hasattr(
                        agent.workflow, "_step_handlers"
                    ):
                        agent.workflow._step_handlers[step_name] = method
                    count += 1

        logger.info(
            f"✅ Registered {count} steps from {instance.__class__.__name__} "
            f"into workflow '{workflow_name}'"
        )

    def debug_print(self):
        """Print a nice overview of all registered workflows and steps."""
        print("\n" + "=" * 90)
        print("📋 WORKFLOW REGISTRY")
        print("=" * 90)

        for wf_name, wf_meta in self.workflows.items():
            print(f"\n🔹 Workflow: {wf_name}")
            print(f"   Description: {wf_meta['description']}")
            print(f"   Steps: {len(wf_meta['steps'])}")

            for step_name in wf_meta["steps"]:
                meta = self.metadata.get(step_name, {})
                ts = meta.get("toolsets", [])
                tl = meta.get("tools", [])
                dyn = "✓" if meta.get("dynamic_allowed", True) else "✗"

                print(
                    f"     • {step_name:25} | toolsets={ts or '-':30} | "
                    f"tools={tl or '-':20} | dynamic={dyn}"
                )

        print(
            f"\n✅ Total Workflows: {len(self.workflows)} | Total Steps: {len(self.metadata)}"
        )
        print("=" * 90 + "\n")


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
