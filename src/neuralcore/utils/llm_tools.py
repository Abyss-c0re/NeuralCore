from typing import List, Callable, Any, Optional
from neuralcore.actions.actions import Action, ActionSet
import inspect


class InternalTools:
    """
    Wraps a client (LLMClient) and optional agent.run method into an ActionSet.
    Generates parameters & tags automatically, avoiding 'unexpected kwargs' issues.

    Supports:
    - include/exclude filters for method registration
    - lazy registration of additional methods after initialization
    """

    def __init__(
        self,
        client: Any,
        description: str,
        methods: Optional[List[Callable]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.client = client
        self.description = description
        self.include: List[str] = include if include is not None else []
        self.exclude: List[str] = exclude if exclude is not None else []
        self._action_set: Optional[ActionSet] = None

        # Auto-discover methods if not provided
        self.methods: List[Callable] = methods or [
            getattr(client, m)
            for m in dir(client)
            if callable(getattr(client, m)) and not m.startswith("_")
        ]

        # Apply include/exclude filters
        if self.include:
            self.methods = [m for m in self.methods if m.__name__ in self.include]
        if self.exclude:
            self.methods = [m for m in self.methods if m.__name__ not in self.exclude]

    def _infer_tags(self, name: str, doc: str) -> List[str]:
        tags = ["internal", "llm", "ai", "tool", "action"]
        lname, ldoc = name.lower(), (doc or "").lower()
        if "chat" in lname or "messages" in ldoc:
            tags += ["chat", "conversation", "dialog"]
        if "ask" in lname:
            tags += ["qa", "question", "answer"]
        if "run" in lname:
            tags += ["agent", "workflow", "task", "delegate"]
        if "stream" in lname:
            tags += ["stream", "realtime", "real-time"]
        if "image" in ldoc or "vision" in ldoc:
            tags += ["vision", "image", "multimodal"]
        if "code" in ldoc:
            tags += ["code", "programming"]
        if "math" in ldoc:
            tags += ["math", "calculation"]
        if "plan" in ldoc or "goal" in ldoc:
            tags += ["planning", "strategy"]
        return list(set(tags))

    def _create_action_from_method(self, method: Callable) -> Action:
        sig = inspect.signature(method)
        parameters, required = {}, []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            p_type = "string"
            desc = f"{name} parameter"

            if "prompt" in name.lower():
                desc = "User prompt or question"
            elif "system" in name.lower():
                desc = "System prompt / role instruction"
            elif "temperature" in name.lower():
                p_type, desc = "number", "Sampling temperature (0–2.0)"
            elif "max_tokens" in name.lower():
                p_type, desc = "integer", "Maximum tokens to generate"
            elif "messages" in name.lower():
                p_type, desc = "array", "List of chat messages"

            parameters[name] = {"type": p_type, "description": desc}
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                parameters[name]["default"] = param.default

        async def executor_wrapper(**provided_kwargs):
            sig = inspect.signature(method)
            bound_args = {}
            extra_kwargs = {}

            for name, value in provided_kwargs.items():
                if name == "kwargs":
                    # skip literal 'kwargs' key
                    continue
                if name in sig.parameters:
                    bound_args[name] = value
                else:
                    # Only send unknown args if method supports **kwargs
                    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        extra_kwargs[name] = value

            result = method(**bound_args, **extra_kwargs)
            if inspect.iscoroutine(result):
                result = await result
            return result


        doc = inspect.getdoc(method) or f"Call {method.__name__}"
        return Action(
            name=method.__name__,
            description=doc.split("\n")[0],
            parameters=parameters,
            executor=executor_wrapper,
            required=required,
            tags=self._infer_tags(method.__name__, doc),
            action_type="tool",
        )

    def register_method(self, method: Callable):
        """Register a single method after initialization."""
        if self._action_set:
            try:
                self._action_set.add(self._create_action_from_method(method))
            except Exception as e:
                print(f"Warning: failed to register {method.__name__}: {e}")
        else:
            self.methods.append(method)

    def as_action_set(self, name: str = "InternalTools") -> ActionSet:
        """Convert selected methods into an ActionSet."""
        if self._action_set:
            return self._action_set

        action_set = ActionSet(name=name)
        action_set.description = self.description

        for method in self.methods:
            try:
                action_set.add(self._create_action_from_method(method))
            except Exception as e:
                print(f"Warning: failed to wrap {method.__name__}: {e}")

        self._action_set = action_set
        return action_set
