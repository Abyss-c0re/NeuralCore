from typing import List, Callable, Any, Optional
import inspect

from neuralcore.actions.actions import Action, ActionSet


class InternalTools:
    """
    Convenience factory that turns selected methods of a client object
    (usually LLMClient) into an ActionSet that can be passed directly
    to another LLM as tools.
    """

    def __init__(
        self,
        client: Any,
        description: str,
        methods: List[Callable],
        temperature: float = 0.35,
        max_tokens : int = 4096,
    ):
        self.client = client
        self.description = description.strip()
        self.methods = methods
        self.temperature = temperature
        self.max_tokens  = max_tokens 

        self._action_set: Optional[ActionSet] = None

    def _infer_tags(self, name: str, doc: str) -> List[str]:
        """
        Automatically infer useful search tags for the tool.
        """

        base_tags = [
            "llm",
            "model",
            "ai",
            "reasoning",
            "analysis",
            "thinking",
        ]

        name_lower = name.lower()
        doc_lower = doc.lower()

        if "stream" in name_lower:
            base_tags += ["stream", "streaming", "real-time"]

        if "chat" in name_lower:
            base_tags += ["chat", "conversation", "dialog"]

        if "ask" in name_lower:
            base_tags += ["question", "answer", "qa"]

        if "image" in doc_lower or "vision" in doc_lower:
            base_tags += ["vision", "image", "multimodal"]

        if "code" in doc_lower:
            base_tags += ["code", "programming"]

        if "math" in doc_lower:
            base_tags += ["math", "calculation"]

        if "plan" in doc_lower:
            base_tags += ["planning", "strategy"]

        return list(set(base_tags))

    def _create_action_from_method(self, method: Callable) -> Action:
        """Convert a bound method into an Action with reasonable defaults."""

        name = method.__name__

        doc = (inspect.getdoc(method) or "").strip()
        if not doc:
            doc = f"Call {name} on the specialized LLM instance."

        tags = self._infer_tags(name, doc)

        sig = inspect.signature(method)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():

            if param_name == "self":
                continue

            param_type = "string"
            description = ""

            if "messages" in param_name.lower():
                description = "List of chat messages (OpenAI format)"
                param_type = "array"

            elif "prompt" in param_name.lower() or param_name == "user_content":
                description = "User prompt or question"

            elif param_name in ("temperature", "temp"):
                description = "Sampling temperature (0.0–2.0)"
                param_type = "number"

            elif param_name in ("max_tokens", "max_new_tokens"):
                description = "Maximum number of tokens to generate"
                param_type = "integer"

            elif "stream" in param_name.lower():
                description = "Whether to stream the response"
                param_type = "boolean"

            elif param_name == "image_base64":
                description = "Base64-encoded image (for vision models)"

            elif param_name == "system":
                description = "System prompt / role instruction"

            properties[param_name] = {
                "type": param_type,
                "description": description,
            }

            if param.default is not inspect.Parameter.empty:
                properties[param_name]["default"] = param.default
            else:
                if len(required) == 0 and param_name not in ("kwargs",):
                    required.append(param_name)

        async def wrapped_executor(**kwargs) -> Any:
            result = method(**kwargs)
            if inspect.iscoroutine(result):
                result = await result
            return result

        wrapped_executor.__name__ = f"wrapped_{name}"

        return Action(
            name=name,
            description=doc.split("\n")[0].strip() or f"{name} on specialized model",
            parameters=properties,
            executor=wrapped_executor,
            required=required,
            action_type="tool",
            tags=tags,
        )

    def as_action_set(self, name: str = "InternalTools") -> ActionSet:

        if self._action_set is not None:
            return self._action_set

        action_set = ActionSet(name=name)
        action_set.description = self.description

        for method in self.methods:
            try:
                action = self._create_action_from_method(method)
                action_set.add(action)
            except Exception as e:
                print(f"Warning: could not create tool from {method.__name__}: {e}")

        if len(action_set) > 0:
            action_set.description += f"\nContains {len(action_set)} tools."

        self._action_set = action_set
        return action_set