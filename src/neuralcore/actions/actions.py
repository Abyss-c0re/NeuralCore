import asyncio

from typing import Any, Callable, Dict, List, Optional, Awaitable
from inspect import signature

from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


class Action:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        executor: Callable,
        required: Optional[List[str]] = None,
        action_type: str = "tool",
        strict: bool = False,
        require_confirmation: bool = False,
        confirmation_preview: Optional[Callable[[dict], str]] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ):
        if action_type not in {"tool", "function"}:
            raise ValueError("action_type must be 'tool' or 'function'")

        self.name = name
        self.description = description
        self.executor = executor
        self.type = action_type
        self.strict = strict
        self.tags = tags or []
        self.usage_count = 0
        self.aliases = aliases or []

        self.require_confirmation = require_confirmation
        self.confirmation_preview = confirmation_preview or (
            lambda kwargs: f"Executing {name} with {kwargs}"
        )

        self._agent = None  # hidden runtime binding

        params_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                k: v for k, v in parameters.items() if k not in ("agent", "self")
            },
        }

        if required:
            required = [r for r in required if r not in ("agent", "self")]
            params_schema["required"] = required

        if strict:
            params_schema["additionalProperties"] = False

        self._raw_schema = {
            "name": name,
            "description": description,
            "parameters": params_schema,
        }

        self._search_text = " ".join(
            [self.name, self.description, " ".join(self.tags)]
        ).lower()

        sig = signature(executor)
        params = list(sig.parameters.values())
        self._needs_agent = bool(params and params[0].name in ("self", "agent"))

    def bind_agent(self, agent):
        self._agent = agent
        return self

    async def __call__(self, **kwargs) -> Any:
        logger.info(f"[ACTION START] {self.name}")
        logger.debug(f"[ACTION INPUT] {self.name} kwargs={kwargs}")

        if self.require_confirmation:
            preview = self.confirmation_preview(kwargs)
            logger.info(f"[ACTION CONFIRMATION REQUIRED] {self.name} preview={preview}")
            raise ConfirmationRequired(self.name, kwargs, preview)

        try:
            call_args = []

            if self._needs_agent:
                if self._agent is None:
                    raise RuntimeError(
                        f"Action '{self.name}' expects agent/self, but no agent is bound"
                    )
                call_args.append(self._agent)

            result = self.executor(*call_args, **kwargs)

            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                logger.debug(f"[ACTION AWAITING] {self.name} awaiting async result")
                result = await result

            self.usage_count += 1

            logger.debug(
                f"[ACTION RAW RESULT] {self.name} type={type(result).__name__} result={str(result)[:500]}"
            )

            if result is None or result == "" or result == {}:
                normalized = {
                    "status": "success",
                    "action": self.name,
                    "message": f"{self.name} executed successfully",
                    "args": kwargs,
                }
                logger.info(f"[ACTION NORMALIZED EMPTY RESULT] {self.name}")
                logger.debug(f"[ACTION OUTPUT] {self.name} result={normalized}")
                return normalized

            logger.info(f"[ACTION SUCCESS] {self.name}")
            logger.debug(f"[ACTION OUTPUT] {self.name} result={str(result)[:500]}")
            return result

        except ConfirmationRequired:
            raise

        except Exception as exc:
            logger.error(f"[ACTION ERROR] {self.name} error={exc}", exc_info=True)
            return {
                "status": "error",
                "action": self.name,
                "error": str(exc),
                "args": kwargs,
            }
        finally:
            self._agent = None


class ActionSet:
    """
    Manages a collection of Actions.
    Provides OpenAI-compatible `tools` list for LLMClient.
    """

    def __init__(
        self,
        name: str = "Actions",
        description: str = "",  # ← new
        actions: Optional[List[Action]] = None,  # optional convenience
    ):
        self.name = name
        self.description = description.strip()  # ← new
        self.actions: List[Action] = []
        self.by_name: Dict[str, Action] = {}

        if actions:
            for action in actions:
                self.add(action)

    def add(self, action: Action) -> None:
        if action.name in self.by_name:
            logger.error(f"[ACTIONSET DUPLICATE] {action.name}")
            raise ValueError(f"Action '{action.name}' already exists in this set")

        self.actions.append(action)
        self.by_name[action.name] = action

        logger.info(f"[ACTIONSET ADD] {action.name}")
        logger.debug(f"[ACTIONSET STATE] total_actions={len(self.actions)}")

    def get_llm_tools(
        self, include_tools: bool = True, include_functions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Returns list in modern OpenAI `tools` format, ready to pass to LLMClient.
        """
        tools = []
        for action in self.actions:
            if (include_tools and action.type == "tool") or (
                include_functions and action.type == "function"
            ):
                tool_spec = {
                    "type": "function",
                    "function": {
                        "name": action._raw_schema["name"],
                        "description": action._raw_schema["description"],
                        "parameters": action._raw_schema["parameters"],
                    },
                }
                if action.strict:
                    tool_spec["function"]["strict"] = True
                tools.append(tool_spec)
        return tools

    def get_executor(self, name: str, agent=None) -> Optional[Action]:
        action = self.by_name.get(name)
        if not action:
            logger.warning(f"[ACTIONSET RESOLVE FAIL] No executor for '{name}'")
            return None

        logger.debug(f"[ACTIONSET RESOLVE] Found executor for '{name}'")

        if agent is not None:
            action.bind_agent(agent)

        return action

    def describe(self) -> Dict[str, Any]:  # ← new helper
        """Returns a lightweight metadata dict useful for tool search / routing."""
        return {
            "name": self.name,
            "description": self.description,
            "action_count": len(self.actions),
            "action_names": [a.name for a in self.actions],
            # You can add more discoverable fields later, e.g.:
            # "categories": [...],
            # "domain": "web" | "math" | "files" | ...
        }

    def remove(self, action: Action) -> None:
        """Remove a specific Action instance from the set."""
        if action not in self.actions:
            return
        self.actions.remove(action)
        self.by_name.pop(action.name, None)

    def remove_by_name(self, name: str) -> None:
        """Remove an action by its name (most convenient for unloading)."""
        action = self.by_name.pop(name, None)
        if action is not None:
            self.actions.remove(action)

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        desc_part = f" – {self.description[:60]}..." if self.description else ""
        return f"<ActionSet '{self.name}' ({len(self)} actions){desc_part}>"
