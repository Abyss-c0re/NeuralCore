import re
import json
import asyncio

from inspect import _empty
from typing import List, Dict, Any, Optional, Union, get_origin

from neuralcore.actions.actions import Action
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


TOKENIZER = re.compile(r"\b\w+(?:[-_]\w+)*\b")


def _tokenize(text: str) -> list[str]:
    return TOKENIZER.findall(text.lower())


def map_type_to_json(param_annotation):
    """Convert Python type annotation to JSON Schema type."""
    if param_annotation is _empty:
        return "string"  # default type

    # Handle basic types
    if param_annotation in PYTHON_TO_JSON_TYPE:
        return PYTHON_TO_JSON_TYPE[param_annotation]

    # Handle typing generics like list[str], dict[str, int], etc.
    origin = get_origin(param_annotation)
    if origin in PYTHON_TO_JSON_TYPE:
        return PYTHON_TO_JSON_TYPE[origin]

    # Fallback
    print(f"[Warning] Unmapped annotation {param_annotation}, defaulting to 'string'")
    return "string"


def safe_parse_json(raw_text: str):
    """Safely extract and parse JSON from raw LLM output."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    json_text = match.group()

    # Escape unescaped backslashes
    json_text = json_text.replace("\\", "\\\\")
    # Remove invalid control characters
    json_text = re.sub(r"[\x00-\x1f]", "", json_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parsing still failed: {e}")
        return None


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Never crash on Action, custom objects, Pydantic models, etc."""

    def default(o: Any):
        if isinstance(o, Action):  # ← explicit protection
            return {
                "name": getattr(o, "name", "unknown"),
                "type": getattr(o, "type", "tool"),
                # never serialize .executor or internal state
            }
        if hasattr(o, "model_dump"):  # Pydantic v2
            return o.model_dump()
        if hasattr(o, "dict"):  # Pydantic v1
            return o.dict()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        return str(o)

    try:
        return json.dumps(obj, default=default, ensure_ascii=False, **kwargs)
    except Exception:
        return str(obj)  # ultimate fallback — never crash the UI


async def drain_queue_to_string(queue: asyncio.Queue) -> str:
    chunks = []
    while True:
        item = await queue.get()
        if item is None:
            break
        chunks.append(item)
    return "".join(chunks)


def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


@staticmethod
def prepare_chat_messages(
    user_content: Optional[Union[str, List[Dict]]] = None,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build and clean messages for sending to the LLM.
    """
    # Normalize input
    if history is not None:
        messages = [m.copy() for m in history]
    elif isinstance(user_content, list) and all(
        isinstance(m, dict) for m in user_content
    ):
        # Already list of dicts
        messages = [m.copy() for m in user_content]
    elif isinstance(user_content, list) and all(
        isinstance(m, str) for m in user_content
    ):
        # List of strings → wrap
        messages = [{"role": "user", "content": s} for s in user_content]
    elif isinstance(user_content, str):
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = []

    # Merge system prompt
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # --- Now safe to do dedup / thinking ---
    sys_parts = []
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            continue  # skip bad entries
        if m.get("role") == "system":
            content = (m.get("content") or "").strip()
            if content:
                sys_parts.append(content)
        else:
            cleaned.append(m)

    if sys_parts:
        cleaned.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})

    # Remove consecutive assistant messages
    deduped = []
    prev_role = None
    for m in cleaned:
        role = m.get("role")
        if role == "assistant" and prev_role == "assistant":
            continue
        deduped.append(m)
        prev_role = role
    cleaned = deduped

    # Handle thinking
    if enable_thinking and cleaned:
        last = cleaned[-1]
        if last.get("role") == "assistant":
            has_content = bool(last.get("content") and last["content"].strip())
            has_tool_calls = bool(last.get("tool_calls"))

            if has_content:
                if has_tool_calls:
                    last["content"] = ""
                else:
                    cleaned.pop()

        if cleaned and cleaned[-1].get("role") == "assistant":
            cleaned.pop()

    return cleaned
