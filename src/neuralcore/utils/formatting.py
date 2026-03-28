import re
import json

import asyncio

from typing import List, Dict, Any, Optional, Union

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


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
