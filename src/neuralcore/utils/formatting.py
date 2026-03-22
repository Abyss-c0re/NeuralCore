import re
import json

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