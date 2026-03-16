import aiofiles
from src.neuralcore.actions.actions import Action, ActionSet


def exec_write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    Write or append full content to a file.
    Designed specifically for LLMs: pass multi-line code, JSON, Markdown, etc.
    No shell escaping issues — content is written exactly as provided.
    """
    mode = "a" if append else "w"
    try:
        with open(file_path, mode, encoding="utf-8") as f:
            # Ensure clean ending (optional but nice for code files)
            if content and not content.endswith("\n"):
                content += "\n"
            f.write(content)

        action = "Appended to" if append else "Wrote"
        return (
            f"{action} '{file_path}' "
            f"({len(content)} characters, {content.count(chr(10)) + 1} lines)"
        )
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


def exec_replace_block(
    file_path: str, old_content: str, new_content: str, replace_all: bool = False
) -> str:
    """
    Surgically replace an exact block of text (multi-line OK) in a file.
    Designed for LLMs: copy-paste the exact old code block → new code block.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        count = text.count(old_content)

        if count == 0:
            return f"Error: old_content not found in '{file_path}'"

        if count > 1 and not replace_all:
            return (
                f"Error: old_content appears {count} times (not unique). "
                f"Make it more unique by including more surrounding code, "
                f"or set replace_all=True to replace every occurrence."
            )

        if replace_all:
            new_text = text.replace(old_content, new_content)
        else:
            new_text = text.replace(old_content, new_content, 1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        replaced = count if replace_all else 1
        return (
            f"Replaced {replaced} occurrence(s) in '{file_path}'\n"
            f"   Old block: {len(old_content)} chars → New block: {len(new_content)} chars"
        )

    except FileNotFoundError:
        return f"File not found: '{file_path}'"
    except Exception as e:
        return f"Error during replace: {str(e)}"

# Synchronous File Open Function
def open_file_sync(file_path: str) -> str:
    """
    Open a file synchronously and read its content.
    Returns the content of the file or an error message.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error while reading file '{file_path}': {str(e)}"

# Asynchronous File Open Function
async def open_file_async(file_path: str) -> str:
    """
    Open a file asynchronously and read its content.
    Returns the content of the file or an error message.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error while reading file '{file_path}': {str(e)}"

# Synchronous File Open Action
open_file_sync_action = Action(
    name="open_file_sync",
    description=(
        "Open a file synchronously and read its content. "
        "Use this for blocking file I/O operations where you need to read file contents in a single step."
    ),
    tags=[
        "file",
        "filesystem",
        "read",
        "open",
        "sync",
        "blocking",
        "text",
        "content",
        "file_io",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to open and read."
        },
    },
    executor=open_file_sync,
    required=["file_path"],
)

# Asynchronous File Open Action
open_file_async_action = Action(
    name="open_file_async",
    description=(
        "Open a file asynchronously and read its content. "
        "Ideal for non-blocking file I/O operations where you need to perform other tasks while waiting for file reading to complete."
    ),
    tags=[
        "file",
        "filesystem",
        "read",
        "open",
        "async",
        "non-blocking",
        "text",
        "content",
        "file_io",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to open and read asynchronously."
        },
    },
    executor=open_file_async,
    required=["file_path"],
)

write_file_action = Action(
    name="write_file",
    description=(
        "Create or overwrite a file with full content. "
        "Use this when generating code files (Python, JS, etc.), configs, "
        "or any multi-line text. LLM can output clean, properly indented code "
        "directly in the 'content' parameter — no escaping needed."
    ),
    tags=[
        "file",
        "filesystem",
        "write",
        "create",
        "save",
        "generate",
        "code",
        "script",
        "config",
        "project",
        "overwrite",
        "append",
        "developer",
        "programming",
        "text",
        "source",
    ],
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the file to create/overwrite",
        },
        "content": {
            "type": "string",
            "description": (
                "Full file content. Paste entire code blocks here. "
                "Supports newlines, quotes, indentation — everything."
            ),
        },
        "append": {
            "type": "boolean",
            "description": "Append to existing file instead of overwriting",
            "default": False,
        },
    },
    executor=exec_write_file,
    required=["file_path", "content"],
)

replace_block_action = Action(
    name="replace_block",
    description=(
        "Surgically edit a file by replacing an exact code block or text section. "
        "Best tool for LLM code editing. "
        "1. Use 'cat' or 'read_file' first to see the code. "
        "2. Copy the EXACT old block (including indentation) into 'old_content'. "
        "3. Paste the new version into 'new_content'. "
        "Works perfectly for functions, classes, JSON blocks, etc. "
        "Safe: refuses if block is not unique unless replace_all=True."
    ),
    tags=[
        "file",
        "filesystem",
        "edit",
        "modify",
        "replace",
        "patch",
        "update",
        "refactor",
        "code",
        "function",
        "class",
        "json",
        "config",
        "source",
        "developer",
        "programming",
        "text",
        "surgical",
    ],
    parameters={
        "file_path": {"type": "string", "description": "Path to the file to edit"},
        "old_content": {
            "type": "string",
            "description": (
                "Exact text to find and replace (multi-line code block OK). "
                "Copy-paste directly from cat output — no escaping needed."
            ),
        },
        "new_content": {
            "type": "string",
            "description": (
                "New text to insert in its place (full new code block). "
                "Must have correct indentation and formatting."
            ),
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace ALL occurrences instead of just the first",
            "default": False,
        },
    },
    executor=exec_replace_block,
    required=["file_path", "old_content", "new_content"],
)


# ─────────────────────────────────────────────────────────────
# Putting it all together
# ─────────────────────────────────────────────────────────────
def get_file_actions():
    file_tools = ActionSet(
        name="FileEditingTools",
        description=(
            "Safe, targeted tools for creating and modifying file contents: "
            "write new files from scratch, or perform precise block replacements within existing files. "
            "Does **not** include deletion, moving, copying, directory creation, or navigation commands."
        ),
    )

    for act in [
        open_file_sync_action,
        open_file_async_action,
        write_file_action,
        replace_block_action,
    ]:
        file_tools.add(act)
    return file_tools
