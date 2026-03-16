import aiofiles


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
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error while reading file '{file_path}': {str(e)}"
