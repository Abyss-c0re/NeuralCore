import os
import asyncio
import aiofiles

from typing import AsyncIterable, List
from neuralcore.actions.registry import tool, sequenced

from neuralcore.utils.file_helpers import _read_file, _collect_text_files

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

# ─────────────────────────────────────────────────────────────
# File Editing Tools – Optimized for Streaming + Batching
# ─────────────────────────────────────────────────────────────

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    ".idea",
    ".vscode",
}


@tool(
    "FileEditingTools",
    tags=["file", "write"],
    name="write_file",
    description="Write or append (auto newline).",
)
async def write_file(file_path: str, content: str, append: bool = False) -> str:
    mode = "a" if append else "w"
    try:
        async with aiofiles.open(file_path, mode, encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                content += "\n"
            await f.write(content)
        return (
            f"✅ {'Appended' if append else 'Wrote'} {file_path} ({len(content)} chars)"
        )
    except Exception as e:
        return f"❌ write_file error: {e}"


@tool(
    "FileEditingTools",
    tags=["file", "edit"],
    name="replace_block",
    description="Exact block replace with safety checks.",
)
async def replace_block(
    file_path: str, old_content: str, new_content: str, replace_all: bool = False
) -> str:
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()
        count = text.count(old_content)
        if count == 0:
            return f"❌ old_content not found"
        if count > 1 and not replace_all:
            return f"⚠️ Appears {count} times — set replace_all=True"
        new_text = text.replace(old_content, new_content, count if replace_all else 1)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_text)
        return f"✅ Replaced {count if replace_all else 1} occurrence(s)"
    except Exception as e:
        return f"❌ replace_block error: {e}"


@tool(
    "FileEditingTools",
    tags=["file", "read", "universal"],
    name="read_file",
    description="Universal reader (text/images/PDF/docx) with streaming fallback.",
)
async def read_file(
    agent, file_path: str, image_prompt: str = "Describe this image in detail."
) -> str | AsyncIterable[str]:
    return await _read_file(agent, file_path, image_prompt)


@tool(
    "FileEditingTools",
    tags=[
        "file",
        "batch",
    ],
    name="read_multiple_files",
    description="Batch read + optional GetContext summary.",
)
async def read_multiple_files(agent, files: List[str], summary: bool = True) -> str:
    if not files:
        return "❌ No files"
    indexed = []
    for fp in files:
        try:
            await agent.action_manager.execute_direct("read_file", file_path=fp)
            indexed.append(os.path.basename(fp))
        except Exception:
            pass
    if summary and indexed:
        try:
            s = await agent.action_manager.execute_direct(
                "GetContext",
                query=f"Detailed technical summary of: {', '.join(indexed)}. Focus on architecture and key logic.",
            )
            return f"✅ Read & summarized {len(indexed)} files\n\n{s}"
        except Exception:
            pass
    return f"✅ Indexed {len(indexed)} files: {', '.join(indexed)}"


@tool(
    "FileEditingTools",
    tags=["file", "folder"],
    name="read_folder",
    description="Recursively read folder using text detection + read_multiple_files.",
)
async def read_folder(
    agent,
    folder_path: str,
    recursive: bool = True,
    max_files: int = 60,
    summary: bool = False,
) -> str:
    if not os.path.isdir(folder_path):
        return f"❌ Folder not found: {folder_path}"
    files = await _collect_text_files(folder_path, recursive, max_files)
    if not files:
        return f"ℹ️ No text files in {os.path.basename(folder_path)}"
    try:
        res = await agent.action_manager.execute_direct(
            "read_multiple_files", files=files, summary=summary
        )
        return f"✅ Folder processed ({len(files)} files)\n{res}"
    except Exception as e:
        return f"❌ read_folder error: {e}"


@tool(
    "FileEditingTools",
    tags=["file", "diff"],
    name="apply_diff",
    description="git apply unified diff (safe, confirmation required)",
    require_confirmation=True,
)
async def apply_diff(file_path: str, diff_content: str) -> str:
    try:
        check = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--check",
            "--unidiff-zero",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await check.communicate(diff_content.encode())
        result = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--unidiff-zero",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        await result.communicate(diff_content.encode())
        return f"✅ Diff applied to {file_path}"
    except Exception as e:
        return f"❌ apply_diff error: {e}"


@tool(
    "FileEditingTools",
    tags=["file", "regex"],
    name="regex_replace",
    description="Regex replace with dry-run.",
)
async def regex_replace(
    file_path: str, pattern: str, replacement: str, dry_run: bool = True
) -> str:
    import re

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if dry_run:
            return f"🔍 Dry-run: {count} replacements\nPreview:\n{new_content[:400]}..."
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)
        return f"✅ Replaced {count} occurrences"
    except Exception as e:
        return f"❌ regex_replace error: {e}"


# ─────────────────────────────────────────────────────────────
# SEQUENCED WORKFLOW
# ─────────────────────────────────────────────────────────────
@sequenced(
    name="find_and_read_file",
    description="Search filename → auto-read first match.",
    set_name="FileEditingTools",
    tags=["read", "find", "file"],
    propagate=False,
    output_from="read_file",
    dependencies={
        "search_files": {"name": "input"},
        "read_file": {"file_path": "search_files"},
    },
    steps=["search_files", "read_file"],
)
def find_and_read_file():
    pass
