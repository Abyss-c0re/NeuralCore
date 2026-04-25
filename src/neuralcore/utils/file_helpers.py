import os
import base64
import asyncio
import aiofiles
from PIL import Image
from io import BytesIO
from typing import AsyncIterable, List
from contextlib import asynccontextmanager

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()

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
    ".pytest_cache",
    ".mypy_cache",
    "target",
    "Cargo.lock",
    ".next",
    ".nuxt",
    "coverage",
    "htmlcov",
}
IGNORE_FILES = {".DS_Store", "Thumbs.db", ".gitignore", "*.pyc", "*.pyo", "*.pyd"}


# ─────────────────────────────────────────────────────────────
# ASYNC TEXT FILE DETECTOR 
# ─────────────────────────────────────────────────────────────
async def _is_text_file(file_path: str, sample_size: int = 8192) -> bool:
    """
    Async content-based text detection.
    Replaces all hardcoded extension lists.
    """
    if not os.path.isfile(file_path) or os.path.islink(file_path):
        return False
    try:
        async with aiofiles.open(file_path, "rb") as f:
            sample = await f.read(sample_size)
        if not sample:
            return True
        if b"\x00" in sample:
            return False
        control_chars = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
        return (control_chars / len(sample)) < 0.06
    except (PermissionError, OSError, asyncio.TimeoutError):
        return False
    except Exception as e:
        logger.debug(f"_is_text_file failed for {file_path}: {e}")
        return False



# ─────────────────────────────────────────────────────────────
# UNIVERSAL TEXT FILE COLLECTOR 
# ─────────────────────────────────────────────────────────────
async def _collect_text_files(
    folder_path: str,
    recursive: bool = True,
    max_files: int = 200,
    sort_by: str = "name",  # "name" | "path" | "size"
) -> List[str]:
    """
    Recursively collect text files using async content detection.
    No extensions hardcoded. Respects IGNORE_* and max_files.
    """
    if not os.path.isdir(folder_path):
        return []

    candidates: List[str] = []
    files_count = 0

    for root, dirs, files in os.walk(folder_path, followlinks=False):
        if not recursive:
            dirs.clear()

        # Filter ignored dirs in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in sorted(files):
            if f in IGNORE_FILES:
                continue

            file_path = os.path.join(root, f)
            candidates.append(file_path)
            files_count += 1

            if files_count >= max_files * 2:  # small buffer for filtering
                break

        if files_count >= max_files * 2:
            break

    if not candidates:
        return []

    # Parallel text checks (much faster than sequential for large folders)
    tasks = [_is_text_file(p) for p in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    text_files = [p for p, ok in zip(candidates, results) if ok is True][:max_files]

    # Optional sorting
    if sort_by == "name":
        text_files.sort(key=os.path.basename)
    elif sort_by == "size":
        text_files.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)

    return text_files


async def _read_pdf_file(file_path: str) -> str | AsyncIterable[str]:
    """Internal helper for PDF files."""
    try:
        from pypdf import PdfReader

        if not os.path.isfile(file_path):
            return f"Error: PDF '{file_path}' not found."

        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        if total_pages <= 8:
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip() or "No text extracted."

        # Large PDF → streaming in batches
        async def stream_pdf():
            batch: List[str] = []
            batch_size = 10

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    batch.append(f"--- Page {i + 1}/{total_pages} ---\n{page_text}\n")

                if len(batch) >= batch_size or i == total_pages - 1:
                    if batch:
                        yield "".join(batch)
                    batch.clear()

                if i % batch_size == 0:
                    await asyncio.sleep(0)

        return stream_pdf()

    except Exception as e:
        logger.error(f"_read_pdf_file error: {e}", exc_info=True)
        return f"Error reading PDF '{file_path}': {str(e)}"


async def _read_docx_file(file_path: str) -> str | AsyncIterable[str]:
    """Internal helper for DOCX files."""
    try:
        from docx import Document

        if not os.path.isfile(file_path):
            return f"Error: DOCX '{file_path}' not found."

        doc = Document(file_path)
        paragraphs = list(doc.paragraphs)

        if len(paragraphs) <= 60:
            text = "\n".join(para.text for para in paragraphs)
            return text.strip() or "No text extracted."

        async def stream_docx():
            batch: List[str] = []
            batch_size = 25

            for i, para in enumerate(paragraphs):
                if para.text.strip():
                    batch.append(para.text + "\n")

                if len(batch) >= batch_size or i == len(paragraphs) - 1:
                    if batch:
                        yield "".join(batch)
                    batch.clear()

                if i % batch_size == 0:
                    await asyncio.sleep(0)

        return stream_docx()

    except Exception as e:
        logger.error(f"_read_docx_file error: {e}", exc_info=True)
        return f"Error reading DOCX '{file_path}': {str(e)}"


async def _read_image_file(
    agent, file_path: str, prompt: str = "Describe this image in detail."
) -> str:
    """Internal helper for images (uses vision model)."""
    try:
        if not os.path.isfile(file_path):
            return f"Error: Image '{file_path}' not found."

        loop = asyncio.get_running_loop()

        def _encode():
            with Image.open(file_path) as img:
                img.thumbnail((1024, 1024))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

        base64_img = await loop.run_in_executor(None, _encode)
        description = await agent.client.describe_image(
            image_base64=base64_img, prompt=prompt
        )
        return f"Image description: {description}"

    except Exception as e:
        logger.error(f"_read_image_file error: {e}", exc_info=True)
        return f"Error processing image '{file_path}': {str(e)}"


async def _read_text_file(file_path: str) -> str | AsyncIterable[str]:
    """Internal helper for plain text files with proper streaming."""
    try:
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' not found."

        # Small file → fast non-streaming path
        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            preview = await f.read(4096)
            if len(preview) < 4096:
                return preview

        # Large file → streaming
        @asynccontextmanager
        async def open_file():
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                yield f

        async def stream_file() -> AsyncIterable[str]:
            async with open_file() as f:
                chunk = await f.read(4096)
                if chunk:
                    yield chunk
                while True:
                    chunk = await f.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return stream_file()

    except Exception as e:
        logger.error(f"_read_text_file error for '{file_path}': {e}", exc_info=True)
        return f"Error reading text file '{file_path}': {str(e)}"


async def _read_file(
    agent, file_path: str, image_prompt: str = "Describe this image in detail."
) -> str | AsyncIterable[str]:
    """Main entry point for reading any file.
    Dispatches to the correct internal handler based on file extension."""
    try:
        if not os.path.exists(file_path):
            return f"Error: Path '{file_path}' does not exist."

        if os.path.isdir(file_path):
            return f"Error: '{file_path}' is a directory. Use read_folder instead."

        ext = os.path.splitext(file_path)[1].lower()

        # Dispatch based on extension
        if ext in {".pdf"}:
            return await _read_pdf_file(file_path)

        elif ext in {".docx"}:
            return await _read_docx_file(file_path)

        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}:
            return await _read_image_file(agent, file_path, image_prompt)

        else:
            # Default: treat as text file (.txt, .md, .py, .json, .log, etc.)
            return await _read_text_file(file_path)

    except Exception as e:
        logger.error(
            f"read_file dispatcher error for '{file_path}': {e}", exc_info=True
        )
        return f"Error reading '{file_path}': {str(e)}"
