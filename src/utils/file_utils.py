# test/file_utils.py
# Fully adapted FileUtils — NO manager dependency anymore!
# Works perfectly with the new TUI + ContextManager

import os
import magic
import asyncio
import aiofiles
from typing import Callable, Optional
from PIL import Image
from io import BytesIO
import base64

from src.utils.logger import Logger
from src.config.settings import (
    IGNORE_DOT_FILES,
    SUPPORTED_EXTENSIONS,
    IGNORED_FOLDERS,
    MAX_FILE_SIZE,
    MAX_LINES,
    PROCESS_IMAGES,
    IMG_INPUT_RES,
)

logger = Logger.get_logger()


class FileUtils:
    def __init__(
        self,
        index_file: Optional[Callable] = None,
        add_folder: Optional[Callable] = None,
        image_processor: Optional[Callable] = None,
    ) -> None:
        """
        New clean constructor — no manager needed.
        
        Parameters:
            index_file:       async callable(file_path: str, content: str, folder: bool = False)
                              → points to history_manager.add_file
            add_folder:       callable(structure: dict) → points to history_manager.add_folder_structure
            image_processor:  async callable for vision (optional). Pass None if you don't have vision yet.
        """
        self.index_file = index_file
        self.add_folder = add_folder
        self.image_processor = image_processor
        self.file_locks: dict[str, asyncio.Lock] = {}

    def set_index_functions(self, index_file: Callable, add_folder: Callable) -> None:
        """Kept for backward compatibility if you still use it somewhere."""
        self.index_file = index_file
        self.add_folder = add_folder

    async def process_file_or_folder(self, target: str) -> str | int | None:
        """Same logic as before, but now fully manager-free."""
        target = target.strip()

        if not os.path.exists(target):
            choice = await self.prompt_search(target)
            if not choice or choice == "nothing":
                logger.info(f"Nothing found for target: {target}")
                return -1
            target = choice

        if os.path.isfile(target):
            content = await self.read_file(target)
            if self.index_file and content is not None:
                await self.index_file(target, content)  # folder=False by default
        elif os.path.isdir(target):
            await self.read_folder(target)

        logger.info("File operations complete")
        return "File processing complete"  # returned to the tool → visible in chat

    async def read_file(
        self,
        file_path: str,
        max_file_size: int = MAX_FILE_SIZE,
        max_lines: int = MAX_LINES,
    ) -> str | None:
        try:
            if not self._is_safe_file(file_path):
                logger.info(f"Skipping unsupported file: {file_path}")
                return None

            if file_path not in self.file_locks:
                self.file_locks[file_path] = asyncio.Lock()

            async with self.file_locks[file_path]:
                # === VISION HANDLING (no manager) ===
                if PROCESS_IMAGES and self.image_processor and self._is_image(file_path):
                    logger.info(f"Processing image with vision: {file_path}")
                    description = await self.image_processor(
                        file_path, "Describe this image", True
                    )
                    return f"Image description by vision model: {description}"

                if self._is_image(file_path):
                    logger.info(f"Skipping image (no vision enabled): {file_path}")
                    return None  # prevent trying to read binary as text

                logger.info(f"Reading text file: {file_path}")
                content = await _read_file(file_path)

                if content:
                    n_lines = len(content.splitlines())
                    if os.path.getsize(file_path) > max_file_size or n_lines > max_lines:
                        content = await self._read_last_n_lines(content, max_lines)

                return content

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    # === All the helper methods below are unchanged (only removed UI parts) ===

    def _is_safe_file(self, file_path: str) -> bool:
        if os.path.getsize(file_path) == 0:
            return False
        if any(file_path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return True
        if "." not in os.path.basename(file_path):
            return self._is_text_file(file_path)
        return False

    def _is_text_file(self, file_path: str) -> bool:
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path).startswith("text")
        except Exception:
            return False

    def _is_image(self, file_path: str) -> bool:
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path).startswith("image")
        except Exception:
            return False

    async def _read_last_n_lines(self, file_content: str, num_lines: int) -> str:
        lines = file_content.splitlines()
        start = max(0, len(lines) - num_lines)
        return "\n".join(lines[start:]) or "[File is empty]"

    def generate_structure(
        self,
        folder_path: str,
        root_folder: str,
        prefix: str = "",
        ignored_folders: list = IGNORED_FOLDERS,
        ignore_dot_files: bool = IGNORE_DOT_FILES,
    ) -> dict:
        """Exactly the same as before — used by ContextManager."""
        logger.info(f"Generating folder structure for {folder_path}")
        structure = {}
        folder_name = os.path.basename(folder_path)
        structure[folder_name] = {}

        try:
            items = sorted(os.listdir(folder_path))
        except Exception as e:
            logger.error(f"Error listing folder {folder_path}: {e}")
            return structure

        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item not in ignored_folders:
                if ignore_dot_files and item.startswith("."):
                    continue
                structure[folder_name][item] = self.generate_structure(
                    item_path, root_folder, prefix + "--"
                )
            elif os.path.isfile(item_path):
                relative = os.path.relpath(item_path, root_folder) if root_folder else item_path
                structure[folder_name][item] = relative

        return structure

    async def read_folder(
        self,
        folder_path: str,
        root_folder: str | None = None,
        ignored_folders: list = IGNORED_FOLDERS,
    ) -> str | None:
        if root_folder is None:
            root_folder = folder_path

        all_contents = []

        try:
            logger.info(f"Generating structure for folder: {folder_path}")
            generated_structure = self.generate_structure(folder_path, root_folder)
            if self.add_folder:
                self.add_folder(generated_structure)

            for root, _, files in os.walk(folder_path):
                if any(ignored in root.split(os.sep) for ignored in ignored_folders):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    content = await self.read_file(file_path)
                    if self.index_file and content:
                        await self.index_file(file_path, content, folder=True)
                    elif content:
                        all_contents.append(content.strip())

            return "\n".join(all_contents) if all_contents else None

        except Exception as e:
            logger.error(f"Error reading folder {folder_path}: {e}")
            return None

    async def search_files(self, missing_path: str = "", search_dir: str | None = None) -> list[str]:
        if not search_dir:
            search_dir = os.path.expanduser("~")
        results = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files + dirs:
                if missing_path.lower() in name.lower():
                    results.append(os.path.join(root, name))
        return results

    async def prompt_search(self, missing_path: str) -> str:
        """
        Adapted for TUI: NO popup, NO blocking input.
        Automatically picks the first match (exactly like we did for yes/no prompt).
        """
        results = await self.search_files(missing_path)
        if not results:
            logger.info(f"No matches found for '{missing_path}'")
            return "nothing"

        choice = results[0]
        logger.info(f"Auto-selected first match for '{missing_path}': {choice}")
        return choice


async def _read_file(file_path: str) -> str | None:
    """Same async reader used by ContextManager and FileUtils."""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None