import sys
import importlib
from pathlib import Path
from neuralcore.actions.manager import registry

from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


def load_tool_sets(loader, app_root: Path, sets_to_load: list[str] | None = None):
    """
    Load all tool sets for the given list of set names.
    If sets_to_load is None, load all sets in the config.

    Each set can either specify:
    - folder: a path to .py files (external)
    - default internal folder: app_root/tools/<set_name>/
    """
    sets_cfg = loader.config.get("tools", {})

    logger.debug(f" App root: {app_root}")
    logger.debug(f" Sets to load: {sets_to_load or 'ALL'}")
    logger.debug(f" Found sets in config: {list(sets_cfg.keys())}")

    for set_name, cfg in sets_cfg.items():
        if sets_to_load and set_name not in sets_to_load:
            logger.debug(f" Skipping set '{set_name}' (not requested)")
            continue

        logger.debug(f" Loading tool set '{set_name}'")

        folder = cfg.get("folder")

        # Determine folder path
        if folder:
            folder_path = Path(folder).expanduser().resolve()
            logger.debug(f" Using external folder for set '{set_name}': {folder_path}")
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"[Warning] Tool folder '{folder_path}' does not exist")
                continue
        else:
            folder_path = app_root / "tools" / set_name
            if not folder_path.exists():
                folder_path = app_root / "tools"
                if not folder_path.exists():
                    print(f"[Warning] No tools folder for set '{set_name}'")
                    continue
                else:
                    logger.debug(
                        f" Fallback folder for set '{set_name}': {folder_path}"
                    )
            else:
                logger.debug(
                    f" Using internal folder for set '{set_name}': {folder_path}"
                )

        # Temporarily add folder to sys.path and import all .py files
        sys.path.insert(0, str(folder_path))
        imported_any = False
        for py_file in folder_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            try:
                importlib.import_module(py_file.stem)
                logger.info(f" Imported '{py_file.name}' for set '{set_name}'")
                imported_any = True
            except Exception as e:
                logger.error(f" Failed to import {py_file.name}: {e}")
        sys.path.pop(0)

        # Check registry
        if imported_any:
            if registry.sets.get(set_name):
                logger.info(f" Tool set '{set_name}' registered successfully")
            else:
                # The module imported but the set name is not exactly matching
                print(
                    f"[Warning] Tool set '{set_name}' imported but not found in registry"
                )
        else:
            print(f"[Warning] No .py files imported for set '{set_name}'")
