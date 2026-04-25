import os
import json
import heapq
import hashlib
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections.abc import AsyncIterable

import aiofiles

from neuralcore.utils.logger import Logger
from neuralcore.cognition.items import KnowledgeItem
from neuralcore.utils.search import cosine_sim, keyword_score
from neuralcore.utils.file_helpers import _read_file

logger = Logger.get_logger()


class KnowledgeBase:
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.agent = context_manager.agent
        self.loader = context_manager.agent.loader

        # === CONFIG ===
        app_config = self.loader.get_app_config() or {}
        kb_config = app_config.get("knowledge_base", {})

        self.enabled: bool = kb_config.get("enabled", True)

        if not self.enabled:
            logger.info("⚠️  KnowledgeBase is DISABLED via config (enabled: false)")
            self.base_folder = Path("~/.neuralcore/kb_disabled").expanduser()
            self.index: Dict[str, Any] = {"items": {}, "stats": {}}
            self.auto_reindex = False
            return

        # === NORMAL INITIALIZATION ===
        self.base_folder = (
            Path(kb_config.get("base_folder", "~/.neuralcore/kb"))
            .expanduser()
            .resolve()
        )
        self.auto_reindex = kb_config.get("auto_reindex", True)
        self.reindex_interval = kb_config.get("reindex_interval_seconds", 300)

        self.base_folder.mkdir(parents=True, exist_ok=True)
        (self.base_folder / "items").mkdir(exist_ok=True)

        self.index: Dict[str, Any] = {}
        self.index_path = self.base_folder / "index.json"
        self._last_reindex = 0.0

        self._load_index_sync()
        self._reindex_lock = asyncio.Lock()

        logger.info(
            f"✅ KnowledgeBase initialized | root={self.base_folder} | "
            f"items={len(self.index.get('items', {}))}"
        )

    # ====================== START BACKGROUND WATCHER (MISSING BEFORE) ======================
    async def start_background_watcher(self):
        """Start the background reindex watcher with non-blocking first scan."""
        if not (self.enabled and self.auto_reindex):
            logger.info("KnowledgeBase auto-reindex is disabled")
            return

        # === NON-BLOCKING FIRST SCAN ===
        logger.info("🔄 Performing initial reindex scan (background)...")
        asyncio.create_task(self._run_initial_reindex())  # ← Run in background

        # Start periodic watcher
        asyncio.create_task(self._background_watcher())
        logger.info("🔄 Background reindex watcher started (periodic every 5 min)")

    async def _run_initial_reindex(self):
        """Run the first reindex in background without blocking UI."""
        try:
            count = await self.reindex_changed_files()
            if count > 0:
                logger.info(f"✅ Initial reindex completed — {count} files indexed")
            else:
                logger.debug("Initial reindex completed — no changes")
        except Exception as e:
            logger.error(f"Initial reindex failed: {e}", exc_info=True)

    # ====================== INDEX MANAGEMENT ======================
    def _load_index_sync(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.index = {
                        "schema_version": loaded.get("schema_version", 2),
                        "last_updated": loaded.get(
                            "last_updated", datetime.now().isoformat()
                        ),
                        "stats": loaded.get(
                            "stats",
                            {"total_items": 0, "total_chunks": 0, "by_category": {}},
                        ),
                        "items": loaded.get("items", {}),
                    }
            except Exception as e:
                logger.error(f"Failed to load index.json: {e}")
                self.index = self._get_empty_index()
        else:
            self.index = self._get_empty_index()
            self._save_index_sync()

    def _get_empty_index(self):
        return {
            "schema_version": 2,
            "last_updated": datetime.now().isoformat(),
            "stats": {"total_items": 0, "total_chunks": 0, "by_category": {}},
            "items": {},
        }

    async def _save_index(self):
        try:
            tmp_path = self.index_path.with_suffix(".tmp")
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(self.index, indent=2, ensure_ascii=False, default=str)
                )
            tmp_path.replace(self.index_path)
        except Exception as e:
            logger.error(f"Failed to save index.json: {e}")

    def _save_index_sync(self):
        try:
            tmp_path = self.index_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(self.index_path)
        except Exception as e:
            logger.error(f"Failed to save index.json: {e}")

    def _update_stats(self):
        items = self.index.get("items", {})
        by_cat: Dict[str, int] = {}
        total_chunks = 0
        for meta in items.values():
            cat = meta.get("top_category", "unknown")
            by_cat[cat] = by_cat.get(cat, 0) + 1
            total_chunks += meta.get("chunk_count", 0)

        self.index["stats"] = {
            "total_items": len(items),
            "total_chunks": total_chunks,
            "by_category": by_cat,
        }
        self.index["last_updated"] = datetime.now().isoformat()

    # ====================== CATEGORY HELPERS ======================
    def _detect_category_from_path(self, file_path: Path) -> str:
        try:
            rel = file_path.relative_to(self.base_folder)
            parts = rel.parts
            return str(Path(*parts[:-1])) if len(parts) >= 2 else "general"
        except ValueError:
            return "general"

    def _get_item_folder(self, key: str) -> Path:
        return self.base_folder / "items" / key

    # ====================== ADD / INDEX FILE ======================
    async def add_file(
        self, file_path: str, category: Optional[str] = None, source: str = "manual"
    ) -> Optional[str]:
        if not self.enabled:
            return None

        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None

        category_path = category or self._detect_category_from_path(path)
        key = await self._index_single_file(path, category_path, source=source)
        await self._save_index()
        await self._trigger_consolidation()
        return key

    async def _index_single_file(
        self, file_path: Path, category_path: str, source: str = "file"
    ) -> Optional[str]:
        if not self.enabled:
            return None

        content_hash = hashlib.md5(
            str(file_path).encode() + str(time.time()).encode()
        ).hexdigest()[:16]
        parent_key = f"kb_{content_hash}"

        item_folder = self._get_item_folder(parent_key)
        item_folder.mkdir(parents=True, exist_ok=True)

        # === PROPER STREAMING — NO MERGE, NO FULL CONTENT IN MEMORY ===
        try:
            # Direct await (no broken to_thread + asyncio.run)
            content_or_stream = await _read_file(self.agent, str(file_path))
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None

        embeddings_list = []
        content_lines = []
        chunk_count = 0

        if isinstance(content_or_stream, AsyncIterable):
            logger.debug(f"📡 Streaming chunks from {file_path.name} (no full merge)")
            async for raw_text in content_or_stream:
                if not raw_text or len(raw_text.strip()) < 10:
                    continue

                # Chunk each arriving piece independently (merge=False keeps them small)
                sub_items = self.context_manager._chunk_text(
                    raw_text, parent_key, "file", merge=False
                )

                for item in sub_items:
                    txt = item.content.strip()
                    if len(txt) < 20:
                        continue

                    emb = await self.context_manager.fetch_embedding(
                        txt, prefix="passage"
                    )
                    if emb.size > 0:
                        embeddings_list.append(emb)
                        content_lines.append(
                            json.dumps({"chunk_index": chunk_count, "text": txt})
                        )
                        chunk_count += 1

        else:
            # Fallback for non-streaming readers (small files)
            content_str = str(content_or_stream).strip()
            if len(content_str) < 30:
                logger.warning(f"Very little text from {file_path.name}")
                return None

            chunk_items = self.context_manager._chunk_text(
                content_str, parent_key, "file", merge=False
            )

            for item in chunk_items:
                txt = item.content.strip()
                if len(txt) < 20:
                    continue

                emb = await self.context_manager.fetch_embedding(txt, prefix="passage")
                if emb.size > 0:
                    embeddings_list.append(emb)
                    content_lines.append(
                        json.dumps({"chunk_index": chunk_count, "text": txt})
                    )
                    chunk_count += 1

        if not embeddings_list:
            logger.warning(f"No valid embeddings for {file_path.name}")
            return None

        logger.info(f"📄 {file_path.name} → {len(embeddings_list)} chunks (streamed)")

        # === SAVE (unchanged, but now fed from stream) ===
        content_path = item_folder / "content.jsonl"
        async with aiofiles.open(content_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(content_lines))

        emb_dict = {f"chunk_{i}": emb for i, emb in enumerate(embeddings_list)}
        await asyncio.to_thread(
            np.savez_compressed, item_folder / "embeddings.npz", **emb_dict
        )

        # meta + index update (same as before)
        meta = {
            "key": parent_key,
            "source_type": "file",
            "category_path": category_path,
            "top_category": category_path.split("/")[0]
            if "/" in category_path
            else category_path,
            "original_path": str(file_path),
            "hash": hashlib.md5(open(file_path, "rb").read()).hexdigest(),
            "mtime": file_path.stat().st_mtime,
            "size_bytes": file_path.stat().st_size,
            "chunk_count": len(embeddings_list),
            "embedding_dim": len(embeddings_list[0]) if embeddings_list else 0,
            "source": source,
            "added_at": datetime.now().isoformat(),
            "is_merged": False,
        }

        async with aiofiles.open(item_folder / "meta.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(meta, indent=2))

        self.index["items"][parent_key] = {
            "original_path": str(file_path),
            "category_path": category_path,
            "top_category": meta["top_category"],
            "hash": meta["hash"],
            "mtime": meta["mtime"],
            "size_bytes": meta["size_bytes"],
            "chunk_count": len(embeddings_list),
            "embedding_dim": meta["embedding_dim"],
            "archived": False,
            "source": source,
            "added_at": meta["added_at"],
        }

        self._update_stats()
        logger.info(
            f"✅ Indexed {file_path.name} → {category_path} ({len(embeddings_list)} chunks)"
        )
        return parent_key

    # ====================== RETRIEVAL ======================
    async def retrieve(
        self,
        query: str,
        k: int = 12,
        categories: Optional[List[str]] = None,
        top_category: Optional[str] = None,
        include_archived: bool = False,
    ) -> List[KnowledgeItem]:
        if not self.enabled or not query.strip():
            return []

        items = self.index.get("items", {})
        filtered_keys = [
            key
            for key, meta in items.items()
            if (not meta.get("archived") or include_archived)
            and (not top_category or meta.get("top_category") == top_category)
            and (
                not categories
                or any(
                    meta.get("category_path", "").startswith(cat) for cat in categories
                )
            )
        ]

        if not filtered_keys:
            logger.debug("[KB] No matching files for query")
            return []

        # Compute query embedding once
        query_words = query.lower().split()
        query_emb = await self.context_manager.fetch_embedding(query, prefix="query")

        # Min-heap to keep only top-k (memory-safe even with 10k+ chunks)
        top_k: List[Tuple[float, int, KnowledgeItem]] = []
        counter = 0
        loaded_chunks = 0

        for key in filtered_keys:
            meta = items[key]
            item_folder = self._get_item_folder(key)
            emb_path = item_folder / "embeddings.npz"
            content_path = item_folder / "content.jsonl"

            if not emb_path.exists() or not content_path.exists():
                continue

            try:
                emb_data = await asyncio.to_thread(np.load, emb_path)
                async with aiofiles.open(content_path, "r", encoding="utf-8") as f:
                    content_lines = [
                        json.loads(line)
                        for line in (await f.read()).splitlines()
                        if line.strip()
                    ]

                for i, line in enumerate(content_lines):
                    emb_key = f"chunk_{i}"
                    if emb_key not in emb_data:
                        continue

                    loaded_chunks += 1
                    item = KnowledgeItem(
                        key=f"{key}_chunk{i}",
                        source_type="file",
                        content=line["text"],
                        metadata={
                            "category_path": meta["category_path"],
                            "original_path": meta["original_path"],
                            "chunk_index": i,
                            "parent_key": key,
                        },
                    )
                    item.embedding = emb_data[emb_key]

                    kw = keyword_score(query_words, item.content)
                    dense = (
                        cosine_sim(query_emb, item.embedding)
                        if query_emb.size > 0
                        else 0.0
                    )
                    score = 0.6 * dense + 0.4 * kw

                    entry: Tuple[float, int, KnowledgeItem] = (score, counter, item)
                    if len(top_k) < k:
                        heapq.heappush(top_k, entry)
                    elif score > top_k[0][0]:
                        heapq.heapreplace(top_k, entry)
                    counter += 1

            except Exception as e:
                logger.warning(f"Failed to load item {key}: {e}")

        if not top_k:
            return []

        # Sort descending (best first)
        top_k.sort(key=lambda x: x[0], reverse=True)
        result = [item for _, _, item in top_k]

        logger.debug(
            f"[KB] retrieve → {len(result)} chunks (from {loaded_chunks} total, "
            f"kept top-{k}, heap peak={len(top_k)})"
        )
        return result

    # ====================== BACKGROUND REINDEXING ======================
    async def _background_watcher(self):
        while True:
            await asyncio.sleep(self.reindex_interval)
            try:
                await self.reindex_changed_files()
            except Exception as e:
                logger.error(f"Background reindex failed: {e}")

    async def reindex_changed_files(self) -> int:
        if not self.enabled:
            return 0

        async with self._reindex_lock:  # ← Prevent double runs
            logger.info("🔄 Starting incremental reindex scan...")
            indexed_count = 0

            SKIP_FILES = {"content.jsonl", "embeddings.npz", "meta.json", "index.json"}
            skip_dirs = {"_manual", "_archive", "__pycache__", "items"}

            for root, dirs, files in os.walk(self.base_folder):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                if "items" in Path(root).parts:
                    continue

                for filename in files:
                    if filename.startswith(".") or filename in SKIP_FILES:
                        continue

                    file_path = Path(root) / filename
                    if any(part.startswith("kb_") for part in file_path.parts):
                        continue

                    try:
                        rel_path = file_path.relative_to(self.base_folder)
                        category_path = (
                            str(rel_path.parent)
                            if rel_path.parent != Path(".")
                            else "general"
                        )

                        existing = None
                        for key, meta in self.index.get("items", {}).items():
                            if meta.get("original_path") == str(file_path):
                                existing = meta
                                break

                        current_mtime = file_path.stat().st_mtime
                        current_size = file_path.stat().st_size

                        if (
                            existing
                            and existing.get("mtime") == current_mtime
                            and existing.get("size_bytes") == current_size
                        ):
                            continue

                        await self._index_single_file(
                            file_path, category_path, source="auto"
                        )
                        indexed_count += 1
                        await asyncio.sleep(0)

                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")

            await self._save_index()

            if indexed_count > 0:
                logger.info(f"✅ Reindex complete — {indexed_count} files updated")
                await self._trigger_consolidation()
            else:
                logger.debug("Reindex complete — no changes detected")

            return indexed_count

    # ====================== TRAINING TRIGGER ======================
    async def _trigger_consolidation(self):
        try:
            if (
                hasattr(self.context_manager, "consolidator")
                and self.context_manager.consolidator
            ):
                asyncio.create_task(
                    self.context_manager.consolidator.extract_and_consolidate()
                )
                logger.debug("[KB] Consolidation triggered")
        except Exception as e:
            logger.warning(f"Failed to trigger consolidation: {e}")

    # ====================== UTILITY ======================
    def get_stats(self) -> Dict:
        if not self.enabled:
            return {"enabled": False, "message": "KnowledgeBase is disabled"}
        return self.index.get("stats", {})

    def list_categories(self) -> List[str]:
        cats = set()
        for meta in self.index.get("items", {}).values():
            cats.add(meta.get("category_path", "unknown"))
        return sorted(cats)

    async def clear(self, hard: bool = False):
        self.index = self._get_empty_index()
        await self._save_index()
        if hard:
            import shutil

            shutil.rmtree(self.base_folder, ignore_errors=True)
            self.base_folder.mkdir(parents=True, exist_ok=True)
            (self.base_folder / "items").mkdir(exist_ok=True)
        logger.info("KnowledgeBase cleared")
