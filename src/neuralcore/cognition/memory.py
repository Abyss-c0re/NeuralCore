import re
import os
import asyncio
import time
import numpy as np
import hashlib
import json
from typing import Tuple, List, Dict, Any, AsyncIterable, Optional
from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder
from neuralcore.utils.config import get_loader
from neuralcore.clients.factory import get_clients
from neuralcore.utils.text_tokenizer import TextTokenizer
from neuralcore.utils.search import keyword_score, cosine_sim, cosine_sim_batch
from neuralcore.agents.state import AgentState
from neuralcore.cognition.items import KnowledgeItem, Topic
from neuralcore.cognition.consolidator import KnowledgeConsolidator
from neuralcore.cognition.knowledge import KnowledgeBase
from sklearn.feature_extraction.text import TfidfVectorizer
from fastembed import TextEmbedding

# ─────────────────────────────────────────────────────────────
# TOPIC & OFF-TOPIC DETECTION
# ─────────────────────────────────────────────────────────────
MSG_THR = 0.55  #
NUM_MSG = 8
OFF_THR = 0.65
OFF_FREQ = 4
SLICE_SIZE = 6

CHUNK_SIZE_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 128  # ~17% overlap
TOOL_OUTCOME_NO_CHUNK_THRESHOLD = 1500  #
logger = Logger.get_logger()


# ─────────────────────────────────────────────────────────────
# CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────
class ContextManager:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.max_tokens = agent.max_tokens
        clients = get_clients()
        self.client = clients.get("main")
        self.embeddings = clients.get("embeddings")
        loader = get_loader()
        embed_config = loader.get_client_config("embeddings") or {}
        self.fast_embedder = self._init_fastembed(embed_config)
        self.tokenizer = None
        try:
            self.tokenizer = TextTokenizer.get_instance()
        except ValueError:
            if self.client:
                loader = get_loader()
                cfg = loader.get_client_config("main")
                tokenizer_source = cfg.get("tokenizer")
                if not tokenizer_source:
                    raise ValueError("Tokenizer not found in main client config")
                self.tokenizer = TextTokenizer(tokenizer_source)
        if self.client and not getattr(self.client, "tokenizer", None):
            self.client.tokenizer = self.tokenizer
        if self.embeddings and not getattr(self.embeddings, "tokenizer", None):
            self.embeddings.tokenizer = self.tokenizer
        self.similarity_threshold = MSG_THR
        self.topics: List[Topic] = []
        self.current_topic = Topic("Initial topic")
        self.short_term_mem: Dict[str, KnowledgeItem] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        # Sparse index (TF-IDF) – now lazy + async
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.kb_keys_ordered: List[str] = []
        self._sparse_index_dirty = True
        self._last_rebuild_size = 0
        self.fs_state = {"cwd": ".", "known_folders": [], "negative_findings": []}
        self.action_log: List[Dict[str, Any]] = []
        self.files_checked: set[str] = set()
        self.context_stats = {
            "kb_added": 0,
            "messages_added": 0,
            "topics_switched": 0,
            "prunes": 0,
        }

        self.mode: str = "chat"
        self._mode_lock = asyncio.Lock()
        self.pure_chat_history: List[Dict[str, str]] = []
        self._last_analysis_ts = 0.0
        self.tools_executed: List[str] = []
        self.tool_call_history: List[Dict[str, Any]] = []
        self.knowledge_base = KnowledgeBase(self)
        self.consolidator = KnowledgeConsolidator(agent)
        self.recency_decay_lambda = self.consolidator.recency_decay_lambda

    def _init_fastembed(self, embed_config: dict):
        """Helper method to initialize FastEmbed with full path handling."""
        model_name = embed_config.get("fastembed_model", "BAAI/bge-small-en-v1.5")
        local_path = embed_config.get("fastembed_local_path")
        cache_dir = embed_config.get("fastembed_cache_dir")
        local_files_only = embed_config.get("fastembed_local_files_only", False)
        init_kwargs = {"model_name": model_name}
        # === Handle tilde expansion for local_path ===
        if local_path:
            expanded_local = os.path.expanduser(local_path)
            expanded_local = os.path.expandvars(expanded_local)
            expanded_local = os.path.abspath(expanded_local)  # make it absolute
            if os.path.isdir(expanded_local):
                init_kwargs["specific_model_path"] = expanded_local
                logger.info(f"✅ FastEmbed: Loading from local path → {expanded_local}")
            else:
                logger.warning(
                    f"⚠️ fastembed_local_path '{local_path}' (resolved to '{expanded_local}') does not exist. "
                    f"Falling back to Hugging Face cache/download."
                )
        # === Cache directory===
        if cache_dir:
            expanded_cache = os.path.abspath(
                os.path.expanduser(os.path.expandvars(cache_dir))
            )
            init_kwargs["cache_dir"] = expanded_cache
            logger.info(f" FastEmbed cache directory: {expanded_cache}")
        # === Offline mode ===
        if local_files_only:
            init_kwargs["local_files_only"] = True
            logger.info(" FastEmbed: local_files_only=True (strict offline mode)")
            # Initialize
        return TextEmbedding(**init_kwargs)

    async def set_mode(self, mode: str) -> None:
        async with self._mode_lock:
            if mode not in ("chat", "agentic", "investigation"):
                mode = "chat"
            old = self.mode
            self.mode = mode
            logger.info(f"ContextManager mode changed: {old} → {mode}")

    def reset(self, hard: bool = True) -> None:
        """Reset ContextManager to a clean state.
        Args:
            hard: If True (default), clears **everything** including embedding cache and short-term memory.
                  If False, preserves some caches (e.g. embeddings) so it survives across sub-tasks.
        """
        logger.info(f"ContextManager.reset() called — hard={hard}")
        self.mode = "chat"
        self.current_topic = Topic("Initial topic")
        self.topics.clear()
        self.short_term_mem.clear()
        if hard:
            self.embedding_cache.clear()
        self._sparse_index_dirty = True
        self.tfidf_matrix = None
        self.kb_keys_ordered.clear()
        self._last_rebuild_size = 0
        self.tfidf_vectorizer = None
        self.pure_chat_history.clear()
        for topic in self.topics:
            topic.history.clear()
            topic.archived_history.clear()
            topic.history_tokens.clear()
            topic.history_embeddings.clear()
        self.current_topic.history.clear()
        self.current_topic.archived_history.clear()
        self.current_topic.history_tokens.clear()
        self.current_topic.history_embeddings.clear()
        self.tools_executed.clear()
        self.tool_call_history.clear()
        self.action_log.clear()
        self.files_checked.clear()
        self.context_stats = {
            "kb_added": 0,
            "messages_added": 0,
            "topics_switched": 0,
            "prunes": 0,
        }
        self.fs_state = {"cwd": ".", "known_folders": [], "negative_findings": []}
        logger.info(
            f"ContextManager reset complete. "
            f"KB items: {len(self.short_term_mem)} | Mode: {self.mode}"
        )

    def clear_tool_history(self) -> None:
        """Light reset — only clears tool call history and executed list."""
        self.tools_executed.clear()
        self.tool_call_history.clear()
        logger.info("Tool history cleared (KB and conversation preserved)")

    class TaskContext:
        def __init__(self, name: str, parent: "ContextManager"):
            self.name = name
            self.parent = parent
            self.important_files: set[str] = set()
            self.key_results: List[Dict[str, Any]] = []
            self.findings: List[str] = []
            self.hypotheses: List[str] = []
            self.max_results = 15

        async def add_important_result(
            self,
            title: str,
            content: str,
            source: str = "tool",
            metadata: dict | None = None,
        ) -> None:
            if not content or not content.strip():
                return
            summary = content[:600] + ("..." if len(content) > 600 else "")
            entry = {
                "title": title,
                "summary": summary,
                "source": source,
                "ts": time.time(),
                "metadata": metadata or {},
            }
            self.key_results.append(entry)
            if len(self.key_results) > self.max_results:
                self.key_results.pop(0)
            await self.parent.add_external_content(
                source_type=f"task_result_{self.name}",
                content=f"[{title}] {summary}\nMetadata: {metadata or {}}",
                metadata={"task": self.name, **(metadata or {})},
            )
            self.parent._log_action(
                "task_result", f"Task {self.name} → {title}", metadata
            )

        def add_important_file(self, filepath: str):
            self.important_files.add(filepath)
            self.parent.files_checked.add(filepath)

        def get_context(self, max_tokens: int = 3500) -> str:
            lines = [PromptBuilder.task_context_header(self.name)]
            if self.important_files:
                lines.append(
                    PromptBuilder.task_context_important_files_section(
                        list(self.important_files)
                    )
                )
            if self.key_results:
                lines.append(
                    PromptBuilder.task_context_key_results_section(self.key_results)
                )
            if self.findings:
                lines.append(PromptBuilder.task_context_findings_section(self.findings))
            if self.hypotheses:
                lines.append(
                    PromptBuilder.task_context_hypotheses_section(self.hypotheses)
                )
            text = "\n".join(lines)
            return text[: max_tokens * 4]

    def create_task_context(self, task_name: str) -> "ContextManager.TaskContext":
        if not hasattr(self, "_task_contexts"):
            self._task_contexts: Dict[str, ContextManager.TaskContext] = {}
        if task_name not in self._task_contexts:
            self._task_contexts[task_name] = self.TaskContext(task_name, self)
            logger.info(f"Created new task context: {task_name}")
        return self._task_contexts[task_name]

    def get_task_context(self, task_name: str) -> "ContextManager.TaskContext | None":
        return getattr(self, "_task_contexts", {}).get(task_name)

    def list_active_tasks(self) -> List[str]:
        return list(getattr(self, "_task_contexts", {}).keys())

    async def fetch_embedding(
        self, text: str, size: int = 500, prefix: str | None = None
    ) -> np.ndarray:
        if not text or not text.strip():
            return np.array([])
        prefix_str = prefix or "default"
        cache_key = hashlib.md5(
            f"{prefix_str}:{text[:1000]}".encode("utf-8")
        ).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        emb = np.array([])
        if self.fast_embedder is not None:
            try:

                def _embed_sync():
                    # Proper prefix handling for BGE models
                    if prefix == "query":
                        return np.asarray(
                            list(self.fast_embedder.embed([text], prefix="query"))[0],
                            dtype=np.float32,
                        )
                    else:
                        return np.asarray(
                            list(self.fast_embedder.embed([text], prefix="passage"))[0],
                            dtype=np.float32,
                        )

                emb = await asyncio.to_thread(_embed_sync)
            except Exception as e:
                logger.warning(f"FastEmbed failed: {e}")
        elif self.embeddings is not None:
            try:
                emb = await self.embeddings.fetch_embedding(text, size)
            except Exception as e:
                logger.warning(f"Fallback embeddings failed: {e}")
        if emb.size > 0 and np.isfinite(emb).all():
            self.embedding_cache[cache_key] = emb
        return emb

    def _log_action(
        self, action_type: str, description: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        entry = {
            "type": action_type,
            "desc": description[:200],
            "ts": time.time(),
            "meta": metadata or {},
        }
        self.action_log.append(entry)
        if len(self.action_log) > 60:
            self.action_log.pop(0)
        if action_type.startswith("add_"):
            self.context_stats["messages_added"] += 1
        elif action_type == "switch_topic":
            self.context_stats["topics_switched"] += 1
        elif action_type == "prune":
            self.context_stats["prunes"] += 1

    def _extract_potential_file_or_param_mentions(self, query: str) -> List[str]:
        """Lightweight extraction of likely file paths, filenames, IDs from query."""
        if not query:
            return []

        patterns = [
            r'["\']([^"\']+\.(?:py|js|ts|json|txt|md|log|csv|yaml|yml|toml))["\']',  # common extensions
            r'["\']([/\\][^"\']+)["\']',  # paths
            r"\b(\S+\.\S{2,5})\b",  # filename-like
        ]
        mentions = []
        for pat in patterns:
            mentions.extend(re.findall(pat, query))

        words = re.findall(r"\b\w{3,}\b", query)
        mentions.extend([w for w in words if "." in w or "/" in w or "\\" in w])
        return list(dict.fromkeys(mentions))  # dedup preserve order

    # ─────────────────────────────────────────────────────────────
    # CONTEXT SUMMARY, PRUNING, INVESTIGATION STATE
    # ─────────────────────────────────────────────────────────────
    def get_context_summary(self, max_messages: int = 10, max_chars: int = 1000) -> str:
        """Improved context summary:
        - Max 40% of budget for recent messages
        - Remaining 60% split for last 4 tool outcomes (via _get_recent_tool_outcomes)
        """
        if self.mode == "chat":
            if not self.pure_chat_history:
                return ""
            recent = self.pure_chat_history[-4:]
            lines = [PromptBuilder.context_summary_chat_header()]
            lines.extend(
                f"[{msg['role'].upper()}] {msg['content'][:120]}..." for msg in recent
            )
            return "\n".join(lines)
        # ── Calculate budget split ─────────────────────────────────────
        msg_budget = int(max_chars * 0.40)  # 40% → messages
        tools_budget = max_chars - msg_budget  # 60% → tools + other sections

        files = sorted(list(self.files_checked))
        tools_executed = self.tools_executed
        recent_messages = self.current_topic.history[-max_messages:]
        message_summaries = [
            f"[{msg['role'].upper()}] {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}"
            for msg in recent_messages
        ]
        total_tokens = (
            sum(self.current_topic.history_tokens)
            if self.current_topic.history_tokens
            else 0
        )
        lines = [
            PromptBuilder.context_summary_header(),
            PromptBuilder.context_summary_files_section(files),
            PromptBuilder.context_summary_tools_section(tools_executed),
            PromptBuilder.context_summary_kb_section(len(self.short_term_mem)),
            PromptBuilder.context_summary_token_section(total_tokens, self.max_tokens),
            PromptBuilder.context_summary_topic_section(self.current_topic.name),
            "",
            PromptBuilder.context_summary_last_messages_header(),
        ]

        msg_section = "\n".join(message_summaries)
        if len(msg_section) > msg_budget:
            msg_section = (
                msg_section[:msg_budget].rsplit("\n", 1)[0] + "\n…(messages truncated)"
            )
        lines.extend([msg_section, ""])

        try:
            recent_tools = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.run(
                    self._get_recent_tool_outcomes(
                        limit=4,
                        max_tokens_per_result=800,
                        max_total_tokens=tools_budget,
                    )
                ),
            )
            recent_tools = (
                asyncio.get_event_loop().run_until_complete(recent_tools)
                if asyncio.get_event_loop().is_running()
                else ""
            )
            if recent_tools and recent_tools.strip():
                lines.append(PromptBuilder.context_summary_recent_tools_header())
                lines.append(recent_tools)
        except Exception as e:
            logger.warning(f"Failed to fetch recent tool outcomes for summary: {e}")
        full_summary = "\n".join(lines)
        if len(full_summary) > max_chars:
            cutoff = max_chars - 60
            summary_lines = full_summary[:cutoff].rsplit("\n", 1)[0]
            full_summary = summary_lines + "\n…(summary truncated)"
        logger.debug(
            f"get_context_summary generated {len(full_summary)} chars "
            f"(~40% messages, ~60% tools+metadata)"
        )
        return full_summary

    def prune_sub_agent_noise(self):
        if not hasattr(self, "current_topic") or not self.current_topic.history:
            return
        original_len = len(self.current_topic.history)
        kept = []
        for msg in self.current_topic.history:
            content_lower = msg.get("content", "").lower()
            role = msg.get("role", "")
            if (
                role in ("tool", "system")
                or "add_external_content" in content_lower
                or any(
                    k in content_lower
                    for k in [
                        "important",
                        "result",
                        "file",
                        "finding",
                        "kb",
                        "indexed",
                        "tool_outcome",
                    ]
                )
            ):
                kept.append(msg)
                continue
            if (
                role == "assistant"
                and len(self.current_topic.history)
                - self.current_topic.history.index(msg)
                <= 3
            ):
                kept.append(msg)
        self.current_topic.history = kept[-15:]
        pruned = original_len - len(self.current_topic.history)
        if pruned > 0:
            logger.info(f"[PRUNE] Sub-agent noise cleaned: {pruned} messages removed")
            self.context_stats["prunes"] += 1

    # ─────────────────────────────────────────────────────────────
    # CHUNKING
    # ─────────────────────────────────────────────────────────────
    def _chunk_text(
        self,
        text: str,
        parent_key: str,
        source_type: str,
        merge: bool = False,
    ) -> List[KnowledgeItem]:
        """Chunk text.
        If merge=True → always return exactly ONE KnowledgeItem with full content.
        If merge=False → return multiple chunk items (legacy behavior).
        """
        if len(text) < TOOL_OUTCOME_NO_CHUNK_THRESHOLD:
            chunk_texts = [text]
        else:
            if self.tokenizer:
                chunk_texts = self.tokenizer.split_text_into_chunks(
                    text, max_tokens=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS
                )
            else:
                chunk_size = CHUNK_SIZE_TOKENS * 4
                overlap = CHUNK_OVERLAP_TOKENS * 4
                chunks = []
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunks.append(text[start:end])
                    start = end - overlap
                chunk_texts = chunks
        if not chunk_texts:
            return []
        if merge:
            full_content = "\n\n".join(chunk_texts)
            content_hash = hashlib.md5(full_content.encode("utf-8")).hexdigest()[:12]
            item_key = f"{parent_key}_full_{content_hash}"
            metadata = {
                "parent_key": parent_key,
                "total_chunks": len(chunk_texts),
                "source_type": source_type,
                "is_merged": True,
                "chunked": len(chunk_texts) > 1,
            }
            return [KnowledgeItem(item_key, source_type, full_content, metadata)]
        else:
            items = []
            for i, chunk_content in enumerate(chunk_texts):
                if not chunk_content.strip():
                    continue
                content_hash = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()[
                    :12
                ]
                chunk_key = f"{parent_key}_chunk_{content_hash}"
                metadata = {
                    "parent_key": parent_key,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "source_type": source_type,
                }
                items.append(
                    KnowledgeItem(chunk_key, source_type, chunk_content, metadata)
                )
            return items

    async def _add_streaming_chunk_batch(
        self,
        tool_name: str,
        accumulated_text: str,
        parent_key: str,
        chunk_index: int,
        metadata: dict,
        timestamp: float,
        silent: bool = True,
    ) -> None:
        """Add a batch of accumulated text as one proper chunk."""
        if not accumulated_text.strip():
            return
        chunk_content = (
            f"Tool: {tool_name} (streaming batch {chunk_index})\n"
            f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n\n"
            f"{accumulated_text}"
        )
        chunk_meta = {
            "tool": tool_name,
            "timestamp": timestamp,
            "batch_index": chunk_index,
            "parent_key": parent_key,
            "streamed": True,
            "is_large_item": True,
            **metadata,
        }
        await self.add_external_content(
            source_type="tool_outcome_chunk",
            content=chunk_content,
            metadata=chunk_meta,
            silent=silent,
        )

    # ─────────────────────────────────────────────────────────────
    # ADD EXTERNAL CONTENT (lazy, does NOT rebuild immediately)
    # ─────────────────────────────────────────────────────────────
    async def add_external_content(
        self,
        source_type: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        silent: bool = False,
    ) -> str | None:
        """Add content to knowledge base.
        - Streaming chunks (tool_outcome_chunk) are allowed MULTIPLE times under the same parent_key.
        - Normal items keep strict deduplication.
        """
        if not content or len(content.strip()) < 15:
            return None
        metadata = metadata or {}
        content_lower = content.lower()
        # Contamination guard
        forbidden = PromptBuilder.contamination_forbidden_phrases()
        is_contaminated = any(phrase.lower() in content_lower for phrase in forbidden)
        if is_contaminated:
            if source_type in ("tool_outcome", "task_result"):
                for phrase in forbidden:
                    content = re.sub(
                        re.escape(phrase) + r".*?(\n|$)",
                        "",
                        content,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
            else:
                logger.warning(
                    f"[KB GUARD] Blocked contaminated content (source={source_type})"
                )
                return None
        # Parent key
        parent_key = metadata.get("parent_key") or (
            metadata.get("path")
            or metadata.get("filename")
            or f"{source_type}_{hash(content[:300])}"
        )
        parent_key = str(parent_key)
        is_streaming_chunk = source_type == "tool_outcome_chunk"
        # Strict deduplication ONLY for normal (non-streaming) items
        if not is_streaming_chunk and parent_key in self.files_checked:
            logger.debug(f"→ Skipped duplicate under parent_key: {parent_key}")
            return None
        # Process chunks / content
        chunk_items = self._chunk_text(content, parent_key, source_type)
        added = 0
        for item in chunk_items:
            if item.key in self.short_term_mem:
                continue
            item.metadata["is_large_item"] = metadata.get("is_large_item", False)
            item.embedding = await self.fetch_embedding(item.content, prefix="passage")
            self.short_term_mem[item.key] = item
            added += 1
        if added > 0:
            self._sparse_index_dirty = True
            self.context_stats["kb_added"] += added
            if not silent:
                logger.info(
                    f"✅ Added {added} clean chunks for parent_key={parent_key} | source={source_type}"
                )
            else:
                logger.debug(
                    f"[STREAM] Added {added} chunks under parent_key={parent_key} | source={source_type}"
                )
        else:
            logger.debug(f"→ Skipped duplicate content for parent_key={parent_key}")
        # ONLY mark as seen for NON-streaming items
        if not is_streaming_chunk:
            self.files_checked.add(parent_key)
        if not silent:
            self._log_action(
                "add_knowledge",
                f"Added {source_type} → {parent_key} ({added} chunks)",
                metadata,
            )
        return parent_key

    # ─────────────────────────────────────────────────────────────
    # SPARSE INDEX (TF-IDF) – fully async + lazy
    # ─────────────────────────────────────────────────────────────
    def _rebuild_sparse_index_sync(self):
        """Robust TF-IDF rebuild – fixes 'no terms remain' crash."""
        if len(self.short_term_mem) < 1:
            self.tfidf_matrix = None
            self.kb_keys_ordered = []
            return
        texts = [item.content for item in self.short_term_mem.values()]
        self.kb_keys_ordered = list(self.short_term_mem.keys())
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                min_df=1,
                max_df=0.98,
                ngram_range=(1, 2),
                max_features=10000,
            )
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.debug(f"Rebuilt TF-IDF: {getattr(self.tfidf_matrix, 'shape', None)}")
        except Exception as e:
            logger.warning(f"TF-IDF failed ({e}), using fallback")
            self.tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    async def _rebuild_sparse_index(self):
        """Async wrapper for CPU-heavy TF-IDF rebuild."""
        await asyncio.to_thread(self._rebuild_sparse_index_sync)
        self._last_rebuild_size = len(self.short_term_mem)

    async def _ensure_sparse_index(self):
        """Lazy + async rebuild."""
        if (
            not self._sparse_index_dirty
            and len(self.short_term_mem) == self._last_rebuild_size
        ):
            return
        await self._rebuild_sparse_index()
        self._sparse_index_dirty = False

    def _get_consolidator(self):
        """Lazy access to consolidator (safe even if not fully initialized yet)."""
        if self.consolidator is None:
            logger.debug("Consolidator not yet initialized")
            return None
        return self.consolidator

    # ─────────────────────────────────────────────────────────────
    # SPARSE RETRIEVAL
    # ─────────────────────────────────────────────────────────────
    async def _sparse_retrieve(self, query: str) -> Dict[str, float]:
        """Safe sparse retrieval with aggressive type handling for Pyright."""
        await self._ensure_sparse_index()
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            logger.debug("TF-IDF index not ready")
            return {}

        def _sparse_sync():
            try:
                if self.tfidf_vectorizer is None:
                    return {}
                if (
                    not hasattr(self.tfidf_vectorizer, "vocabulary_")
                    or len(self.tfidf_vectorizer.vocabulary_) == 0
                ):
                    if len(self.short_term_mem) > 0:
                        texts = [item.content for item in self.short_term_mem.values()]
                        logger.debug(f"Fitting TF-IDF on {len(texts)} documents")
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                    else:
                        return {}
                query_vec = self.tfidf_vectorizer.transform([query])
                matrix = self.tfidf_matrix
                vec = query_vec
                if hasattr(vec, "T"):
                    scores_matrix = matrix.dot(vec.T)  # type: ignore[attr-defined]
                else:
                    scores_matrix = matrix.dot(vec.transpose())  # type: ignore[attr-defined]
                scores = scores_matrix.toarray().flatten()  # type: ignore[attr-defined]
                result: Dict[str, float] = {}
                for idx, score in enumerate(scores):
                    if score > 1e-8 and idx < len(self.kb_keys_ordered):
                        result[self.kb_keys_ordered[idx]] = float(score)
                return result
            except Exception as e:
                logger.warning(f"Sparse retrieval failed: {e}", exc_info=False)
                return {}

        return await asyncio.to_thread(_sparse_sync)

    # ─────────────────────────────────────────────────────────────
    # KNOWLEDGE RETRIEVAL (now fully async)
    # ─────────────────────────────────────────────────────────────
    async def _retrieve_relevant_knowledge(self, query: str, max_kb_tokens: int) -> str:
        """Hybrid RAG with parameter/file-aware boosting for exact tool matches.
        All prompt text moved to PromptBuilder."""
        if not self.short_term_mem or not query.strip():
            logger.debug(
                "retrieve_relevant_knowledge: KB empty or query empty → returning ''"
            )
            return ""
        logger.debug(
            f"[RETRIEVE] START (legacy path) | query='{query[:100]}...' | "
            f"KB_size={len(self.short_term_mem)} | max_tokens={max_kb_tokens}"
        )
        # Use centralized prompt for embedding query
        embedding_query = PromptBuilder.knowledge_retrieval_embedding_query(query)
        query_emb = await self.fetch_embedding(embedding_query, prefix="query")
        sparse_scores = await self._sparse_retrieve(query)
        # Dense ranking
        dense_ranked = []
        if query_emb.size > 0:
            for key, item in self.short_term_mem.items():
                if (
                    getattr(item, "embedding", None) is not None
                    and item.embedding.size > 0
                ):
                    sim = cosine_sim(query_emb, item.embedding)
                    dense_ranked.append((sim, key))
            dense_ranked.sort(reverse=True)
            dense_ranked = dense_ranked[:50]
        # Sparse ranking
        sparse_ranked = sorted(
            [(score, key) for key, score in sparse_scores.items() if score > 0],
            reverse=True,
        )[:50]
        logger.debug(
            f"[RETRIEVE] Dense candidates: {len(dense_ranked)} | Sparse candidates: {len(sparse_ranked)}"
        )
        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        k = 60
        for rank, (_, key) in enumerate(dense_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        for rank, (_, key) in enumerate(sparse_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        # ── PARAMETER/FILE BOOST ──
        file_mentions = self._extract_potential_file_or_param_mentions(query)
        if file_mentions:
            logger.debug(f"[BOOST] Detected file/param mentions: {file_mentions}")
            boosted = 0
            for key, item in self.short_term_mem.items():
                if item.source_type != "tool_outcome":
                    continue
                boost = 0.0
                meta_str = json.dumps(item.metadata, ensure_ascii=False).lower()
                args_str = json.dumps(
                    item.metadata.get("args", {}), ensure_ascii=False
                ).lower()
                content_lower = item.content.lower()
                for mention in file_mentions:
                    m_lower = mention.lower()
                    if (
                        m_lower in meta_str
                        or m_lower in args_str
                        or m_lower in content_lower
                    ):
                        boost += 5.0
                    elif any(
                        word in content_lower
                        for word in m_lower.split()
                        if len(word) > 2
                    ):
                        boost += 2.0
                if boost > 0:
                    current = rrf_scores.get(key, 0.0)
                    rrf_scores[key] = current + boost
                    boosted += 1
                    logger.debug(f"[BOOST] +{boost:.1f} to {key} (tool_outcome)")
            if boosted > 0:
                logger.debug(f"[BOOST] Applied boosts to {boosted} tool outcomes")
        # Final scored list
        scored = sorted(
            [
                (rrf_scores.get(key, 0.0), key, self.short_term_mem[key])
                for key in rrf_scores
            ],
            reverse=True,
        )
        logger.debug(f"[RETRIEVE] Final scored items before formatting: {len(scored)}")
        parts: List[str] = []
        used = 0
        tool_outcome_count = 0
        blocked_count = 0
        forbidden_phrases = PromptBuilder.contamination_forbidden_phrases()
        for score, key, item in scored[:40]:
            if any(forbidden in item.content for forbidden in forbidden_phrases):
                blocked_count += 1
                continue
            if item.source_type == "tool_outcome":
                tool_outcome_count += 1
            meta_str = ", ".join(f"{k}={v}" for k, v in item.metadata.items() if v)
            header = PromptBuilder.knowledge_block_header(
                source_type=item.source_type,
                key=item.key,
                meta_str=meta_str,
            )
            block = (
                f"{header}{item.content}\n{PromptBuilder.knowledge_block_separator()}"
            )
            block_tokens = (
                self.tokenizer.count_tokens(block)
                if self.tokenizer
                else len(block) // 4
            )
            if used + block_tokens > max_kb_tokens:
                logger.debug(
                    f"[TOKEN] Budget reached ({used}/{max_kb_tokens}) — stopping at item {key}"
                )
                break
            parts.append(block)
            used += block_tokens
        result = "".join(parts)
        logger.debug(
            PromptBuilder.retrieve_relevant_knowledge_summary(
                returned_chars=len(result),
                tool_outcome_count=tool_outcome_count,
                blocked_count=blocked_count,
                total_scored=len(scored),
            )
        )
        logger.debug(
            f"[RETRIEVE] END — returned {len(result)} chars, {tool_outcome_count} tool outcomes"
        )
        return result

    async def _get_broad_candidates(self, query: str) -> List[KnowledgeItem]:
        """Returns broad candidates using RRF fusion + file/param boosting + recency decay.
        Now uses vectorized dense similarity for speed.
        """
        if not self.short_term_mem:
            logger.debug("[BROAD] KB empty → returning []")
            return []

        logger.debug(
            f"[BROAD] START | query='{query[:100]}...' | KB_size={len(self.short_term_mem)}"
        )

        # === DENSE RANKING (vectorized) ===
        query_emb = await self.fetch_embedding(
            PromptBuilder.knowledge_retrieval_embedding_query(query), prefix="query"
        )

        dense_ranked = []
        if query_emb.size > 0:
            # Prepare all embeddings in order
            items = list(self.short_term_mem.items())
            embeddings = [item.embedding for _, item in items]

            # Vectorized similarity
            sims = cosine_sim_batch(query_emb, embeddings)

            for (key, item), sim in zip(items, sims):
                if sim <= 0:
                    continue

                # Dynamic recency decay
                ts = item.metadata.get("timestamp", time.time())
                recency_hours = max(0.0, (time.time() - ts) / 3600.0)
                decay = np.exp(-self.recency_decay_lambda * recency_hours)
                decayed_sim = sim * decay

                dense_ranked.append((decayed_sim, key, item))

        logger.debug(f"[BROAD] Dense ranked: {len(dense_ranked)} items")

        # === SPARSE RANKING (unchanged) ===
        sparse_scores = await self._sparse_retrieve(query)
        sparse_ranked = []
        for key, score in sparse_scores.items():
            if score > 0 and key in self.short_term_mem:
                item = self.short_term_mem[key]
                ts = item.metadata.get("timestamp", time.time())
                recency_hours = max(0.0, (time.time() - ts) / 3600.0)
                decay = np.exp(-self.recency_decay_lambda * recency_hours)
                decayed_score = score * decay
                sparse_ranked.append((decayed_score, key, item))

        logger.debug(f"[BROAD] Sparse ranked: {len(sparse_ranked)} items")

        # === RRF FUSION ===
        rrf_scores: Dict[str, float] = {}
        k = 60
        for rank, (_, key, _) in enumerate(dense_ranked[:50], start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        for rank, (_, key, _) in enumerate(sparse_ranked[:50], start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)

        # === FILE/PARAM BOOSTING (unchanged) ===
        file_mentions = self._extract_potential_file_or_param_mentions(query)
        if file_mentions:
            logger.debug(f"[BOOST] Detected mentions: {file_mentions}")
            boosted = 0
            for key, item in self.short_term_mem.items():
                if item.source_type != "tool_outcome":
                    continue
                boost = 0.0
                meta_str = json.dumps(item.metadata, ensure_ascii=False).lower()
                args_str = json.dumps(
                    item.metadata.get("args", {}), ensure_ascii=False
                ).lower()
                content_lower = item.content.lower()
                for mention in file_mentions:
                    m_lower = mention.lower()
                    if (
                        m_lower in meta_str
                        or m_lower in args_str
                        or m_lower in content_lower
                    ):
                        boost += 5.0
                    elif any(
                        word in content_lower
                        for word in m_lower.split()
                        if len(word) > 2
                    ):
                        boost += 2.0
                if boost > 0:
                    current = rrf_scores.get(key, 0.0)
                    rrf_scores[key] = current + boost
                    boosted += 1
            if boosted > 0:
                logger.debug(f"[BOOST] Applied boosts to {boosted} tool outcomes")

        # === FINAL CANDIDATES ===
        scored = [
            (rrf_scores.get(key, 0.0), item)
            for key, item in self.short_term_mem.items()
            if key in rrf_scores
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [item for score, item in scored[:100]]

        logger.debug(
            f"[BROAD] END → returning {len(candidates)} broad candidates "
            f"(top RRF score: {scored[0][0] if scored else 0:.4f})"
        )
        return candidates

    async def ranked_retrieve(
        self,
        query: str,
        max_kb_tokens: int = 4500,
        research_mode: bool = False,
    ) -> str:
        """Proper chunk-aware RAG retrieval with recency decay.
        - Pulls many small chunks from KnowledgeBase
        - Reranks with consolidator (if present)
        - Applies dynamic decay to favor recent memories
        - Strictly respects token budget
        """
        if not query.strip():
            return ""
        logger.debug(
            f"[RANKED_RETRIEVE] START | query='{query[:80]}...' | "
            f"research={research_mode} | budget={max_kb_tokens}"
        )
        # 1. Get candidates (decay is already applied inside _get_broad_candidates)
        short_term = await self._get_broad_candidates(query)
        long_term = []
        if getattr(self.knowledge_base, "enabled", False):
            try:
                long_term = await self.knowledge_base.retrieve(query, k=100)
                logger.debug(f"Persistent KB returned {len(long_term)} chunk items")
            except Exception as e:
                logger.warning(f"KB retrieve failed: {e}")

        # 2. Deduplicate
        all_candidates = short_term + long_term
        seen = set()
        unique = []
        for item in all_candidates:
            key = getattr(item, "key", id(item))
            if key not in seen:
                seen.add(key)
                unique.append(item)

        if research_mode:
            unique = [
                item
                for item in unique
                if getattr(item, "source_type", "") in ("tool_outcome", "file")
            ]
            if not unique:
                logger.debug("[RANKED_RETRIEVE] No research-mode candidates")
                return ""
        if not unique:
            logger.debug("[RANKED_RETRIEVE] No candidates")
            return ""

        # 3. Rerank (decay already baked into candidate scores)
        rerank_k = min(len(unique), 75)
        if self.consolidator:
            reranked = await self.consolidator.rerank(query, unique, k=rerank_k)
        else:
            reranked = unique[:rerank_k]

        # 4. Greedy selection within strict token budget
        parts = []
        remaining = max_kb_tokens
        used_tokens = 0
        for item in reranked:
            content = getattr(item, "content", "").strip()
            if not content:
                continue
            item_tokens = self.count_tokens([{"role": "user", "content": content}])
            if item_tokens <= remaining:
                parts.append(content)
                remaining -= item_tokens
                used_tokens += item_tokens
            else:
                if remaining > 300:
                    ratio = remaining / item_tokens
                    truncated = content[: int(len(content) * ratio * 0.92)]
                    parts.append(truncated + " …")
                    used_tokens += self.count_tokens(
                        [{"role": "user", "content": truncated}]
                    )
                break

        result = "\n\n---\n\n".join(parts).strip()
        logger.debug(
            f"[RANKED_RETRIEVE] END → {len(result):,} chars / ~{used_tokens} tokens "
            f"from {len(parts)} chunks (budget={max_kb_tokens})"
        )
        return result

    def _is_duplicate_tool_call(
        self, tool_name: str, clean_args: Dict[str, Any]
    ) -> bool:
        """Check if the exact same tool + arguments was called recently.
        Uses last 8 entries for deduplication window."""
        if not self.tool_call_history or len(self.tool_call_history) < 2:
            return False
        # Normalize current args for comparisons
        try:
            current_key = json.dumps(clean_args, sort_keys=True, ensure_ascii=False)
        except Exception:
            current_key = str(clean_args)
        # Check last N calls (excluding the one we just added)
        for entry in reversed(
            self.tool_call_history[:-1][-8:]
        ):  # last 8 before this one
            if entry["tool"] != tool_name:
                continue
            try:
                entry_key = json.dumps(
                    entry["args"], sort_keys=True, ensure_ascii=False
                )
            except Exception:
                entry_key = str(entry["args"])
            if entry_key == current_key:
                return True
        return False

    async def add_message(
        self, role: str, message: str, embedding: np.ndarray | None = None
    ) -> None:
        if not message or not message.strip():
            logger.debug(f"Skipping empty {role} message")
            return
        if embedding is None or embedding.size == 0:
            embedding = await self.fetch_embedding(message, prefix="passage")
        token_count = (
            self.tokenizer.count_tokens(message)
            if self.tokenizer
            else len(message) // 4
        )
        if self.mode == "chat":
            self.pure_chat_history.append({"role": role, "content": message})
            if len(self.pure_chat_history) > 30:
                self.pure_chat_history.pop(0)
            self._log_action(
                "add_message", f"{role} (chat) ({len(message)} chars)", {"role": role}
            )
            return
        topic = await self._match_topic(embedding, exclude_topic=self.current_topic)
        if topic:
            await self.switch_topic(topic)
        await self.current_topic.add_message(role, message, embedding, token_count)
        if time.time() - self._last_analysis_ts > 2.0:
            asyncio.create_task(self._analyze_history())
            self._last_analysis_ts = time.time()
        self._log_action(
            "add_message", f"{role} message ({len(message)} chars)", {"role": role}
        )

    # ─────────────────────────────────────────────────────────────
    # SWITCH / MATCH TOPIC
    # ─────────────────────────────────────────────────────────────
    async def switch_topic(self, topic: Topic) -> None:
        async with asyncio.Lock():
            if topic.name != self.current_topic.name:
                if not any(t.name == self.current_topic.name for t in self.topics):
                    self.topics.append(self.current_topic)
                logger.info(f"Switched topic → {topic.name}")
                self.current_topic = topic
                self._log_action("switch_topic", f"Switched to {topic.name}")

    async def _match_topic(
        self, embedding: np.ndarray, exclude_topic: Topic | None = None
    ) -> Topic | None:
        if len(self.topics) == 0:
            return None

        async def compute(topic: Topic) -> Tuple[float, Topic]:
            if len(topic.embedded_description) == 0 or len(embedding) == 0:
                return 0.0, topic
            return cosine_sim(embedding, topic.embedded_description), topic

        results = await asyncio.gather(
            *[compute(t) for t in self.topics if t is not exclude_topic]
        )
        best_sim, best_topic = 0.0, None
        for sim, t in results:
            if sim > best_sim and sim >= self.similarity_threshold:
                best_sim, best_topic = sim, t
        return best_topic

    # ─────────────────────────────────────────────────────────────
    # HISTORY ANALYSIS
    # ─────────────────────────────────────────────────────────────
    async def _score_messages_by_relevance(
        self, messages: list, ref_emb: np.ndarray
    ) -> list:
        scored = []
        for msg in messages:
            emb = await self.fetch_embedding(msg["content"], prefix="passage")
            score = cosine_sim(emb, ref_emb) if len(emb) > 0 else 0.0
            scored.append((score, msg, emb))
        return scored

    async def _analyze_history(self) -> None:
        if self.mode == "chat":
            return
        topic = self.current_topic
        if len(topic.history) <= 4:
            return
        if topic.embedded_description.size == 0 and topic.description.strip():
            topic.embedded_description = await self.fetch_embedding(
                topic.description, prefix="passage"
            )
        window_size = min(len(topic.history), SLICE_SIZE * 3)
        window_msgs = topic.history[-window_size:]
        scored_msgs = await self._score_messages_by_relevance(
            window_msgs, topic.embedded_description
        )
        off_topic_indices = [
            i for i, (s, _, _) in enumerate(scored_msgs) if s < OFF_THR
        ]
        if len(off_topic_indices) <= len(scored_msgs) / 2:
            return
        off_start_idx = len(topic.history) - window_size + off_topic_indices[0]
        segment = topic.history[off_start_idx:]
        segment_tokens = [
            self.tokenizer.count_tokens(m["content"])
            if self.tokenizer
            else len(m["content"]) // 4
            for m in segment
        ]
        name, desc = await self.generate_topic_info_from_history(segment)
        if not name or not desc:
            return
        emb = await self.fetch_embedding(desc)
        matched = await self._match_topic(emb, exclude_topic=topic)
        target_topic = matched or Topic(name, desc)
        if not matched:
            target_topic.embedded_description = emb
        segment_embeddings = await asyncio.gather(
            *(self.fetch_embedding(m["content"]) for m in segment)
        )
        for m, emb_m, tks in zip(segment, segment_embeddings, segment_tokens):
            await target_topic.add_message(
                m["role"], m["content"], emb_m, token_count=tks
            )
        await self.switch_topic(target_topic)
        topic.history = topic.history[:off_start_idx]
        topic.archived_history.extend(segment)
        logger.info(
            f"Off-topic segment moved → {target_topic.name} | {len(segment)} messages"
        )
        await self.current_topic.add_message(
            role="system",
            content=f"⚠️ {len(off_topic_indices)} off-topic messages detected.",
            embedding=np.array([]),
            token_count=0,
        )

    async def generate_topic_info_from_history(
        self, history: list, max_retries: int = 3
    ) -> Tuple[str, str] | Tuple[None, None]:
        if not self.client:
            raise RuntimeError("LLM client not loaded...")
        attempt = 0
        while attempt < max_retries:
            await asyncio.sleep(0.3)
            try:
                response = await self.client.ask(PromptBuilder.topics_helper(history))
                if not isinstance(response, str):
                    attempt += 1
                    continue
                clean = re.sub(
                    r"^```(?:json)?|```$", "", response, flags=re.IGNORECASE
                ).strip()
                # JSON-first parsing
                try:
                    data = json.loads(clean)
                    name = data.get("name") or data.get("topic") or "unknown"
                    desc = data.get("description") or data.get("desc") or ""
                    if name and desc:
                        return name, desc
                except json.JSONDecodeError:
                    pass
                # fallback regex
                matches = re.findall(r':\s*"([^"]+)"', clean)
                name = matches[0] if matches else "unknown"
                desc = matches[1] if len(matches) > 1 else ""
                if name and desc:
                    return name, desc
            except Exception as e:
                logger.error(
                    f"Topic extraction failed (attempt {attempt + 1} Reason: {e})",
                    exc_info=True,
                )
                attempt += 1
        return None, None

    # ─────────────────────────────────────────────────────────────
    # PROVIDE CONTEXT, PRUNE, TOKEN COUNT, ARCHIVED CONTEXT, RECORD TOOL OUTCOME
    # ─────────────────────────────────────────────────────────────
    async def provide_context(
        self,
        query: str = "",
        max_input_tokens: int = 10000,
        reserved_for_output: int = 24000,
        include_logs: bool = False,
        min_history_tokens: int = 2000,
        max_kb_tokens: int = 18000,
        chat: bool = False,
        lightweight_agentic: bool = False,
        research_mode: bool = False,
        return_as_string: bool = False,
    ) -> List[Dict[str, Any]] | str:
        messages: List[Dict[str, Any]] = []
        state = self.agent.state
        logger.debug(
            f"provide_context called | query='{query[:100]}...' | chat={chat} | "
            f"lightweight_agentic={lightweight_agentic} | research_mode={research_mode} | "
            f"has_state={state is not None} | return_as_string={return_as_string} | include_logs={include_logs}"
        )
        if not len(query) > 0:
            return "No query was provided"

        # ── UNIVERSAL LOGS HANDLING (works in EVERY mode) ──
        logs_section = ""
        if include_logs:
            try:
                log_lines = Logger.get_log_data(level="info", max_entries=100)
                if log_lines:
                    logs_section = PromptBuilder.recent_logs_section(log_lines)
            except Exception:
                pass

        if chat:
            # ── RICH CHAT MODE  ──
            logger.debug("→ Entering CHAT mode")
            if self.pure_chat_history:
                recent = self.pure_chat_history[-12:]
                for msg in recent:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            total_tokens = self.count_tokens(messages)
            if total_tokens > max_input_tokens - reserved_for_output:
                self.prune_to_fit_context(
                    messages,
                    max_tokens=max_input_tokens - reserved_for_output,
                    min_keep_messages=4,
                    system_role="system",
                    user_role="user",
                    assistant_role="assistant",
                    tool_role="tool",
                )
        else:
            # ── AGENTIC MODE ──
            logger.debug("→ Entering AGENTIC mode")
            tokens_used = 0
            if lightweight_agentic and state is not None:
                # ── LIGHTWEIGHT PATH FOR LONG-RUNNING LOOPS ──
                logger.debug(
                    "→ Using IMPROVED LIGHTWEIGHT agentic context (long-running mode)"
                )
                warnings = state.validate_state_integrity()
                if warnings:
                    logger.warning(
                        f"AgentState integrity warnings before lightweight context: {warnings}"
                    )
                full_context = PromptBuilder.lightweight_agentic_context(state)
                messages.append({"role": "system", "content": full_context})
                tokens_used = self.count_tokens(messages)

                # === INJECT LAST FULL TOOL RESULT (via helper) ===
                latest_tool = self._get_latest_tool_outcome()
                if latest_tool:
                    tool_block = (
                        f"\n\n=== LAST TOOL RESULT (FULL) ===\n"
                        f"Tool: {latest_tool['tool_name']}\n"
                        f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_tool['timestamp']))}\n\n"
                        f"{latest_tool['content']}\n"
                        f"=== END LAST TOOL RESULT ==="
                    )
                    tool_tokens = self.count_tokens(
                        [{"role": "user", "content": tool_block}]
                    )
                    if tokens_used + tool_tokens < (
                        max_input_tokens - reserved_for_output - 800
                    ):
                        messages.append({"role": "system", "content": tool_block})
                        tokens_used += tool_tokens
                        logger.debug(
                            f"→ [LIGHTWEIGHT] Injected FULL tool outcome "
                            f"({latest_tool['tool_name']}, {len(latest_tool['content'])} chars)"
                        )

            else:
                # ── RICH AGENTIC MODE (includes research_mode) ──
                summary = self.get_context_summary()
                full_system = (
                    f"\n\n{summary}\n\n{PromptBuilder.context_summary_instruction()}"
                )
                messages.append({"role": "system", "content": full_system})
                tokens_used = self.count_tokens(messages)

                # === INJECT LAST FULL TOOL RESULT (via helper) ===
                latest_tool = self._get_latest_tool_outcome()
                if latest_tool:
                    tool_block = (
                        f"\n\n=== LAST TOOL RESULT (FULL) ===\n"
                        f"Tool: {latest_tool['tool_name']}\n"
                        f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_tool['timestamp']))}\n\n"
                        f"{latest_tool['content']}\n"
                        f"=== END LAST TOOL RESULT ==="
                    )
                    tool_tokens = self.count_tokens(
                        [{"role": "user", "content": tool_block}]
                    )
                    if tokens_used + tool_tokens < (
                        max_input_tokens - reserved_for_output - 800
                    ):
                        messages.append({"role": "system", "content": tool_block})
                        tokens_used += tool_tokens
                        logger.debug(
                            f"→ [RICH] Injected FULL tool outcome "
                            f"({latest_tool['tool_name']}, {len(latest_tool['content'])} chars)"
                        )

            # ── COMMON FINAL STEPS (KB + HISTORY) ──
            query_tokens = (
                self.count_tokens([{"role": "user", "content": query}])
                if query.strip()
                else 0
            )
            target = max_input_tokens - reserved_for_output
            remaining = max(0, target - tokens_used - query_tokens)
            if research_mode:
                kb_tokens = min(18000, max(4000, remaining // 2))
            else:
                kb_tokens = (
                    min(max_kb_tokens, remaining // 3)
                    if not lightweight_agentic
                    else 1800
                )

            context_parts: List[str] = []

            if not lightweight_agentic and query.strip():
                if research_mode and state is not None and state.executed_functions:
                    executed_tools_summary = []
                    for func in state.executed_functions[-8:]:
                        name = func.get("name") or "tool"
                        args = func.get("args", {})
                        if args:
                            args_str = ", ".join(
                                f"{k}={v}" for k, v in list(args.items())[:5]
                            )
                            executed_tools_summary.append(
                                f"Name: {name} Args: {args_str}"
                            )
                        else:
                            executed_tools_summary.append(name)
                    tools_str = " | ".join(executed_tools_summary)
                    enriched_query = f"{query} [EXECUTED TOOLS: {tools_str}]"
                else:
                    enriched_query = query

                kb_text = await self.ranked_retrieve(
                    enriched_query,
                    kb_tokens,
                    research_mode=research_mode,
                )
                if kb_text:
                    context_parts.append(
                        PromptBuilder.relevant_external_context_section(kb_text)
                    )

            if lightweight_agentic:
                logger.debug("→ LIGHTWEIGHT MODE: skipping full conversation history")
            elif research_mode:
                logger.debug(
                    "→ RESEARCH MODE: skipping conversation history, only relevant KB context"
                )
            else:
                history_budget = max(remaining - kb_tokens, min_history_tokens)
                recent_msgs: List[Dict[str, str]] = []
                for msg, t in zip(
                    reversed(self.current_topic.history),
                    reversed(self.current_topic.history_tokens),
                ):
                    if t > history_budget:
                        break
                    recent_msgs.insert(0, msg)
                    history_budget -= t
                    tokens_used += t
                messages.extend(recent_msgs)

            provided_context = (
                "\n\n---\n\n".join(context_parts)
                if context_parts
                else "[No additional external context]"
            )
            user_query_text = query.strip() or "[AUTONOMOUS CONTINUATION]"
            user_content = f"""{user_query_text}
            {provided_context}"""
            messages.append({"role": "user", "content": user_content})

        # ═══════════════════════════════════════════════════════════════
        # ── FINAL NORMALIZATION: SYSTEM MESSAGE ALWAYS FIRST + LOGS COMBINED ──
        # ═══════════════════════════════════════════════════════════════
        system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"), None
        )

        if system_idx is None:
            if chat:
                default_system = PromptBuilder.casual_chat_system_prompt()
            else:
                default_system = getattr(
                    self.client,
                    "system_message",
                    PromptBuilder.default_agent_system_prompt(),
                )
            messages.insert(0, {"role": "system", "content": default_system})
            system_idx = 0
            logger.debug(
                f"→ Injected {'chat-friendly' if chat else 'default'} system message "
                f"via PromptBuilder (include_logs={include_logs})"
            )
        else:
            if system_idx != 0:
                sys_msg = messages.pop(system_idx)
                messages.insert(0, sys_msg)
                logger.debug(
                    f"→ Moved system message from index {system_idx} to position 0"
                )

        if logs_section and logs_section not in messages[0].get("content", ""):
            current_content = messages[0]["content"].rstrip()
            messages[0]["content"] = f"{current_content}\n\n{logs_section}"
            logger.debug("→ Combined recent logs into the first system message")

        # ── FINAL PRUNE (if needed) ──
        final_tokens = self.count_tokens(messages)
        target = max_input_tokens - reserved_for_output
        if final_tokens > target:
            removed, pruned_turns = self.prune_to_fit_context(
                messages,
                max_tokens=target,
                min_keep_messages=2 if lightweight_agentic else 4,
                system_role="system",
                user_role="user",
                assistant_role="assistant",
                tool_role="tool",
            )
            if pruned_turns and not lightweight_agentic:
                self.current_topic.archived_history.extend(pruned_turns)
            logger.debug(f"Pruned {removed} turns to fit token limit")

        logger.debug(
            f"provide_context finished | messages={len(messages)} | first_role={messages[0]['role'] if messages else 'empty'} | "
            f"lightweight={lightweight_agentic} | research_mode={research_mode} | "
            f"include_logs={include_logs} | return_as_string={return_as_string} | tokens={final_tokens}"
        )

        if return_as_string:
            formatted = []
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                content = str(msg.get("content", "")).strip()
                formatted.append(f"{role}:\n{content}")
            return "\n\n".join(formatted)

        return messages

    def prune_to_fit_context(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        min_keep_messages: int = 5,
        system_role: str = "system",
        user_role: str = "user",
        assistant_role: str = "assistant",
        tool_role: str = "tool",
    ) -> Tuple[int, List[Dict[str, str]]]:
        if not messages or max_tokens <= 0:
            return 0, []

        def count_list(mes):
            if not self.tokenizer:
                return 0
            try:
                return self.tokenizer.count_message_tokens(mes)
            except Exception:
                return 0

        current = count_list(messages)
        if current <= max_tokens:
            return 0, []
        protected = 0
        for i, m in enumerate(messages):
            if m.get("role") == system_role:
                protected = i + 1
            elif m.get("role") == user_role and i == len(messages) - 1:
                protected = max(protected, i + 1)
        pruned_turns: List[Dict[str, str]] = []
        i = protected
        removed = 0
        effective_max = max_tokens - 256
        while current > effective_max and i < len(messages) - min_keep_messages:
            turn_start = i
            while i < len(messages) and messages[i].get("role") not in (
                assistant_role,
                user_role,
            ):
                i += 1
            if i >= len(messages):
                break
            turn_end = i
            while turn_end < len(messages) and messages[turn_end].get("role") in (
                assistant_role,
                tool_role,
            ):
                turn_end += 1
            if turn_end == turn_start:
                break
            pruned_turns.extend(messages[turn_start:turn_end])
            del messages[turn_start:turn_end]
            removed += turn_end - turn_start
            current = count_list(messages)
            i = turn_start
            if current > effective_max:
                for j in range(turn_start - 1, turn_start - 10, -1):
                    if j >= 0 and messages[j].get("role") in (
                        assistant_role,
                        tool_role,
                    ):
                        pruned_turns.append(messages[j])
                        del messages[j]
                        removed += 1
                        current = count_list(messages)
                        i = j
                        break
        if removed > 0:
            self._log_action(
                "prune", f"Pruned {removed} turns", {"kept": len(messages)}
            )
        logger.info(f"Pruned {removed} messages, Final Count: {current}")
        return removed, pruned_turns

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.tokenizer:
            return 0
        try:
            return self.tokenizer.count_message_tokens(messages)
        except Exception:
            return 0

    async def get_archived_context(self, query: str, max_tokens: int = 4000) -> str:
        if not self.current_topic.archived_history or not query.strip():
            return ""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not loaded...")
        query_emb = await self.fetch_embedding(query, prefix="query")
        query_words = re.findall(r"\b\w+\b", query.lower())
        scored = []
        for msg in self.current_topic.archived_history:
            text = msg["content"]
            emb = await self.fetch_embedding(text, prefix="passage")
            if len(emb) == 0:
                continue
            sim = cosine_sim(query_emb, emb)
            kw = keyword_score(query_words, text[:400])
            scored.append((0.65 * sim + 0.35 * kw, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        parts = []
        used = 0
        for _, msg in scored[:20]:
            block = f"[{msg['role'].upper()}] {msg['content'][:3000]}\n{'─' * 40}\n"
            toks = (
                self.tokenizer.count_tokens(block)
                if hasattr(self.tokenizer, "count_tokens")
                else len(block) // 4
            )
            if used + toks > max_tokens:
                break
            parts.append(block)
            used += toks
        return "".join(parts)

    async def record_tool_outcome(
        self,
        tool_name: str,
        result: Any | AsyncIterable[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record tool outcome — supports both full result and streaming."""
        metadata = metadata or {}
        timestamp = time.time()
        args = metadata.get("args", {}) or {}
        # Lightweight call history
        clean_args = {k: v for k, v in args.items() if k not in ("agent", "self")}
        call_entry = {
            "tool": tool_name,
            "args": clean_args,
            "timestamp": timestamp,
        }
        self.tool_call_history.append(call_entry)
        if len(self.tool_call_history) > 20:
            self.tool_call_history.pop(0)
        # ── STREAMING PATH ─────────────────────────────────────────────────────
        if result is not None and hasattr(result, "__aiter__"):
            logger.info(f"🔄 Streaming tool outcome detected: {tool_name}")
            await self._record_streaming_tool_outcome(
                tool_name=tool_name,
                chunk_stream=result,
                metadata=metadata,
                timestamp=timestamp,
            )
            return
        # ── FULL RESULT PATH ───────────────────────────────────────────────────
        if isinstance(result, (dict, list, tuple)):
            try:
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                result_str = str(result)
        else:
            result_str = str(result) if result is not None else ""
        # Short results → still finalize for consistency
        if len(result_str) < 30:
            self._finalize_tool_recording(
                tool_name=tool_name,
                preview=result_str[:200],
                success=True,
                metadata=metadata,
            )
            return
        is_large = len(result_str) > 800
        content = self._build_tool_content(
            tool_name, timestamp, len(result_str), is_large, result_str
        )
        await self.add_external_content(
            source_type="tool_outcome",
            content=content,
            metadata={
                "tool": tool_name,
                "timestamp": timestamp,
                "original_length": len(result_str),
                "is_large_item": is_large,
                "args": clean_args,
                "streamed": False,
                **metadata,
            },
        )
        # <<< FINALIZE with metadata >>>
        self._finalize_tool_recording(
            tool_name=tool_name,
            preview=result_str[:200],  # consistent short preview
            success=True,
            metadata=metadata,
        )

    def _get_latest_tool_outcome(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent non-streamed tool outcome.
        Smartly extracts real tool name from metadata or content.
        Safe sorting (no KnowledgeItem comparison)."""
        if not self.short_term_mem:
            return None

        tool_outcomes = [
            (item.metadata.get("timestamp", 0), item)
            for item in self.short_term_mem.values()
            if item.source_type == "tool_outcome"
            and not item.metadata.get("streamed", False)
        ]
        if not tool_outcomes:
            return None

        # Safe sort using only timestamp + unique id tie-breaker
        tool_outcomes.sort(key=lambda x: (x[0], id(x[1])), reverse=True)

        _, latest_item = tool_outcomes[0]

        full_content = latest_item.content.strip()
        if not full_content:
            return None

        # === Smart tool name extraction ===
        tool_name = latest_item.metadata.get("tool") or latest_item.metadata.get("name")

        if not tool_name:
            match = re.match(r"Tool:\s*(\S+)", full_content)
            if match:
                tool_name = match.group(1)
                lines = full_content.splitlines()
                if len(lines) > 1:
                    full_content = "\n".join(lines[1:]).strip()

        if not tool_name:
            tool_name = "tool"

        return {
            "tool_name": tool_name,
            "content": full_content,
            "timestamp": latest_item.metadata.get("timestamp", time.time()),
        }

    async def _get_recent_tool_outcomes(
        self,
        limit: int = 4,
        max_tokens_per_result: int = 2000,
        max_total_tokens: int = 5000,
    ) -> str:
        """Return the most recent tool outcomes (full results).
        Only real 'tool_outcome' entries are considered (streaming chunks are ignored).
        Respects per-result token limit for very large outputs.
        """
        if not self.short_term_mem:
            logger.debug("No entries in short_term_mem")
            return ""

        # Only consider final consolidated tool_outcomes
        tool_items = [
            (
                item.metadata.get("timestamp", 0),
                key,
                item,
                item.metadata.get("tool") or item.metadata.get("name") or "tool",
                len(item.content),
            )
            for key, item in self.short_term_mem.items()
            if item.source_type == "tool_outcome"
            and item.metadata.get("streamed", False) is not True
            and not str(key).startswith("tool_outcome_chunk")
        ]

        # Sort by timestamp descending (safe)
        tool_items.sort(key=lambda x: x[0], reverse=True)

        parts: List[str] = []
        used_tokens = 0
        for ts, key, item, tool_name, size in tool_items[:limit]:
            timestamp_str = (
                f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}\n"
                if ts > 0
                else ""
            )
            # Truncate content per result
            content = item.content
            if len(content) > max_tokens_per_result * 4:
                content = content[: max_tokens_per_result * 4] + "\n... [truncated]"

            block = PromptBuilder.latest_tool_block(
                tool_name=tool_name,
                timestamp_str=timestamp_str,
                size=size,
                content=content,
            )
            block_tokens = (
                self.tokenizer.count_tokens(block)
                if self.tokenizer
                else len(block) // 4
            )
            if used_tokens + block_tokens > max_total_tokens:
                logger.debug(
                    f"Token budget reached in _get_recent_tool_outcomes ({used_tokens}/{max_total_tokens})"
                )
                break
            parts.append(block)
            used_tokens += block_tokens

        result = "\n".join(parts) if parts else ""
        logger.debug(
            f"_get_recent_tool_outcomes returned {len(result)} chars from {len(parts)} tool outcomes"
        )
        return result

    async def _record_streaming_tool_outcome(
        self,
        tool_name: str,
        chunk_stream: AsyncIterable[str],
        metadata: dict,
        timestamp: float,
    ) -> None:
        """Accumulate text into ~768-token batches and report meaningful progress.
        Now uses adaptive + logarithmic-style progress when total size is unknown."""
        logger.info(f"🔄 Starting streaming ingestion: {tool_name}")
        full_content: List[str] = []
        accumulated: List[str] = []
        accumulated_tokens = 0
        batch_index = 0
        parent_key = (
            f"{tool_name}_"
            f"{int(timestamp)}_"
            f"{hashlib.md5(str(metadata.get('args', {})).encode()).hexdigest()[:12]}"
        )
        # Hints from tool
        total_pages = metadata.get("total_pages") or 0
        estimated_total_chars = metadata.get("estimated_total_chars", 0)
        last_reported_percent = 0
        chars_at_last_report = 0
        try:
            async for chunk in chunk_stream:
                if not chunk or not chunk.strip():
                    continue
                full_content.append(chunk)
                accumulated.append(chunk)
                accumulated_tokens += len(chunk) // 4
                # Flush batch when we reach target size (~768 tokens)
                if accumulated_tokens >= CHUNK_SIZE_TOKENS or len(accumulated) >= 8:
                    batch_text = "".join(accumulated)
                    await self._add_streaming_chunk_batch(
                        tool_name=tool_name,
                        accumulated_text=batch_text,
                        parent_key=parent_key,
                        chunk_index=batch_index,
                        metadata=metadata,
                        timestamp=timestamp,
                    )
                    batch_index += 1
                    accumulated.clear()
                    accumulated_tokens = 0
                # === IMPROVED PROGRESS REPORTING ===
                current_chars = len("".join(full_content))
                if total_pages > 0:
                    # Best case: real page count from PDF tool
                    progress = min(
                        95, int((len(full_content) / (total_pages * 1.05)) * 100)
                    )
                elif estimated_total_chars > 0:
                    progress = min(
                        95, int((current_chars / estimated_total_chars) * 100)
                    )
                else:
                    # Adaptive fallback for unknown total size (much better than fixed *1.8)
                    # Uses logarithmic growth so progress keeps moving slowly toward 95%
                    if current_chars < 10000:
                        progress = 5
                    else:
                        # log10 scaling gives nice slow increase even for very large files
                        progress = min(
                            95,
                            int(
                                10
                                + 85 * (1 - (1 / (1 + (current_chars / 50000) ** 0.6)))
                            ),
                        )
                # Report only when we cross a new 5% threshold (or every ~30-50k chars after 50%)
                if progress >= 10 and (
                    progress // 5 > last_reported_percent // 5
                    or current_chars - chars_at_last_report > 40000
                ):
                    logger.info(
                        f"📈 {tool_name} streaming: {progress}% complete "
                        f"({batch_index} batches, {current_chars:,} chars)"
                    )
                    last_reported_percent = progress
                    chars_at_last_report = current_chars
            # Final batch flush
            if accumulated:
                batch_text = "".join(accumulated)
                await self._add_streaming_chunk_batch(
                    tool_name=tool_name,
                    accumulated_text=batch_text,
                    parent_key=parent_key,
                    chunk_index=batch_index,
                    metadata=metadata,
                    timestamp=timestamp,
                )
                batch_index += 1
        except Exception as e:
            logger.error(f"Error during streaming of {tool_name}: {e}", exc_info=True)
        # === FINAL CONSOLIDATED ENTRY ===
        if full_content:
            full_result = "".join(full_content)
            total_chars = len(full_result)
            await self.add_external_content(
                source_type="tool_outcome",
                content=self._build_tool_content(
                    tool_name, timestamp, total_chars, True, full_result
                ),
                metadata={
                    "tool": tool_name,
                    "timestamp": timestamp,
                    "original_length": total_chars,
                    "is_large_item": True,
                    "streamed": True,
                    "total_chunks": batch_index,
                    "parent_key": parent_key,
                    "args": metadata.get("args", {}),
                    **metadata,
                },
            )
            logger.info(
                f"✅ Streaming completed: {tool_name} → "
                f"1 final tool_outcome + {batch_index} batches "
                f"({total_chars:,} characters)"
            )
            preview = full_result[:200] if full_result else ""
        else:
            logger.warning(
                f"Streaming completed but no content received for {tool_name}"
            )
            preview = ""
        self._finalize_tool_recording(
            tool_name=tool_name,
            preview=preview,
            success=True,
            metadata=metadata,
        )

    def _build_tool_content(
        self,
        tool_name: str,
        timestamp: float,
        size: int,
        is_large: bool,
        result_str: str,
    ) -> str:
        return (
            f"Tool: {tool_name}\n"
            f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n"
            f"Result size: {size:,} characters\n"
            f"Large item: {is_large}\n"
            f"Streamed: True\n\n"
            f"{result_str}"
        )

    def _finalize_tool_recording(
        self,
        tool_name: str,
        preview: str,
        success: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """Original lines preserved + moved add_tool_result logic.
        Now properly receives metadata when available."""
        metadata = metadata or {}
        # === YOUR ORIGINAL LINES ===
        self.tools_executed.append(tool_name)
        logger.debug(f"tool_outcome {tool_name} → {preview}...")
        self._log_action("tool_outcome", f"{tool_name} → {preview}...", {})
        if self.agent is not None and hasattr(self.agent, "state"):
            try:
                self.agent.state.add_tool_result(
                    tool_name=tool_name,
                    result=preview,
                    success=success,
                )
                # Update last_tool_success with full context
                self.agent.state.last_tool_success = {
                    "tool_name": tool_name,
                    "success": success,
                    "timestamp": time.time(),
                    "step_index": getattr(self.agent.state, "current_task_index", -1),
                    "streamed": metadata.get("streamed", False),
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k in ("args", "usage_count", "action_type")
                    },
                }
            except Exception as e:
                logger.warning(f"[STATE RECORD FAILED] {tool_name}: {e}")
        else:
            logger.debug(
                f"Skipping add_tool_result for {tool_name}: no agent.state available"
            )

    async def sync_to_agent_state(self, state: "AgentState") -> None:
        """Automatically populate AgentState from ContextManager (single source of truth).
        investigation_state removed — planning now lives only in agent.state + _task_contexts.
        """
        # 1. Messages (light mirror)
        if self.mode == "chat":
            state.messages = self.pure_chat_history[-20:]
        else:
            recent = (
                self.current_topic.history[-15:] if self.current_topic.history else []
            )
            state.messages = [msg.copy() for msg in recent]
        state.message_count = len(state.messages)
        state.last_message_time = time.time()
        # 2. Tool tracking
        state.tool_results = [
            {
                "name": entry.get("tool", "unknown"),
                "result": str(entry.get("result", ""))[:300]
                + ("..." if len(str(entry.get("result", ""))) > 300 else ""),
                "success": True,
                "timestamp": entry.get("timestamp", time.time()),
            }
            for entry in self.tool_call_history[-12:]
        ]
        state.executed_functions = [
            {"name": t, "timestamp": time.time()} for t in self.tools_executed[-20:]
        ]
        # 3. Investigation / Planning state
        if hasattr(self, "_task_contexts") and self._task_contexts:
            state.planned_tasks = list(self._task_contexts.keys())
            state.task_expected_outcomes = [
                f"Complete task: {name}" for name in state.planned_tasks
            ]
        # else: leave whatever is already in state (from build_tasks_from_plan etc.)
        if state.planned_tasks and isinstance(state.current_task_index, int):
            if 0 <= state.current_task_index < len(state.planned_tasks):
                state.current_task = state.planned_tasks[state.current_task_index]
        # 4. Sub-task awareness
        state.sub_task_ids = (
            list(getattr(self, "_task_contexts", {}).keys())
            if hasattr(self, "_task_contexts")
            else state.sub_task_ids or []
        )
        state.active_sub_agents = [f"task_{k}" for k in state.sub_task_ids]
        # 5. Metrics
        state.loop_count = self.context_stats.get("messages_added", 0) // 2
        state.total_tool_calls = len(self.tools_executed)
        logger.debug(
            f"ContextManager → AgentState synced | tasks={len(state.planned_tasks)}"
        )
