import re
import os
import asyncio
import time
import numpy as np
import hashlib
import json
from typing import Tuple, List, Dict, Any, Optional
from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder as PromptHelper
from neuralcore.utils.config import get_loader
from neuralcore.clients.factory import get_clients
from neuralcore.utils.text_tokenizer import TextTokenizer
from neuralcore.utils.search import keyword_score, cosine_sim
from neuralcore.agents.state import AgentState

# scikit-learn for sparse vectors
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    FASTEMBED_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# TOPIC & OFF-TOPIC DETECTION
# ─────────────────────────────────────────────────────────────
MSG_THR = 0.55  # Slightly higher → more stable topic matching
NUM_MSG = 8  # More messages considered for analysis
OFF_THR = 0.65  # Lowered a bit → catches drifting conversation earlier
OFF_FREQ = 4
SLICE_SIZE = 6  # Bigger window for off-topic detection

# ─────────────────────────────────────────────────────────────
# CHUNKING STRATEGY (Best balance for 64GB RAM)
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 768  # ← Recommended increase from 512
CHUNK_OVERLAP_TOKENS = 128  # ~17% overlap (good sweet spot)
MAX_CHUNKS_PER_ITEM = 12  # Allow more chunks for large files/code
TOOL_OUTCOME_NO_CHUNK_THRESHOLD = 1500  # Only chunk very large tool outputs

logger = Logger.get_logger()


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE ITEM
# ─────────────────────────────────────────────────────────────
class KnowledgeItem:
    def __init__(
        self,
        key: str,
        source_type: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ):
        self.key = key
        self.source_type = source_type
        self.content = content
        self.metadata = metadata or {}
        self.embedding: np.ndarray = np.array([])
        self.sparse_vector = None
        self.word_set = set(re.findall(r"\b\w+\b", content.lower()))


# ─────────────────────────────────────────────────────────────
# TOPIC
# ─────────────────────────────────────────────────────────────
class Topic:
    def __init__(self, name: str = "", description: str = "") -> None:
        self.name = name
        self.description = description
        self.embedded_description = np.array([])
        self.history: List[Dict[str, str]] = []
        self.history_tokens: List[int] = []
        self.archived_history: List[Dict[str, str]] = []
        self.history_embeddings: List[np.ndarray] = []

    async def add_message(
        self, role: str, content: str, embedding: np.ndarray, token_count: int
    ) -> None:
        self.history.append({"role": role, "content": content})
        self.history_embeddings.append(embedding)
        self.history_tokens.append(token_count)


# ─────────────────────────────────────────────────────────────
# CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────
class ContextManager:
    def __init__(self, max_tokens: int = 28000) -> None:
        self.max_tokens = max_tokens
        clients = get_clients()
        self.client = clients.get("main")
        self.embeddings = clients.get("embeddings")

        loader = get_loader()
        embed_config = loader.get_client_config("embeddings") or {}

        self.use_fastembed = embed_config.get("use_fastembed", True)
        self.fast_embedder = None
        self.fastembed_model = None

        # Initialize FastEmbed with full config support
        if self.use_fastembed and TextEmbedding is not None:
            self._init_fastembed(embed_config)

        # Tokenizer setup (unchanged)
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
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
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
        self.tools_executed: List[str] = []
        self.context_stats = {
            "kb_added": 0,
            "messages_added": 0,
            "topics_switched": 0,
            "prunes": 0,
        }
        self.investigation_state = {
            "goal": "",
            "subtasks": [],
            "completed": [],
            "pending": [],
            "findings": [],
            "hypotheses": [],
            "unknowns": [],
        }

        self.mode: str = "chat"
        self._mode_lock = asyncio.Lock()
        self.pure_chat_history: List[Dict[str, str]] = []
        self._last_analysis_ts = 0.0
        self.tools_executed: List[str] = []
        self.tool_call_history: List[Dict[str, Any]] = []

    def _init_fastembed(self, embed_config: dict):
        """Helper method to initialize FastEmbed with full path handling."""
        model_name = embed_config.get("fastembed_model", "BAAI/bge-small-en-v1.5")

        local_path = embed_config.get("fastembed_local_path")
        cache_dir = embed_config.get("fastembed_cache_dir")
        local_files_only = embed_config.get("fastembed_local_files_only", False)

        init_kwargs = {"model_name": model_name}

        try:
            # === Handle tilde expansion for local_path ===
            if local_path:
                # Expand ~, ~user, and environment variables
                expanded_local = os.path.expanduser(local_path)
                expanded_local = os.path.expandvars(
                    expanded_local
                )  # in case someone uses $HOME
                expanded_local = os.path.abspath(expanded_local)  # make it absolute

                if os.path.isdir(expanded_local):
                    init_kwargs["specific_model_path"] = expanded_local
                    logger.info(
                        f"✅ FastEmbed: Loading from local path → {expanded_local}"
                    )
                else:
                    logger.warning(
                        f"⚠️  fastembed_local_path '{local_path}' (resolved to '{expanded_local}') does not exist. "
                        f"Falling back to Hugging Face cache/download."
                    )

            # === Cache directory (also expand ~) ===
            if cache_dir:
                expanded_cache = os.path.abspath(
                    os.path.expanduser(os.path.expandvars(cache_dir))
                )
                init_kwargs["cache_dir"] = expanded_cache
                logger.info(f"   FastEmbed cache directory: {expanded_cache}")

            # === Offline mode ===
            if local_files_only:
                init_kwargs["local_files_only"] = True
                logger.info("   FastEmbed: local_files_only=True (strict offline mode)")
            if self.use_fastembed and TextEmbedding is not None:
                # Initialize
                self.fast_embedder = TextEmbedding(**init_kwargs)
                self.fastembed_model = model_name

                logger.info(f"✅ FastEmbed ready (model: {model_name})")

        except Exception as e:
            logger.warning(f"⚠️ FastEmbed init failed: {e}")
            self.fast_embedder = None

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
            hard: If True (default), clears embedding cache and sparse index completely.
                  If False, keeps cache for faster re-initialization.
        """
        logger.info(f"ContextManager.reset() called — hard={hard}")

        # Core identity & mode
        self.mode = "chat"
        self.current_topic = Topic("Initial topic")
        self.topics.clear()

        # Knowledge Base & embeddings
        self.knowledge_base.clear()
        self.embedding_cache.clear() if hard else None
        self._sparse_index_dirty = True
        self.tfidf_matrix = None
        self.kb_keys_ordered.clear()
        self._last_rebuild_size = 0
        self.tfidf_vectorizer = None  # will be re-created on next use

        # Histories & topics
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

        # Tool & action tracking
        self.tools_executed.clear()
        self.tool_call_history.clear()
        self.action_log.clear()
        self.files_checked.clear()

        # Investigation & task state
        self.investigation_state = {
            "goal": "",
            "subtasks": [],
            "completed": [],
            "pending": [],
            "findings": [],
            "hypotheses": [],
            "unknowns": [],
        }
        if hasattr(self, "_task_contexts"):
            self._task_contexts.clear()

        # Stats & FS state
        self.context_stats = {
            "kb_added": 0,
            "messages_added": 0,
            "topics_switched": 0,
            "prunes": 0,
        }
        self.fs_state = {"cwd": ".", "known_folders": [], "negative_findings": []}

        # Clear any pending locks if needed (but _mode_lock stays alive)
        logger.info(
            f"ContextManager reset complete. "
            f"KB cleared: {len(self.knowledge_base)} items remaining. "
            f"Mode: {self.mode}"
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

        def add_finding(self, finding: str):
            cleaned = finding[:400]
            self.findings.append(cleaned)
            self.parent.add_finding(cleaned)

        def add_hypothesis(self, hypothesis: str):
            self.hypotheses.append(hypothesis[:300])

        def get_context(self, max_tokens: int = 3500) -> str:
            lines = [f"🔹 TASK CONTEXT: {self.name.upper()}"]
            if self.important_files:
                files_str = ", ".join(sorted(list(self.important_files))[:12])
                lines.append(f"Important files: {files_str}")
            for r in self.key_results[-10:]:
                lines.append(f"• {r['title']}: {r['summary']}")
            if self.findings:
                lines.append(
                    "Key Findings:\n"
                    + "\n".join(f"   - {f}" for f in self.findings[-8:])
                )
            if self.hypotheses:
                lines.append(
                    "Hypotheses:\n"
                    + "\n".join(f"   - {h}" for h in self.hypotheses[-5:])
                )
            text = "\n".join(lines)
            return text[: max_tokens * 4]

    def create_task_context(self, task_name: str) -> "ContextManager.TaskContext":
        if not hasattr(self, "_task_contexts"):
            self._task_contexts: Dict[str, ContextManager.TaskContext] = {}
        if task_name not in self._task_contexts:
            self._task_contexts[task_name] = self.TaskContext(task_name, self)
            self.add_subtask(task_name)
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
            fast_embedder = self.fast_embedder  # Pyright narrowing
            input_text = f"{prefix}: {text}" if prefix else text
            try:

                def _embed_sync():
                    return np.asarray(
                        list(fast_embedder.embed([input_text]))[0], dtype=np.float32
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
        # Simple but effective: paths, filenames with extensions, quoted strings, etc.
        patterns = [
            r'["\']([^"\']+\.(?:py|js|ts|json|txt|md|log|csv|yaml|yml|toml))["\']',  # common extensions
            r'["\']([/\\][^"\']+)["\']',  # paths
            r"\b(\S+\.\S{2,5})\b",  # filename-like
        ]
        mentions = []
        for pat in patterns:
            mentions.extend(re.findall(pat, query))
        # Also plain words that might be filenames/IDs (fallback)
        words = re.findall(r"\b\w{3,}\b", query)
        mentions.extend([w for w in words if "." in w or "/" in w or "\\" in w])
        return list(dict.fromkeys(mentions))  # dedup preserve order

    # ─────────────────────────────────────────────────────────────
    # CONTEXT SUMMARY, PRUNING, INVESTIGATION STATE (unchanged)
    # ─────────────────────────────────────────────────────────────
    def get_context_summary(self, max_messages: int = 10, max_chars: int = 1000) -> str:
        if self.mode == "chat":
            if not self.pure_chat_history:
                return ""
            recent = self.pure_chat_history[-4:]
            lines = ["📋 CHAT CONTEXT (light)"]
            lines.extend(
                f"[{msg['role'].upper()}] {msg['content'][:120]}..." for msg in recent
            )
            return "\n".join(lines)
        files = sorted(list(self.files_checked))
        tools = self.tools_executed
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
        token_warning = (
            "⚠️ Approaching context window limit..."
            if total_tokens > self.max_tokens * 0.8
            else ""
        )
        lines = [
            "📋 LIVE CONTEXT SUMMARY",
            f"• Files checked: {', '.join(files) if files else '—'}",
            f"• Tools used: {', '.join(tools) if tools else '—'}",
            f"• KB items: {len(self.knowledge_base)}",
            f"• Estimated tokens: {total_tokens} / {self.max_tokens} {token_warning}",
            f"• Active topic: {self.current_topic.name}",
            "",
            "📝 LAST MESSAGES",
            *message_summaries,
        ]
        inv = self.investigation_state
        inv_block = [
            "",
            "🧠 INVESTIGATION STATE",
            f"• Goal: {inv['goal'] or '—'}",
            f"• Pending: {', '.join(inv['pending'][:5]) if inv['pending'] else '—'}",
            f"• Completed: {', '.join(inv['completed'][-5:]) if inv['completed'] else '—'}",
        ]
        lines.extend(inv_block)
        full_summary = "\n".join(lines)
        if len(full_summary) > max_chars:
            cutoff = max_chars - 50
            summary_lines = full_summary[:cutoff].rsplit("\n", 1)[0]
            return summary_lines + "\n…(truncated)"
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

    def set_goal(self, goal: str):
        self.investigation_state["goal"] = goal

    def add_subtask(self, task: str):
        if task not in self.investigation_state["subtasks"]:
            self.investigation_state["subtasks"].append(task)
            self.investigation_state["pending"].append(task)

    def complete_subtask(self, task: str):
        if task in self.investigation_state["pending"]:
            self.investigation_state["pending"].remove(task)
            self.investigation_state["completed"].append(task)

    def add_finding(self, finding: str):
        self.investigation_state["findings"].append(finding[:500])

    def add_unknown(self, unknown: str):
        self.investigation_state["unknowns"].append(unknown[:300])

    # ─────────────────────────────────────────────────────────────
    # CHUNKING (improved – uses real tokenizer when available)
    # ─────────────────────────────────────────────────────────────
    def _chunk_text(
        self, text: str, parent_key: str, source_type: str
    ) -> List[KnowledgeItem]:
        if len(text) < TOOL_OUTCOME_NO_CHUNK_THRESHOLD:
            chunk_texts = [text]
        else:
            if self.tokenizer:
                # Use the proper chunking method from TextTokenizer (respects inner tokenizer.encode/decode)
                chunk_texts = self.tokenizer.split_text_into_chunks(
                    text,
                    max_tokens=CHUNK_SIZE_TOKENS,
                    overlap=CHUNK_OVERLAP_TOKENS,
                )
                chunk_texts = chunk_texts[:MAX_CHUNKS_PER_ITEM]
            else:
                # fallback: character-based
                chunk_size = CHUNK_SIZE_TOKENS * 4
                overlap = CHUNK_OVERLAP_TOKENS * 4
                chunks = []
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunks.append(text[start:end])
                    start = end - overlap
                chunk_texts = chunks[:MAX_CHUNKS_PER_ITEM]

        items = []
        for i, chunk_content in enumerate(chunk_texts):
            if not chunk_content.strip():
                continue
            chunk_key = f"{parent_key}_chunk_{i}"
            metadata = {
                "parent_key": parent_key,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "source_type": source_type,
            }
            item = KnowledgeItem(chunk_key, source_type, chunk_content, metadata)
            items.append(item)
        return items

    # ─────────────────────────────────────────────────────────────
    # ADD EXTERNAL CONTENT (now lazy, does NOT rebuild immediately)
    # ─────────────────────────────────────────────────────────────

    async def add_external_content(
        self, source_type: str, content: str, metadata: Dict[str, Any] | None = None
    ) -> str | None:
        if not content or len(content.strip()) < 15:
            return None

        metadata = metadata or {}
        content_lower = content.lower()

        # ── Smart, less aggressive guard ──
        forbidden = [
            "You are a helpful Terminal AI agent",
            "CONTEXT WINDOW STATUS",
            "RELEVANT EXTERNAL CONTEXT",
            "CHAT CONTEXT (light)",
            "[FINAL_ANSWER_COMPLETE]",
            "AUTONOMOUS CONTINUATION",
        ]

        is_contaminated = any(phrase.lower() in content_lower for phrase in forbidden)

        if is_contaminated:
            if source_type in ("tool_outcome", "task_result"):
                # Clean instead of block for legitimate results
                for phrase in forbidden:
                    content = re.sub(
                        re.escape(phrase) + r".*?(\n|$)",
                        "",
                        content,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                logger.debug(f"[KB GUARD] Partially cleaned {source_type}")
            else:
                logger.warning(
                    f"[KB GUARD] Blocked contaminated content (source={source_type})"
                )
                return None

        # Deduplication
        parent_key = (
            metadata.get("path")
            or metadata.get("filename")
            or f"{source_type}_{hash(content[:300])}"
        )

        if parent_key and str(parent_key) in self.files_checked:
            return None

        chunk_items = self._chunk_text(content, parent_key, source_type)

        added = 0
        for item in chunk_items:
            if item.key in self.knowledge_base:
                continue
            item.embedding = await self.fetch_embedding(item.content, prefix="passage")
            self.knowledge_base[item.key] = item
            added += 1

        if added > 0:
            self._sparse_index_dirty = True
            self.context_stats["kb_added"] += added
            logger.info(
                f"✅ Added {added} clean chunks for {parent_key} | source={source_type}"
            )
        else:
            logger.debug(f"→ Skipped duplicate/empty content for {parent_key}")

        if "path" in metadata or "filename" in metadata:
            self.files_checked.add(str(parent_key))

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
        if len(self.knowledge_base) < 1:
            self.tfidf_matrix = None
            self.kb_keys_ordered = []
            return
        texts = [item.content for item in self.knowledge_base.values()]
        self.kb_keys_ordered = list(self.knowledge_base.keys())

        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                min_df=1,  # ← critical
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
        self._last_rebuild_size = len(self.knowledge_base)

    async def _ensure_sparse_index(self):
        """Lazy + async rebuild."""
        if (
            not self._sparse_index_dirty
            and len(self.knowledge_base) == self._last_rebuild_size
        ):
            return
        await self._rebuild_sparse_index()
        self._sparse_index_dirty = False

    # ─────────────────────────────────────────────────────────────
    # SPARSE RETRIEVAL - FIXED TF-IDF CRASH
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

                # Force refit if necessary
                if (
                    not hasattr(self.tfidf_vectorizer, "vocabulary_")
                    or len(self.tfidf_vectorizer.vocabulary_) == 0
                ):
                    if len(self.knowledge_base) > 0:
                        texts = [item.content for item in self.knowledge_base.values()]
                        logger.debug(f"Fitting TF-IDF on {len(texts)} documents")
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                    else:
                        return {}

                query_vec = self.tfidf_vectorizer.transform([query])
                matrix = self.tfidf_matrix
                vec = query_vec

                # Use .dot() with explicit transpose
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
        """Hybrid RAG with parameter/file-aware boosting for exact tool matches."""
        if not self.knowledge_base or not query.strip():
            logger.debug(
                "retrieve_relevant_knowledge: KB empty or query empty → returning ''"
            )
            return ""

        logger.debug(
            f"retrieve_relevant_knowledge START | query='{query[:100]}...' | "
            f"KB_size={len(self.knowledge_base)} | max_kb_tokens={max_kb_tokens}"
        )

        embedding_query = f"Search query for relevant tool results and context: {query}"
        query_emb = await self.fetch_embedding(embedding_query, prefix="query")

        sparse_scores = await self._sparse_retrieve(query)

        # Dense ranking
        dense_ranked = []
        if query_emb.size > 0:
            for key, item in self.knowledge_base.items():
                if item.embedding.size > 0:
                    sim = cosine_sim(query_emb, item.embedding)
                    dense_ranked.append((sim, key))
            dense_ranked.sort(reverse=True)
            dense_ranked = dense_ranked[:50]

        # Sparse ranking
        sparse_ranked = sorted(
            [(score, key) for key, score in sparse_scores.items() if score > 0],
            reverse=True,
        )[:50]

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        k = 60
        for rank, (_, key) in enumerate(dense_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        for rank, (_, key) in enumerate(sparse_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)

        # ── PARAMETER/FILE BOOST: If query mentions a file/param from any tool call ──
        file_mentions = self._extract_potential_file_or_param_mentions(query)
        if file_mentions:
            logger.debug(f"Detected potential file/param mentions: {file_mentions}")
            for key, item in self.knowledge_base.items():
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
                        boost += 5.0  # strong exact match in args/metadata
                    elif any(
                        word in content_lower
                        for word in m_lower.split()
                        if len(word) > 2
                    ):
                        boost += 2.0  # softer keyword overlap

                if boost > 0:
                    current = rrf_scores.get(key, 0.0)
                    rrf_scores[key] = current + boost
                    logger.debug(
                        f"Boosted tool outcome {key} by {boost:.1f} for mentions {file_mentions}"
                    )

        # Final scored list
        scored = sorted(
            [
                (rrf_scores.get(key, 0.0), key, self.knowledge_base[key])
                for key in rrf_scores
            ],
            reverse=True,
        )

        parts: List[str] = []
        used = 0
        tool_outcome_count = 0
        blocked_count = 0

        for score, key, item in scored[:40]:
            # Contamination guard
            if any(
                forbidden in item.content
                for forbidden in [
                    "You are a helpful Terminal AI agent",
                    "CONTEXT WINDOW STATUS",
                    "RELEVANT EXTERNAL CONTEXT",
                    "CHAT CONTEXT (light)",
                    "AUTONOMOUS CONTINUATION",
                    "[FINAL_ANSWER_COMPLETE]",
                ]
            ):
                blocked_count += 1
                continue

            if item.source_type == "tool_outcome":
                tool_outcome_count += 1

            meta_str = ", ".join(f"{k}={v}" for k, v in item.metadata.items() if v)
            header = f"[{item.source_type.upper()}] {item.key}\nMetadata: {meta_str}\n"
            block = f"{header}{item.content}\n{'─' * 50}\n"

            block_tokens = (
                self.tokenizer.count_tokens(block)
                if self.tokenizer
                else len(block) // 4
            )

            if used + block_tokens > max_kb_tokens:
                logger.debug(
                    f"Token budget reached ({used}/{max_kb_tokens}) — stopping"
                )
                break

            parts.append(block)
            used += block_tokens

        result = "".join(parts)

        logger.debug(
            f"retrieve_relevant_knowledge FINISHED | "
            f"returned_chars={len(result)} | "
            f"tool_outcomes_included={tool_outcome_count} | "
            f"blocked_contaminated={blocked_count} | "
            f"total_scored={len(scored)}"
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
    # SWITCH / MATCH TOPIC (unchanged)
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
    # HISTORY ANALYSIS (unchanged except topic generator)
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
                response = await self.client.ask(PromptHelper.topics_helper(history))
                if not isinstance(response, str):
                    attempt += 1
                    continue

                clean = re.sub(
                    r"^```(?:json)?|```$", "", response, flags=re.IGNORECASE
                ).strip()

                # PRODUCTION: JSON-first parsing
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
                    f"Topic extraction failed (attempt {attempt + 1})", exc_info=True
                )
                attempt += 1
        return None, None

    # ─────────────────────────────────────────────────────────────
    # PROVIDE CONTEXT, PRUNE, TOKEN COUNT, ARCHIVED CONTEXT, RECORD TOOL OUTCOME
    # ─────────────────────────────────────────────────────────────
    async def provide_context(
        self,
        query: str = "",
        max_input_tokens: int = 22000,
        reserved_for_output: int = 2048,
        system_prompt: str = "You are a helpful Terminal AI agent with full coding and filesystem support.",
        include_logs: bool = False,
        min_history_tokens: int = 2000,
        max_kb_tokens: int = 5000,
        chat: bool = False,
        state: Optional["AgentState"] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        logger.debug(
            f"provide_context called | query='{query[:100]}...' | chat={chat} | has_state={state is not None}"
        )

        if chat:
            logger.debug("→ Entering CHAT mode")
            # ── RICH CHAT MODE (unchanged) ──
            clean_system = system_prompt.strip()
            messages.append({"role": "system", "content": clean_system})

            if self.pure_chat_history:
                recent = self.pure_chat_history[-12:]
                for msg in recent:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            if query.strip() and self.knowledge_base:
                kb_text = await self._retrieve_relevant_knowledge(
                    query, max_kb_tokens // 2
                )
                if kb_text:
                    messages.append(
                        {
                            "role": "system",
                            "content": f"=== RELEVANT TOOL RESULTS & FILES ===\n{kb_text}\n=== END CONTEXT ===",
                        }
                    )

            messages.append(
                {
                    "role": "user",
                    "content": query.strip() or "[AUTONOMOUS CONTINUATION]",
                }
            )

            total_tokens = self.count_tokens(messages)
            logger.debug(f"CHAT mode - tokens before prune: {total_tokens}")
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
            return messages

        # ── AGENTIC MODE ──
        logger.debug("→ Entering AGENTIC mode")
        base_system = system_prompt.strip()
        summary = self.get_context_summary()
        full_system = base_system
        if summary:
            full_system += f"\n\n{summary}\n\n→ Use this context to stay aware. Never repeat the summary."

        if include_logs:
            try:
                log_lines = Logger.get_log_data(level="info", max_entries=100)
                if log_lines:
                    full_system += (
                        f"\n\n=== RECENT LOGS ===\n"
                        + "\n".join(log_lines)
                        + "\n=== END LOGS ==="
                    )
            except Exception:
                pass

        messages.append({"role": "system", "content": full_system})
        tokens_used = self.count_tokens(messages)
        logger.debug(f"Base system + summary → tokens_used = {tokens_used}")

        # ====================== GOAL & STATE AWARENESS (placed early for better reasoning) ======================
        if state is not None:
            logger.debug(
                "Adding centralized objective reminder from AgentState.get_objective_reminder()"
            )
            objective_text = state.get_objective_reminder()

            messages.append(
                {
                    "role": "system",
                    "content": f"=== OBJECTIVE & CURRENT STATE ===\n{objective_text}\n=== END STATE ===",
                }
            )
            tokens_used = self.count_tokens(messages)
            logger.debug(
                f"Added AgentState objective block → tokens_used = {tokens_used}"
            )

        # ====================== TOOL CALL HISTORY + LAST 3 FULL OUTPUTS ======================
        # Lightweight call history (tools + params only) — always available for operational awareness
        call_history_text = ""
        if self.tool_call_history:
            parts: List[str] = []
            for entry in reversed(self.tool_call_history[-12:]):  # last 12 calls
                ts_str = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
                args_str = (
                    json.dumps(entry["args"], ensure_ascii=False)
                    if entry.get("args")
                    else "{}"
                )
                parts.append(f"[{ts_str}] Tool: {entry['tool']} | Args: {args_str}")
            call_history_text = "\n".join(parts)

        # Full output from the last 3 executions (high-fidelity recent results)
        last_three_text = await self._get_recent_tool_outcomes(
            limit=3,
            max_tokens_per_result=3000,
            max_total_tokens=9000,
        )

        if call_history_text or last_three_text:
            tool_block = "=== TOOL CALL HISTORY & RECENT OUTPUTS ===\n"
            if call_history_text:
                tool_block += f"Recent tool calls (with parameters only):\n{call_history_text}\n\n"
            if last_three_text:
                tool_block += (
                    f"Full output from the last 3 tool executions:\n{last_three_text}\n"
                )
            tool_block += (
                "Use the call history to avoid repeating the same actions. "
                "Use the full outputs directly when they are relevant to the current goal."
            )
            messages.append({"role": "system", "content": tool_block})
            tokens_used = self.count_tokens(messages)
            logger.debug(
                f"Added tool call history + last 3 outputs → tokens_used now = {tokens_used}"
            )

        # ====================== TOKEN BUDGETING ======================
        query_tokens = (
            self.count_tokens([{"role": "user", "content": query}])
            if query.strip()
            else 0
        )
        target = max_input_tokens - reserved_for_output
        remaining = max(0, target - tokens_used - query_tokens)

        logger.debug(
            f"Token budget → target={target} | used={tokens_used} | remaining={remaining} | query_tokens={query_tokens}"
        )

        kb_tokens = min(max_kb_tokens, remaining // 2)
        history_budget = max(remaining - kb_tokens, min_history_tokens)

        logger.debug(f"Allocated → KB={kb_tokens} | History={history_budget}")

        # ====================== LONG-TERM SEMANTIC KB (with file/param boost) ======================
        if query.strip() and kb_tokens > 500:
            kb_text = await self._retrieve_relevant_knowledge(query, kb_tokens)
            if kb_text:
                messages.append(
                    {
                        "role": "system",
                        "content": f"=== RELEVANT EXTERNAL CONTEXT ===\n{kb_text}\n=== END EXTERNAL CONTEXT ===",
                    }
                )
                tokens_used = self.count_tokens(messages)
                remaining = max(0, target - tokens_used - query_tokens)
                history_budget = max(remaining, min_history_tokens)
                logger.debug(f"Added KB context → tokens_used now = {tokens_used}")

        # History
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
        logger.debug(f"Added history messages → tokens_used = {tokens_used}")

        messages.append(
            {
                "role": "user",
                "content": query.strip() or "[AUTONOMOUS CONTINUATION]",
            }
        )

        final_tokens = self.count_tokens(messages)
        logger.debug(f"Final context before prune → {final_tokens} tokens")

        if final_tokens > target:
            removed, pruned_turns = self.prune_to_fit_context(
                messages,
                max_tokens=target,
                min_keep_messages=5,
                system_role="system",
                user_role="user",
                assistant_role="assistant",
                tool_role="tool",
            )
            if pruned_turns:
                self.current_topic.archived_history.extend(pruned_turns)
            logger.debug(f"Pruned {removed} turns to fit token limit")

        logger.debug(
            f"provide_context finished | final messages count = {len(messages)}"
        )
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
        result: Any,
        metadata: dict | None = None,
    ):
        """Record FULL tool result + lightweight call history.
        Skips duplicate calls with identical tool name + arguments to prevent KB bloat."""
        metadata = metadata or {}
        timestamp = time.time()
        args = metadata.get("args", {}) or {}

        # ── NEW: Lightweight tool call history (name + params only, no output) ──
        clean_args = {k: v for k, v in args.items() if k not in ("agent", "self")}
        call_entry = {
            "tool": tool_name,
            "args": clean_args,
            "timestamp": timestamp,
        }
        self.tool_call_history.append(call_entry)
        if len(self.tool_call_history) > 20:  # keep last 20 calls
            self.tool_call_history.pop(0)

        logger.debug(f"[TOOL CALL LOGGED] {tool_name} with args={clean_args}")

        # ── DEDUPLICATION: Check if this exact tool+args was already recorded recently ──
        if self._is_duplicate_tool_call(tool_name, clean_args):
            logger.info(
                f"[TOOL DEDUP] Skipping KB recording for duplicate call: {tool_name} "
                f"with args={clean_args}"
            )
            # Still log the action for summary/stats
            self.tools_executed.append(tool_name)
            self._log_action(
                "tool_outcome",
                f"{tool_name} (deduplicated) → skipped duplicate",
                metadata,
            )
            return

        # ── 1. Convert result to string with maximum fidelity (no size cap) ──
        if isinstance(result, (dict, list, tuple)):
            try:
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                result_str = str(result)
        else:
            result_str = str(result) if result is not None else ""

        original_size = len(result_str)

        # ── 2. Minimal cleaning only ──
        result_str = re.sub(
            r"(?i)You are a helpful Terminal AI agent.*?(?=\n\n|\Z)",
            "",
            result_str,
            flags=re.DOTALL,
        )

        cleaned_size = len(result_str)

        if cleaned_size < 30:
            logger.debug(f"[TOOL OUTCOME] Skipped tiny/empty result from {tool_name}")
            return

        # ── 3. DEBUG: Show exactly what we are about to save ──
        preview = result_str[:600] + ("..." if len(result_str) > 600 else "")
        logger.debug(
            f"[RECORDING FULL] {tool_name} | "
            f"original={original_size:,} chars | "
            f"after_clean={cleaned_size:,} chars\n"
            f"Preview:\n{preview}"
        )

        # ── 4. Build content block ──
        content = (
            f"Tool: {tool_name}\n"
            f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n"
            f"Result size: {original_size:,} characters\n\n"
            f"{result_str}"
        )

        # ── 5. Save to knowledge base (full result) ──
        await self.add_external_content(
            source_type="tool_outcome",
            content=content,
            metadata={
                "tool": tool_name,
                "timestamp": timestamp,
                "original_length": original_size,
                "args": clean_args,  # store args for boosting & dedup
                **metadata,
            },
        )

        self.tools_executed.append(tool_name)
        self._log_action(
            "tool_outcome", f"{tool_name} → {result_str[:120]}...", metadata
        )

        logger.info(
            f"✅ Recorded full tool outcome: {tool_name} ({original_size:,} chars)"
        )

    async def _get_recent_tool_outcomes(
        self,
        limit: int = 4,
        max_tokens_per_result: int = 2000,
        max_total_tokens: int = 5000,
    ) -> str:
        """Guaranteed to return the MOST RECENT tool result (full directory listing)."""
        if not self.knowledge_base:
            logger.debug("No entries in knowledge_base")
            return ""

        tool_items = []
        for key, item in self.knowledge_base.items():
            if item.source_type == "tool_outcome":
                ts = item.metadata.get("timestamp", 0)
                tool_name = item.metadata.get("tool", "unknown")
                size = len(item.content)
                tool_items.append((ts, key, item, tool_name, size))

        # Sort by timestamp (newest first)
        tool_items.sort(reverse=True)

        logger.info(f"_get_recent_tool_outcomes: Found {len(tool_items)} tool outcomes")
        for i, (ts, _, _, name, size) in enumerate(tool_items[:3]):
            logger.info(f"  #{i + 1} → {name} | {size:,} chars | ts={ts:.3f}")

        parts: List[str] = []
        used_tokens = 0

        for ts, key, item, tool_name, size in tool_items[:limit]:
            timestamp_str = (
                f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}\n"
                if ts > 0
                else ""
            )

            block = (
                f"🔥 LATEST TOOL: {tool_name}\n"
                f"{timestamp_str}"
                f"Result size: {size:,} characters\n\n"
                f"{item.content}\n"
                f"{'─' * 90}\n"
            )

            block_tokens = (
                self.tokenizer.count_tokens(block)
                if self.tokenizer
                else len(block) // 4
            )

            if used_tokens + block_tokens > max_total_tokens:
                logger.debug(f"Token budget reached after {len(parts)} results")
                break

            parts.append(block)
            used_tokens += block_tokens

            logger.info(
                f"→ Included {tool_name} ({size:,} chars, ~{block_tokens} tokens)"
            )

        if not parts:
            logger.warning("No recent tool results returned")
            return ""

        logger.info(
            f"_get_recent_tool_outcomes → RETURNED {len(parts)} results ({used_tokens} tokens)"
        )
        return "\n".join(parts)
