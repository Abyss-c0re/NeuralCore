import re
import asyncio
import time
import numpy as np
import hashlib
import json
from typing import Tuple, List, Dict, Any
from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder as PromptHelper
from neuralcore.utils.config import get_loader
from neuralcore.clients.factory import get_clients
from neuralcore.utils.text_tokenizer import TextTokenizer
from neuralcore.utils.search import keyword_score, cosine_sim

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
MSG_THR = 0.55          # Slightly higher → more stable topic matching
NUM_MSG = 8             # More messages considered for analysis
OFF_THR = 0.65          # Lowered a bit → catches drifting conversation earlier
OFF_FREQ = 4
SLICE_SIZE = 6          # Bigger window for off-topic detection

# ─────────────────────────────────────────────────────────────
# CHUNKING STRATEGY (Best balance for 64GB RAM)
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 768         # ← Recommended increase from 512
CHUNK_OVERLAP_TOKENS = 128      # ~17% overlap (good sweet spot)
MAX_CHUNKS_PER_ITEM = 12        # Allow more chunks for large files/code
TOOL_OUTCOME_NO_CHUNK_THRESHOLD = 1500   # Only chunk very large tool outputs

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

        if self.use_fastembed and TextEmbedding:
            try:
                model_name = embed_config.get(
                    "fastembed_model", "BAAI/bge-small-en-v1.5"
                )
                self.fast_embedder = TextEmbedding(model_name=model_name)
                self.fastembed_model = model_name
                logger.info(f"✅ FastEmbed ready (model: {model_name})")
            except Exception as e:
                logger.warning(f"⚠️ FastEmbed init failed: {e}")
                self.fast_embedder = None

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

    async def set_mode(self, mode: str) -> None:
        async with self._mode_lock:
            if mode not in ("chat", "agentic", "investigation"):
                mode = "chat"
            old = self.mode
            self.mode = mode
            logger.info(f"ContextManager mode changed: {old} → {mode}")

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
        if self.mode == "chat" and len(text) < 200:
            logger.debug("Chat mode: skipping embedding for short query")
            return np.array([])
        if not text.strip():
            return np.array([])

        prefix_str = prefix or "default"
        cache_key = hashlib.md5(f"{prefix_str}:{text}".encode("utf-8")).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if self.fast_embedder is not None:
            input_text = f"{prefix}: {text}" if prefix else text

            def _embed_sync():
                try:
                    emb_list = list(self.fast_embedder.embed([input_text]))  # type: ignore[attr-defined]
                    return np.asarray(emb_list[0], dtype=np.float32)
                except Exception:
                    logger.error("FastEmbed failed", exc_info=True)
                    return np.array([])

            emb = await asyncio.to_thread(_embed_sync)
        elif self.embeddings is not None:
            emb = await self.embeddings.fetch_embedding(text, size)
        else:
            emb = np.array([])

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
        if not content or not content.strip():
            return None

        metadata = metadata or {}
        path_key = metadata.get("path") or metadata.get("filename")

        if path_key and path_key in self.files_checked:
            return path_key

        parent_key = path_key or f"{source_type}_{hash(content)}"
        chunk_items = self._chunk_text(content, parent_key, source_type)

        added_keys = []
        for item in chunk_items:
            if item.key in self.knowledge_base:
                continue
            item.embedding = await self.fetch_embedding(item.content, prefix="passage")
            self.knowledge_base[item.key] = item
            added_keys.append(item.key)

        if path_key:
            self.files_checked.add(str(path_key))

        self._log_action(
            "add_knowledge",
            f"Added {source_type} → {parent_key} ({len(chunk_items)} chunks)",
            metadata,
        )
        self.context_stats["kb_added"] += len(added_keys)

        # Mark for lazy async rebuild
        self._sparse_index_dirty = True

        logger.info(f"Added {len(added_keys)} chunks for {parent_key}")
        return parent_key

    # ─────────────────────────────────────────────────────────────
    # SPARSE INDEX (TF-IDF) – fully async + lazy
    # ─────────────────────────────────────────────────────────────
    def _rebuild_sparse_index_sync(self):
        """Synchronous heavy rebuild – called only from thread."""
        if len(self.knowledge_base) < 3:
            return
        texts = [item.content for item in self.knowledge_base.values()]
        self.kb_keys_ordered = list(self.knowledge_base.keys())
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_df=0.85,
                min_df=2,
                ngram_range=(1, 2),
            )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # Safe shape access that satisfies strict type checkers
        shape = getattr(self.tfidf_matrix, "shape", None)
        if shape is not None:
            logger.debug(f"Rebuilt TF-IDF sparse matrix: {shape}")

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

    async def _sparse_retrieve(self, query: str) -> Dict[str, float]:
        """Async sparse retrieval."""
        await self._ensure_sparse_index()
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            return {}

        def _sparse_sync():
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                return {}
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = (self.tfidf_matrix * query_vec.transpose()).toarray().flatten()  # type: ignore[attr-defined]
            result = {}
            for idx, score in enumerate(scores):
                if score > 0 and idx < len(self.kb_keys_ordered):
                    result[self.kb_keys_ordered[idx]] = float(score)
            return result

        return await asyncio.to_thread(_sparse_sync)

    # ─────────────────────────────────────────────────────────────
    # KNOWLEDGE RETRIEVAL (now fully async)
    # ─────────────────────────────────────────────────────────────
    async def _retrieve_relevant_knowledge(self, query: str, max_kb_tokens: int) -> str:
        if not self.knowledge_base or not query.strip():
            return ""
        query_emb = await self.fetch_embedding(query, prefix="query")
        sparse_scores = await self._sparse_retrieve(query)

        dense_ranked = []
        if query_emb.size > 0:
            for key, item in self.knowledge_base.items():
                if item.embedding.size > 0:
                    sim = cosine_sim(query_emb, item.embedding)
                    dense_ranked.append((sim, key))
            dense_ranked.sort(reverse=True)
            dense_ranked = dense_ranked[:50]

        sparse_ranked = sorted(
            [(score, key) for key, score in sparse_scores.items() if score > 0],
            reverse=True,
        )[:50]

        rrf_scores: Dict[str, float] = {}
        k = 60
        for rank, (_, key) in enumerate(dense_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        for rank, (_, key) in enumerate(sparse_ranked, start=1):
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)

        scored = sorted(
            [
                (rrf_scores.get(key, 0.0), self.knowledge_base[key])
                for key in rrf_scores
            ],
            reverse=True,
        )
        parts: List[str] = []
        used = 0
        for _, item in scored[:40]:
            meta_str = ", ".join(f"{k}={v}" for k, v in item.metadata.items() if v)
            header = f"[{item.source_type.upper()}] {item.key}\nMetadata: {meta_str}\n"
            block = f"{header}{item.content}\n{'─' * 50}\n"
            block_tokens = (
                self.tokenizer.count_tokens(block)
                if self.tokenizer
                else len(block) // 4
            )
            if used + block_tokens > max_kb_tokens:
                break
            parts.append(block)
            used += block_tokens
        return "".join(parts)

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
    # (100% unchanged from your original)
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
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if chat or self.mode == "chat":
            clean_system = system_prompt.strip()
            messages.append({"role": "system", "content": clean_system})

            user_content = query.strip()
            if any(
                phrase in user_content.upper()
                for phrase in ["CONTINUATION", "AFTER TOOL", "AFTER_TOOL"]
            ):
                user_content = "Please give the clean final answer based on the tool results you just received."

            messages.append(
                {
                    "role": "user",
                    "content": user_content or "[AUTONOMOUS CONTINUATION]",
                }
            )

            total_tokens = self.count_tokens(messages)
            if total_tokens > max_input_tokens - reserved_for_output:
                self.prune_to_fit_context(
                    messages,
                    max_tokens=max_input_tokens - reserved_for_output,
                    min_keep_messages=3,
                    system_role="system",
                    user_role="user",
                    assistant_role="assistant",
                    tool_role="tool",
                )
            return messages

        base_system = system_prompt.strip()
        summary = self.get_context_summary()
        full_system = base_system
        if summary:
            full_system += f"\n\n{summary}\n\n→ Use this context to stay aware. Never repeat the summary back to the user."

        if include_logs:
            try:
                log_lines = Logger.get_log_data(level="info", max_entries=100)
                if log_lines:
                    log_text = "\n".join(log_lines)
                    full_system += (
                        f"\n\n=== RECENT LOGS (INFO, last {len(log_lines)} lines) ===\n"
                        f"{log_text}\n=== END LOGS ==="
                    )
            except Exception:
                pass

        messages.append({"role": "system", "content": full_system})
        tokens_used = self.count_tokens(messages)

        query_tokens = (
            self.count_tokens([{"role": "user", "content": query}])
            if query.strip()
            else 0
        )

        target_context_tokens = max_input_tokens - reserved_for_output
        remaining_tokens = target_context_tokens - tokens_used - query_tokens
        if remaining_tokens < 0:
            remaining_tokens = 0

        kb_tokens = min(max_kb_tokens, remaining_tokens // 2)
        history_tokens_budget = max(remaining_tokens - kb_tokens, min_history_tokens)

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
                remaining_tokens = target_context_tokens - tokens_used - query_tokens
                history_tokens_budget = max(remaining_tokens, min_history_tokens)

        recent_msgs: List[Dict[str, str]] = []
        for msg, t in zip(
            reversed(self.current_topic.history),
            reversed(self.current_topic.history_tokens),
        ):
            if t > history_tokens_budget:
                break
            recent_msgs.insert(0, msg)
            history_tokens_budget -= t
            tokens_used += t
        messages.extend(recent_msgs)

        messages.append(
            {
                "role": "user",
                "content": query if query.strip() else "[AUTONOMOUS CONTINUATION]",
            }
        )

        total_tokens = self.count_tokens(messages)
        if total_tokens > target_context_tokens:
            removed, pruned_turns = self.prune_to_fit_context(
                messages,
                max_tokens=target_context_tokens,
                min_keep_messages=5,
                system_role="system",
                user_role="user",
                assistant_role="assistant",
                tool_role="tool",
            )
            if pruned_turns:
                self.current_topic.archived_history.extend(pruned_turns)

        messages.append(
            {
                "role": "system",
                "content": (
                    f"⚠️ CONTEXT WINDOW STATUS: max tokens = {self.max_tokens}, "
                    f"current estimated usage = {tokens_used}. "
                    "If you detect the context is too long, summarize or prune older content."
                ),
            }
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

    async def record_tool_outcome(self, tool_name: str, result: str, metadata: dict):
        summary = result[:800] + ("..." if len(result) > 800 else "")
        if "not found" in result.lower() or "no such" in result.lower():
            self.fs_state["negative_findings"].append(
                (tool_name, metadata.get("path", ""))
            )
        await self.add_external_content(
            source_type="tool_outcome",
            content=f"Tool: {tool_name}\nResult: {summary}\nMetadata: {metadata}",
            metadata={
                "tool": tool_name,
                **metadata,
                "negative": "not found" in result.lower(),
            },
        )
        text = result.lower()
        if any(k in text for k in ["error", "exception", "failed"]):
            self.add_unknown(f"{tool_name} issue: {summary[:200]}")
        if "def " in result or "class " in result:
            self.add_finding(f"Code structure found via {tool_name}")
        if "not found" in text:
            self.add_unknown(f"{metadata.get('path', 'unknown')} not found")
        if metadata.get("path"):
            self.complete_subtask(f"inspect {metadata['path']}")
        self.tools_executed.append(tool_name)
        self.files_checked.update(
            m.get("path", "") for m in [metadata] if m.get("path")
        )
        self._log_action(
            "tool_outcome", f"Tool {tool_name} → {summary[:80]}...", metadata
        )
