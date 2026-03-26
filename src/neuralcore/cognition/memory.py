import re
import asyncio
import time
import numpy as np
from typing import Tuple, List, Dict, Any
from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder as PromptHelper
from neuralcore.utils.config import get_loader
from neuralcore.core.client_factory import get_clients
from neuralcore.utils.text_tokenizer import TextTokenizer
from neuralcore.utils.search import keyword_score, cosine_sim

try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    FASTEMBED_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MSG_THR = 0.5
NUM_MSG = 5
OFF_THR = 0.7
OFF_FREQ = 4
SLICE_SIZE = 4

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
        self.embeddings = clients.get(
            "embeddings"
        )  # OG llama.cpp client (kept for fallback)

        # ── FastEmbed config (configurable + fully backward-compatible) ──
        loader = get_loader()
        embed_config = loader.get_client_config("embeddings") or {}
        self.use_fastembed = embed_config.get(
            "use_fastembed", True
        )  # ← default = new lightweight path
        self.fast_embedder = None
        self.fastembed_model = None

        if self.use_fastembed and TextEmbedding:
            try:
                model_name = embed_config.get(
                    "fastembed_model", "BAAI/bge-small-en-v1.5"
                )
                self.fast_embedder = TextEmbedding(model_name=model_name)
                self.fastembed_model = model_name
                logger.info(
                    f"✅ FastEmbed ready (model: {model_name}) – lightweight alternative to llama.cpp"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ FastEmbed init failed, falling back to OG embeddings: {e}"
                )
                self.fast_embedder = None
        else:
            if not self.use_fastembed:
                logger.info(
                    "Using original llama.cpp / OpenAI-style embeddings (use_fastembed=false)"
                )

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

    class TaskContext:
        """Lightweight per-task / per-sub-agent memory.
        Only keeps distilled, important information (files + results).
        Never stores full LLM responses or chat history."""

        def __init__(self, name: str, parent: "ContextManager"):
            self.name = name
            self.parent = parent
            self.important_files: set[str] = set()
            self.key_results: List[Dict[str, Any]] = []  # distilled summaries only
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
            """Distill result and auto-promote to shared Knowledge Base."""
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

            # Auto-promote important result to shared KB (the magic)
            await self.parent.add_external_content(
                source_type=f"task_result_{self.name}",
                content=f"[{title}] {summary}\nMetadata: {metadata or {}}",
                metadata={"task": self.name, **(metadata or {})},
            )

            self.parent._log_action(
                "task_result", f"Task {self.name} → {title}", metadata
            )

        def add_important_file(self, filepath: str):
            """Track files that matter for this task"""
            self.important_files.add(filepath)
            self.parent.files_checked.add(filepath)  # also update global

        def add_finding(self, finding: str):
            """Add finding and sync with global investigation state"""
            cleaned = finding[:400]
            self.findings.append(cleaned)
            self.parent.add_finding(cleaned)

        def add_hypothesis(self, hypothesis: str):
            self.hypotheses.append(hypothesis[:300])

        def get_context(self, max_tokens: int = 3500) -> str:
            """Return clean, focused context for this specific task/sub-agent"""
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
            # Rough token limiting (can be improved with tokenizer later)
            return text[: max_tokens * 4]

    # ── Task Management Methods ──
    def create_task_context(self, task_name: str) -> "ContextManager.TaskContext":
        """Create or get a lightweight task-specific context."""
        if not hasattr(self, "_task_contexts"):
            self._task_contexts: Dict[str, ContextManager.TaskContext] = {}

        if task_name not in self._task_contexts:
            self._task_contexts[task_name] = self.TaskContext(task_name, self)
            self.add_subtask(task_name)  # track in global investigation_state
            logger.info(f"Created new task context: {task_name}")

        return self._task_contexts[task_name]

    def get_task_context(self, task_name: str) -> "ContextManager.TaskContext | None":
        """Retrieve existing task context."""
        return getattr(self, "_task_contexts", {}).get(task_name)

    def list_active_tasks(self) -> List[str]:
        """Helpful for context summary."""
        return list(getattr(self, "_task_contexts", {}).keys())

    # ─────────────────────────────────────────────────────────────
    # FETCH EMBEDDING
    # ─────────────────────────────────────────────────────────────

    async def fetch_embedding(
        self, text: str, size: int = 500, prefix: str | None = None
    ) -> np.ndarray:
        """Unified embedding fetch with detailed logging."""

        if not text.strip():
            logger.info("Embedding skipped: empty input text")
            return np.array([])

        prefix_str = prefix or "default"
        cache_key = f"{prefix_str}:{text}"

        logger.debug(f"Embedding request | prefix={prefix_str} | text_len={len(text)}")

        if cache_key in self.embedding_cache:
            logger.info(f"Cache hit | prefix={prefix_str} | text_len={len(text)}")
            return self.embedding_cache[cache_key]

        logger.info(
            f"Cache miss | prefix={prefix_str} | using={'fastembed' if self.fast_embedder else 'fallback'}"
        )

        # ── FastEmbed path ──
        if self.fast_embedder:
            input_text = f"{prefix}: {text}" if prefix else text

            logger.debug(
                f"FastEmbed input prepared | prefix_applied={bool(prefix)} | input_len={len(input_text)}"
            )

            def _embed_sync():
                if self.fast_embedder:
                    try:
                        logger.debug("FastEmbed inference started")

                        emb_list = list(self.fast_embedder.embed([input_text]))
                        emb = np.asarray(emb_list[0], dtype=np.float32)

                        logger.info(
                            f"FastEmbed success | dim={emb.shape[0]} | dtype={emb.dtype}"
                        )

                        return emb

                    except Exception:
                        logger.error(
                            f"FastEmbed failed | prefix={prefix_str} | text_sample={text[:100]}",
                            exc_info=True,
                        )
                        return np.array([])

            emb = await asyncio.to_thread(_embed_sync)

        # ── Fallback path ──
        elif self.embeddings:
            logger.info("Using fallback embedding provider (llama.cpp / OpenAI-style)")
            emb = await self.embeddings.fetch_embedding(text, size)

        else:
            logger.warning("No embedding provider available")
            emb = np.array([])

        # ── Validation ──
        if emb is not None and emb.size > 0:
            if not np.isfinite(emb).all():
                logger.warning("Embedding contains non-finite values (NaN/Inf)")
                return np.array([])

            self.embedding_cache[cache_key] = emb

            logger.info(f"Embedding cached | prefix={prefix_str} | dim={emb.shape[0]}")

            return emb

        logger.warning(
            f"Embedding failed validation | prefix={prefix_str} | text_len={len(text)}"
        )

        return np.array([])

    # ─────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────
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

    def get_last_user_message(self) -> str | None:
        """
        Return the content of the last message with role 'user' from current topic history.
        Returns None if no user message is found.
        """
        for msg in reversed(self.current_topic.history):
            if msg.get("role") == "user":
                return msg.get("content", "").strip()
        return None

    def get_context_summary(self, max_messages: int = 10, max_chars: int = 1000) -> str:
        """
        Fish memory-style context summary:
        - Full list of files and tools used
        - KB items and messages in topic
        - Token usage warning
        - Only include up to `max_messages` recent messages
        - Truncates to max_chars if necessary
        """

        # # Sub-agent can use clean focused context:
        # task_ctx_str = auth_task.get_context(max_tokens=3000)
        # Full files and tools
        files = sorted(list(self.files_checked))
        tools = self.tools_executed

        # Messages in topic (most recent up to max_messages)
        recent_messages = self.current_topic.history[-max_messages:]
        message_summaries = [
            f"[{msg['role'].upper()}] {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}"
            for msg in recent_messages
        ]

        # Token usage estimate
        total_tokens = (
            sum(self.current_topic.history_tokens)
            if self.current_topic.history_tokens
            else 0
        )
        token_warning = ""
        if total_tokens > self.max_tokens * 0.8:
            token_warning = (
                "⚠️ Approaching context window limit, consider pruning old messages."
            )

        # Build summary lines
        lines = [
            "📋 LIVE CONTEXT SUMMARY",
            f"• Files checked: {', '.join(files) if files else '—'}",
            f"• Tools used: {', '.join(tools) if tools else '—'}",
            f"• KB items: {len(self.knowledge_base)} | Messages in topic: {len(self.current_topic.history)} (showing last {len(recent_messages)})",
            f"• Estimated tokens used: {total_tokens} / {self.max_tokens} {token_warning}",
            f"• Active topic: {self.current_topic.name}",
            f"• Archived: {len(self.current_topic.archived_history)} | Prunes: {self.context_stats['prunes']}",
            f"• Recent action: {self.action_log[-1]['desc'] if self.action_log else '—'}",
            "",
            "📝 LAST MESSAGES",
            *message_summaries,
        ]

        # Investigation state
        inv = self.investigation_state
        inv_block = [
            "",
            "🧠 INVESTIGATION STATE",
            f"• Goal: {inv['goal'] or '—'}",
            f"• Pending: {', '.join(inv['pending'][:5]) if inv['pending'] else '—'}",
            f"• Completed: {', '.join(inv['completed'][-5:]) if inv['completed'] else '—'}",
            f"• Findings: {', '.join(inv['findings'][-3:]) if inv['findings'] else '—'}",
            f"• Unknowns: {', '.join(inv['unknowns'][-3:]) if inv['unknowns'] else '—'}",
        ]
        lines.extend(inv_block)

        # Join and truncate if over max_chars
        full_summary = "\n".join(lines)
        if len(full_summary) > max_chars:
            cutoff = max_chars - 50
            summary_lines = full_summary[:cutoff].rsplit("\n", 1)[0]
            summary_lines += "\n…(truncated)"
            return summary_lines

        return full_summary

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
    # ADD EXTERNAL CONTENT
    # ─────────────────────────────────────────────────────────────
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Return text truncated to approximately max_tokens using the tokenizer."""
        if not self.tokenizer:
            return text[: max_tokens * 4]  # fallback heuristic
        tokens_counted = 0
        words = text.split()
        result_words = []
        for word in words:
            word_tokens = self.tokenizer.count_tokens(word)
            if tokens_counted + word_tokens > max_tokens:
                break
            result_words.append(word)
            tokens_counted += word_tokens
        return " ".join(result_words)

    async def add_external_content(
        self, source_type: str, content: str, metadata: Dict[str, Any] | None = None
    ) -> str | None:
        if not content or not content.strip():
            return None

        metadata = metadata or {}
        path_key = metadata.get("path") or metadata.get("filename")

        # Skip if file already exists in KB
        if path_key and path_key in self.files_checked:
            logger.info(f"Skipped adding external content; already exists: {path_key}")
            return path_key  # optionally return existing key

        # Generate key (fallback to content hash)
        key = path_key or f"{source_type}_{hash(content)}"

        # Avoid overwriting if key already exists
        if key in self.knowledge_base:
            logger.info(f"Skipped adding KB item; key already exists: {key}")
            return key

        # Create new knowledge item with full content
        embedding = await self.fetch_embedding(content, prefix="passage")
        item = KnowledgeItem(key, source_type, content, metadata)
        item.embedding = embedding
        self.knowledge_base[key] = item
        logger.info(f"Added KB item: {key}")

        # Track known files
        if path_key:
            self.files_checked.add(str(path_key))

        self._log_action("add_knowledge", f"Added {source_type} → {key}", metadata)
        self.context_stats["kb_added"] += 1
        return key

    # ─────────────────────────────────────────────────────────────
    # RETRIEVE RELEVANT KNOWLEDGE
    # ─────────────────────────────────────────────────────────────
    async def _retrieve_relevant_knowledge(self, query: str, max_kb_tokens: int) -> str:
        if not self.knowledge_base or not query.strip():
            return ""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded...")

        query_emb = await self.fetch_embedding(query, prefix="query")
        if len(query_emb) == 0:
            return ""
        query_words = re.findall(r"\b\w+\b", query.lower())

        scored = []
        for item in self.knowledge_base.values():
            if item.embedding.size == 0:
                continue
            emb_sim = cosine_sim(query_emb, item.embedding)
            preview = (
                " ".join(str(v) for v in item.metadata.values())
                + " "
                + item.content[:400]  # keep preview small for keyword scoring
            )
            kw = keyword_score(query_words, preview)
            hybrid = 0.65 * emb_sim + 0.35 * kw
            scored.append((hybrid, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        parts: List[str] = []
        used = 0

        for _, item in scored[:40]:
            meta_str = ", ".join(f"{k}={v}" for k, v in item.metadata.items() if v)
            header = f"[{item.source_type.upper()}] {item.key}\nMetadata: {meta_str}\n"
            block = f"{header}{item.content}\n{'─' * 50}\n"

            # calculate tokens dynamically
            block_tokens = (
                self.tokenizer.count_tokens(block)
                if hasattr(self.tokenizer, "count_tokens")
                else len(block) // 4
            )
            if used + block_tokens > max_kb_tokens:
                # optional: include partial block if it fits
                remaining_tokens = max_kb_tokens - used
                if remaining_tokens > 0 and hasattr(self.tokenizer, "truncate_text"):
                    truncated_content = self.truncate_to_tokens(
                        item.content, remaining_tokens
                    )
                    block = f"{header}{truncated_content}\n{'─' * 50}\n"
                    parts.append(block)

                break

            parts.append(block)
            used += block_tokens

        return "".join(parts)

    # ─────────────────────────────────────────────────────────────
    # ADD MESSAGE
    # ─────────────────────────────────────────────────────────────
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

        # Match topic first (optional topic switch)
        topic = await self._match_topic(embedding, exclude_topic=self.current_topic)
        if topic:
            await self.switch_topic(topic)

        await self.current_topic.add_message(role, message, embedding, token_count)
        asyncio.create_task(self._analyze_history())

        self._log_action(
            "add_message", f"{role} message ({len(message)} chars)", {"role": role}
        )

    # ─────────────────────────────────────────────────────────────
    # SWITCH TOPIC
    # ─────────────────────────────────────────────────────────────
    async def switch_topic(self, topic: Topic) -> None:
        async with asyncio.Lock():
            if topic.name != self.current_topic.name:
                if not any(t.name == self.current_topic.name for t in self.topics):
                    self.topics.append(self.current_topic)
                logger.info(f"Switched topic → {topic.name}")
                self.current_topic = topic
                self._log_action("switch_topic", f"Switched to {topic.name}")

    # ─────────────────────────────────────────────────────────────
    # MATCH TOPIC
    # ─────────────────────────────────────────────────────────────
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
    # TOKEN-AWARE HISTORY ANALYSIS & OFF-TOPIC DETECTION
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # MESSAGE RELEVANCE & OFF-TOPIC ANALYSIS
    # ─────────────────────────────────────────────────────────────
    async def _score_messages_by_relevance(
        self, messages: list, ref_emb: np.ndarray
    ) -> list:
        """Compute (score, message, embedding) tuples for relevance-aware pruning."""
        scored = []
        for msg in messages:
            emb = await self.fetch_embedding(msg["content"], prefix="passage")
            score = cosine_sim(emb, ref_emb) if len(emb) > 0 else 0.0
            scored.append((score, msg, emb))
        return scored

    async def _analyze_history(self) -> None:
        topic = self.current_topic
        if len(topic.history) <= 4:
            return  # Not enough messages to analyze

        # Ensure topic embedding
        if topic.embedded_description.size == 0 and topic.description.strip():
            topic.embedded_description = await self.fetch_embedding(
                topic.description, prefix="passage"
            )

        # Sliding window to detect off-topic segments
        window_size = min(len(topic.history), SLICE_SIZE * 3)
        window_msgs = topic.history[-window_size:]

        # Score messages by relevance
        scored_msgs = await self._score_messages_by_relevance(
            window_msgs, topic.embedded_description
        )

        # Identify off-topic messages: score below threshold
        off_topic_indices = [
            i for i, (s, _, _) in enumerate(scored_msgs) if s < OFF_THR
        ]
        if len(off_topic_indices) <= len(scored_msgs) / 2:
            return  # Mostly on-topic, nothing to move

        # Segment of off-topic messages
        off_start_idx = len(topic.history) - window_size + off_topic_indices[0]
        segment = topic.history[off_start_idx:]

        # Token counts for the segment
        segment_tokens = [
            self.tokenizer.count_tokens(m["content"])
            if self.tokenizer
            else len(m["content"]) // 4
            for m in segment
        ]

        # Generate new topic info
        name, desc = await self.generate_topic_info_from_history(segment)
        if not name or not desc:
            return

        emb = await self.fetch_embedding(desc)
        matched = await self._match_topic(emb, exclude_topic=topic)

        # Target topic (existing or new)
        target_topic = matched or Topic(name, desc)
        if not matched:
            target_topic.embedded_description = emb

        # Move messages to target topic with token_count
        segment_embeddings = await asyncio.gather(
            *(self.fetch_embedding(m["content"]) for m in segment)
        )
        for m, emb_m, tks in zip(segment, segment_embeddings, segment_tokens):
            await target_topic.add_message(
                m["role"], m["content"], emb_m, token_count=tks
            )

        # Switch to the new/matched topic
        await self.switch_topic(target_topic)

        # Remove moved messages from original topic
        topic.history = topic.history[:off_start_idx]

        # Archive segment
        topic.archived_history.extend(segment)
        logger.info(
            f"Off-topic segment moved → {target_topic.name} | {len(segment)} messages"
        )
        await self.current_topic.add_message(
            role="system",
            content=f"⚠️ {len(off_topic_indices)} off-topic messages detected. Consider pruning or archiving these to stay within the {self.max_tokens}-token context window.",
            embedding=np.array([]),
            token_count=0,
        )

    # ========================================================================
    # TOPIC EXTRACTION & OFF-TOPIC
    # ========================================================================
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

    # ========================================================================
    # PROVIDE CONTEXT – STRICT token limit + permanent archive
    # ========================================================================
    async def provide_context(
        self,
        query: str = "",
        max_input_tokens: int = 22000,
        reserved_for_output: int = 2048,
        system_prompt: str = "You are a helpful Terminal AI agent with full coding and filesystem support.",
        include_logs: bool = False,
        min_history_tokens: int = 2000,
        max_kb_tokens: int = 5000,
        # ── NEW ARGUMENT ─────────────────────────────────────
        chat: bool = False,  # ← If True → "normal chat mode" (minimal context)
        # ─────────────────────────────────────────────────────
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        # === CHAT MODE: Minimal context for normal conversations ===
        if chat:
            # Just use the base system prompt, no summary, no logs, no KB, no extra status
            clean_system = system_prompt.strip()
            messages.append({"role": "system", "content": clean_system})

            # Add only the current user query
            messages.append(
                {
                    "role": "user",
                    "content": query if query.strip() else "[AUTONOMOUS CONTINUATION]",
                }
            )

            # Optionally still respect the token limit (good practice)
            total_tokens = self.count_tokens(messages)
            if total_tokens > max_input_tokens - reserved_for_output:
                # Light pruning if needed
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

        # === NORMAL MODE (original behavior with full context) ===
        # --- SYSTEM PROMPT + SUMMARY ---
        base_system = system_prompt.strip()
        summary = self.get_context_summary()
        full_system = base_system
        if summary:
            full_system += f"\n\n{summary}\n\n→ Use this context to stay aware. Never repeat the summary back to the user."

        # --- OPTIONAL LOGS ---
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

        # --- QUERY TOKEN ESTIMATE ---
        query_tokens = (
            self.count_tokens([{"role": "user", "content": query}])
            if query.strip()
            else 0
        )

        # --- CALCULATE AVAILABLE TOKENS ---
        target_context_tokens = max_input_tokens - reserved_for_output
        remaining_tokens = target_context_tokens - tokens_used - query_tokens
        if remaining_tokens < 0:
            remaining_tokens = 0

        # --- ALLOCATE KB / HISTORY DYNAMICALLY ---
        kb_tokens = min(max_kb_tokens, remaining_tokens // 2)
        history_tokens_budget = max(remaining_tokens - kb_tokens, min_history_tokens)

        # --- RETRIEVE KB IF QUERY EXISTS ---
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

        # --- ADD RECENT HISTORY ---
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

        # --- ADD USER QUERY ---
        messages.append(
            {
                "role": "user",
                "content": query if query.strip() else "[AUTONOMOUS CONTINUATION]",
            }
        )

        # --- FINAL PRUNE IF STILL OVER ---
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

        # Context window status message (only in normal mode)
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

    # ========================================================================
    # PRUNE (now returns turns for archiving)
    # ========================================================================
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

        # Helper to count tokens
        def count_list(mes):
            if not self.tokenizer:
                return 0
            try:
                return self.tokenizer.count_message_tokens(mes)
            except Exception as e:
                logger.warning(f"Token count failed: {e}")
                return 0

        current = count_list(messages)
        if current <= max_tokens:
            return 0, []

        # Protect System and the Last User Message (usually the query)
        protected = 0
        for i, m in enumerate(messages):
            if m.get("role") == system_role:
                protected = i + 1
            elif m.get("role") == user_role and i == len(messages) - 1:
                protected = max(protected, i + 1)

        pruned_turns: List[Dict[str, str]] = []
        i = protected
        removed = 0

        # Safety Buffer: Add 256 tokens to target to handle BPE/Special tokens
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

            # If we dropped below but slightly over, remove one more turn to be safe
            if current > effective_max:
                # Remove the oldest assistant/tool turn found
                for j in range(turn_start - 1, turn_start - 10, -1):
                    if j >= 0:
                        if messages[j].get("role") in (assistant_role, tool_role):
                            pruned_turns.append(messages[j])
                            del messages[j]
                            removed += 1
                            current = count_list(messages)
                            i = j
                            break

        # ── LOGGING & STATS ───────────────────────────────────────────────────
        if removed > 0:
            self._log_action(
                "prune", f"Pruned {removed} turns", {"kept": len(messages)}
            )

        logger.info(f"Pruned {removed} messages from prompt, Final Count: {current}")
        return removed, pruned_turns

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.tokenizer:
            return 0
        try:
            return self.tokenizer.count_message_tokens(messages)
        except Exception as e:
            logger.warning(f"Token count failed: {e}")
            return 0

    # ========================================================================
    # ARCHIVED CONTEXT RETRIEVAL (type-safe, fixed)
    # ========================================================================
    async def get_archived_context(self, query: str, max_tokens: int = 4000) -> str:
        """Pull relevant old turns from archive if model says 'I don't remember'."""
        if not self.current_topic.archived_history or not query.strip():
            return ""

        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not loaded...")

        query_emb = await self.fetch_embedding(query, prefix="query")
        query_words = re.findall(r"\b\w+\b", query.lower())

        scored = []
        for msg in self.current_topic.archived_history:
            text = msg["content"]
            emb = await self.fetch_embedding(
                text, prefix="passage"
            )  # always safe + cached
            if len(emb) == 0:  # ← FIXED: no more str/None
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
        """Always called after every tool in the loop"""
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

        # Heuristic extraction (fast, no LLM needed)
        text = result.lower()

        if any(k in text for k in ["error", "exception", "failed"]):
            self.add_unknown(f"{tool_name} issue: {summary[:200]}")

        if "def " in result or "class " in result:
            self.add_finding(f"Code structure found via {tool_name}")

        if "not found" in text:
            self.add_unknown(f"{metadata.get('path', 'unknown')} not found")

        # Track progress
        if metadata.get("path"):
            self.complete_subtask(f"inspect {metadata['path']}")

        # ── LOGGING & STATS ───────────────────────────────────────────────────
        self.tools_executed.append(tool_name)
        self.files_checked.update(
            m.get("path", "") for m in [metadata] if m.get("path")
        )
        self._log_action(
            "tool_outcome", f"Tool {tool_name} → {summary[:80]}...", metadata
        )
