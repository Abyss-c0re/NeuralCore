import re
import asyncio
import numpy as np
from neuralcore.utils.logger import Logger
from typing import Tuple, List, Dict, Any
from neuralcore.utils.prompt_builder import PromptBuilder as PromptHelper
from neuralcore.core.client import LLMClient
from neuralcore.utils.text_tokenizer import TextTokenizer


MSG_THR = 0.5  # Simularity threshold for history
CONT_THR = 0.6  # Simularity threshold for content such as files and terminal output
NUM_MSG = 5  # Number of messages submitted to the chatbot from history
OFF_THR = 0.7  # Off-topic threshold
OFF_FREQ = 4  # Off-topic checking frequency (messages)
SLICE_SIZE = 4  # Last N messages to analyze for off-topic


logger = Logger.get_logger()


def cosine_similarity(vec1, vec2):
    """
    Custom function to compute the cosine similarity between two vectors.
    Optimized for speed.
    """
    # Convert the vectors to numpy arrays if they are not already
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate squared norms of the vectors (avoids the use of np.linalg.norm)
    norm_vec1_sq = np.sum(vec1**2)
    norm_vec2_sq = np.sum(vec2**2)

    # If either vector is a zero vector, return 0.0
    if norm_vec1_sq == 0 or norm_vec2_sq == 0:
        return 0.0

    # Calculate and return the cosine similarity
    return dot_product / (np.sqrt(norm_vec1_sq) * np.sqrt(norm_vec2_sq))


class Topic:
    def __init__(self, name: str = "", description: str = "") -> None:
        self.name = name
        self.description = description
        self.embedded_description = np.array([])
        self.history: list[dict[str, str]] = []
        self.history_embeddings = []
        self.embedding_cache: dict[str, np.ndarray] = {}

    async def add_message(self, role: str, message: str, embedding: np.ndarray) -> None:
        self.history.append({"role": role, "content": message})
        self.history_embeddings.append(embedding)
        logger.info(f"Message added to: {self.name}")

    async def get_relevant_context(self, embedding: np.ndarray) -> tuple[float, int]:
        if not self.history_embeddings:
            logger.info("No history embeddings found. Returning empty context.")
            return 0.0, -1

        # Ensure embedding is a numpy array
        embedding = np.array(embedding)

        # Check if each embedding in history_embeddings is also a numpy array
        similarities = [
            cosine_similarity(np.array(embedding), np.array(history_emb))
            for history_emb in self.history_embeddings
        ]

        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        logger.debug(f"Best similarity score: {best_similarity} at index {best_index}")

        return best_similarity, best_index


class ContextManager:
    def __init__(
        self,
        client: LLMClient,
        tokenizer: TextTokenizer,
        # Instance with .generate_structure method (from test.file_utils or manager)
    ) -> None:
        """
        ContextManager.
        Uses two LLMClient instances:
            - client: for all embeddings (fetch_embedding)
            - client: for internal topic extraction queries (can be the same instance if desired)
        file_utils is passed explicitly for folder structure generation and _read_file.
        """
        self.client = client
        self.tokenizer = tokenizer

        self.similarity_threshold = MSG_THR
        self.topics: list[Topic] = []
        self.current_topic = Topic("Initial topic")
        self.embedding_cache: dict[str, np.ndarray] = {}

    async def add_message(
        self, role: str, message: str, embedding: np.ndarray | None = None
    ) -> None:
        if embedding is None or embedding.size == 0:
            embedding = await self.fetch_embedding(message)
        topic = await self._match_topic(embedding, exclude_topic=self.current_topic)
        if topic:
            await self.switch_topic(topic)

        await self.current_topic.add_message(role, message, embedding)
        asyncio.create_task(self._analyze_history())

    async def fetch_embedding(self, text: str) -> np.ndarray:
        """
        Uses the dedicated client (async).
        """
        async with asyncio.Lock():
            if text in self.embedding_cache:
                return self.embedding_cache[text]

        embedding = await self.client.fetch_embedding(text)
        if embedding is not None and embedding.size > 0:  # <- fix here
            self.embedding_cache[text] = embedding
            logger.debug(f"Extracted {len(embedding)} embeddings")
            return embedding
        else:
            return np.array([])

    async def switch_topic(self, topic: Topic) -> None:
        async with asyncio.Lock():
            if topic.name != self.current_topic.name:
                if not any(t.name == self.current_topic.name for t in self.topics):
                    self.topics.append(self.current_topic)
                logger.info(f"Switched to {topic.name}")
                self.current_topic = topic

    async def _match_topic(
        self, embedding: np.ndarray, exclude_topic: Topic | None = None
    ) -> Topic | None:
        if len(self.topics) == 0:
            logger.info("No topics available for matching. Returning None.")
            return None

        async def compute_similarity(topic: Topic) -> tuple[float, Topic]:
            # Ensure embedded_description is a numpy array
            if not isinstance(topic.embedded_description, np.ndarray):
                topic.embedded_description = np.array(topic.embedded_description)

            # Return 0 similarity if either embedding or embedded_description is empty
            if len(topic.embedded_description) == 0 or len(embedding) == 0:
                return 0.0, topic

            # Compute cosine similarity using your custom function
            similarity_result = cosine_similarity(embedding, topic.embedded_description)

            # Directly assign the similarity_result (which is a float) to similarity
            similarity = similarity_result

            logger.debug(
                f"Computed similarity {similarity:.4f} for topic '{topic.name}'"
            )
            return similarity, topic

        # Run compute_similarity in parallel for all topics excluding the given one
        tasks = [
            compute_similarity(topic) for topic in self.topics if topic != exclude_topic
        ]
        results = await asyncio.gather(*tasks)

        best_similarity = 0.0
        best_topic = None

        # Find the best matching topic based on similarity
        for similarity, topic in results:
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_topic = topic

        if best_topic:
            logger.info(
                f"Best matching topic: '{best_topic.name}' with similarity {best_similarity:.4f}"
            )
        else:
            logger.info("No suitable topic found.")
        return best_topic

    async def generate_prompt(
        self,
        query: str,
        num_messages: int = NUM_MSG,
        max_input_tokens: int = 22000,
        reserved_for_output: int = 2048,
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Now supports continuation turns: pass query="" to continue without adding a new user message.
        """
        # ── 0. Preparation ───────────────────────────────────────────────
        if query.strip():
            embedding = await self.fetch_embedding(query)
            # Topic switching only on real new user queries
            current_topic_match = await self._match_topic(embedding)
            if current_topic_match:
                await self.switch_topic(current_topic_match)
        else:
            embedding = None  # continuation turn

        # ── 1. Initialize result ─────────────────────────────────────────
        messages: List[Dict[str, Any]] = []
        total_tokens = 0

        def add_message(role: str, content: str) -> int:
            if not content.strip():
                return 0
            messages.append({"role": role, "content": content})
            return self.tokenizer.count_tokens(content) + 10

        # ... (system prompt, history selection unchanged) ...

        # ── 6. Current user query — ONLY if it's a real new query ────────
        if query.strip():
            user_content = query.strip()
            total_tokens += add_message("user", user_content)

        # Logging unchanged
        logger.info(
            f"generate_prompt → estimated tokens: {total_tokens:,} / {max_input_tokens:,} "
        )
        return messages

    async def generate_topic_info_from_history(
        self, history: list, max_retries: int = 3
    ) -> Tuple[str, str] | Tuple[None, None]:
        attempt = 0
        while attempt < max_retries:
            await asyncio.sleep(1)
            try:
                # === ADAPTED: use _query_llm instead of old self.helper ===
                response = await self.client.ask(PromptHelper.topics_helper(history))

                if not isinstance(response, str):
                    logger.warning(
                        f"Unexpected response type from helper LLM: {type(response)}"
                    )
                    attempt += 1
                    continue

                clean_response = re.sub(
                    r"^```(?:json)?|```$",
                    "",
                    response,
                    flags=re.IGNORECASE,
                ).strip()
                matches = re.findall(r':\s*"([^"]+)"', clean_response)
                extracted_topic_name = matches[0] if len(matches) > 0 else "unknown"
                extracted_topic_description = matches[1] if len(matches) > 1 else ""

                if extracted_topic_name and extracted_topic_description:
                    logger.info(f"Extracted topic: {extracted_topic_name}")
                    return extracted_topic_name, extracted_topic_description
                else:
                    logger.warning("Could not extract valid topic information.")
            except Exception as e:
                logger.error(
                    f"Analyze history attempt {attempt + 1} failed: {str(e)}",
                    exc_info=True,
                )
                attempt += 1
                if attempt < max_retries:
                    logger.info(f"Retrying... (Attempt {attempt + 1})")
                else:
                    logger.error("Max retries reached.")
                    break
        return None, None

    async def _analyze_history(
        self,
        off_topic_threshold: float = OFF_THR,
        off_topic_frequency: int = OFF_FREQ,
        slice_size: int = SLICE_SIZE,
    ) -> None:
        if (
            len(self.current_topic.history) > 4
            and not self.current_topic.description.strip()
        ):
            (
                new_topic_name,
                new_topic_desc,
            ) = await self.generate_topic_info_from_history(self.current_topic.history)
            if new_topic_name and new_topic_desc:
                self.current_topic.name = new_topic_name
                self.current_topic.description = new_topic_desc
                self.current_topic.embedded_description = await self.fetch_embedding(
                    new_topic_desc
                )
                return

        if (
            len(self.current_topic.history) >= off_topic_frequency
            and len(self.current_topic.history) % off_topic_frequency == 0
        ):
            logger.info("Analyzing current topic for potential off-topic segments.")

            current_name = self.current_topic.name
            candidate_slice = self.current_topic.history[-slice_size:]

            candidate_embeddings = await asyncio.gather(
                *(self.fetch_embedding(msg["content"]) for msg in candidate_slice)
            )

            similarities = [
                cosine_similarity(msg_emb, self.current_topic.embedded_description)
                for msg_emb in candidate_embeddings
            ]
            logger.info(f"Per-message similarities: {similarities}")

            if (
                sum(1 for s in similarities if s < off_topic_threshold)
                > len(similarities) / 2
            ):
                off_topic_start_index = len(self.current_topic.history) - slice_size
                for i, sim in enumerate(similarities):
                    if sim < off_topic_threshold:
                        off_topic_start_index = (
                            len(self.current_topic.history) - slice_size + i
                        )
                        break
                off_topic_segment = self.current_topic.history[off_topic_start_index:]

                (
                    candidate_topic_name,
                    candidate_topic_desc,
                ) = await self.generate_topic_info_from_history(off_topic_segment)
                if candidate_topic_name and candidate_topic_desc:
                    candidate_embedding = await self.fetch_embedding(
                        candidate_topic_desc
                    )
                    matched_topic = await self._match_topic(
                        candidate_embedding, exclude_topic=self.current_topic
                    )
                    if matched_topic is not None:
                        logger.info("Matched topic found")
                        for msg in off_topic_segment:
                            msg_emb = await self.fetch_embedding(msg["content"])
                            await matched_topic.add_message(
                                msg["role"], msg["content"], msg_emb
                            )
                        await self.switch_topic(matched_topic)
                    else:
                        logger.info("Creating new topic from off-topic content")
                        new_topic = Topic(candidate_topic_name, candidate_topic_desc)
                        new_topic.embedded_description = candidate_embedding
                        for msg in off_topic_segment:
                            msg_emb = await self.fetch_embedding(msg["content"])
                            await new_topic.add_message(
                                msg["role"], msg["content"], msg_emb
                            )
                        await self.switch_topic(new_topic)

                        # Truncate the old topic (same logic as original)
                        target_topic = next(
                            (t for t in self.topics if t.name == current_name), None
                        )
                        if target_topic:
                            async with asyncio.Lock():
                                target_topic.history = target_topic.history[
                                    :off_topic_start_index
                                ]
                else:
                    logger.warning("Could not generate candidate topic info.")
            else:
                logger.info("Candidate slice does not appear off-topic.")

        return

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Safe token counting wrapper — uses the attached tokenizer.
        Returns 0 if tokenizer is missing or fails.
        """
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return 0
        try:
            return self.tokenizer.count_message_tokens(messages)
        except Exception as e:
            logger.warning(f"Token counting failed in ContextManager: {e}")
            return 0

    def prune_to_fit_context(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        min_keep_messages: int = 5,
        system_role: str = "system",
        user_role: str = "user",
        assistant_role: str = "assistant",
        tool_role: str = "tool",
    ) -> int:
        """
        In-place pruning of the messages list:
        - Keeps system prompt (if present)
        - Keeps the most recent complete turns (assistant + following tool messages)
        - Removes oldest complete turns until under max_tokens
        - Never removes the very last user message or current assistant attempt
        - Returns number of messages removed

        Designed to preserve valid tool-calling structure.
        """
        if not messages or max_tokens <= 0:
            return 0

        current_tokens = self.count_tokens(messages)

        if current_tokens <= max_tokens:
            return 0  # already fits

        # Find protected prefix (system + very last user message if present)
        protected_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == system_role:
                protected_end = i + 1
            elif msg.get("role") == user_role and i == len(messages) - 1:
                protected_end = max(protected_end, i + 1)

        if protected_end >= len(messages):
            # Almost nothing to prune — just truncate last content if needed
            if messages and messages[-1].get("content"):
                content = messages[-1]["content"]
                half = max(200, max_tokens // 3)
                messages[-1]["content"] = content[:half] + " … [truncated]"
            return 0

        # Start pruning from after protected prefix
        i = protected_end
        removed = 0

        while current_tokens > max_tokens and i < len(messages) - min_keep_messages:
            # Look for the start of the next "turn" — usually an assistant message
            turn_start = i
            while i < len(messages) and messages[i].get("role") not in (
                assistant_role,
                user_role,
            ):
                i += 1

            if i >= len(messages):
                break

            turn_end = i
            # Find the end of this turn: assistant + all following tool messages
            while turn_end < len(messages) and messages[turn_end].get("role") in (
                assistant_role,
                tool_role,
            ):
                turn_end += 1

            if turn_end == turn_start:
                # No progress — safety break
                break

            # Remove this old turn
            del messages[turn_start:turn_end]
            removed += turn_end - turn_start

            # Recalculate tokens (unfortunately needed — could be optimized later)
            current_tokens = self.count_tokens(messages)
            i = turn_start  # continue from where we left

        logger.info(
            f"ContextManager pruned {removed} messages → "
            f"now {len(messages)} msgs, ≈{current_tokens} tokens (max was {max_tokens})"
        )

        return removed
