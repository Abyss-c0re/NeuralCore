import os
import re
import asyncio
import numpy as np
from datetime import datetime
from src.utils.logger import Logger
from typing import Tuple, Optional
from src.core.prompt_builder import PromptBuilder as PromptHelper
from src.utils.file_utils import _read_file
from src.core.client import LLMClient


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


class Project:
    def __init__(self, name: str = "") -> None:
        self.name: str = name
        self.file_embeddings: dict[str, dict] = {}
        self.folder_structure: dict = {}

    def _index_content(
        self,
        identifier: str,
        content: str,
        embedding: np.ndarray,
        content_type: str = "file",
    ) -> None:
        """
        Generic method to index any content (files or terminal outputs).
        """
        content_info = {
            "identifier": identifier,
            "content": content,
            "embedding": embedding,
            "type": content_type,
        }
        self.file_embeddings[identifier] = content_info
        logger.debug(
            f"Project '{self.name}': Added {content_type} content with id {identifier}"
        )

    def _index_file(self, file_path: str, content: str, embedding: np.ndarray) -> None:
        """Indexes a file's embedding."""
        self._index_content(file_path, content, embedding, content_type="file")

    def _index_terminal_output(
        self, output: str, identifier: str, embedding: np.ndarray
    ) -> None:
        """Indexes terminal output."""
        if not identifier:
            identifier = f"terminal_{datetime.now().isoformat()}"
        self._index_content(identifier, output, embedding, content_type="terminal")


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
        file_utils,  # Instance with .generate_structure method (from test.file_utils or manager)
    ) -> None:
        """
        ContextManager.
        Uses two LLMClient instances:
            - client: for all embeddings (fetch_embedding)
            - client: for internal topic extraction queries (can be the same instance if desired)
        file_utils is passed explicitly for folder structure generation and _read_file.
        """
        self.client = client
        self.file_utils = file_utils

        self.similarity_threshold = MSG_THR
        self.topics: list[Topic] = []
        self.current_topic = Topic("Initial topic")
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.projects: list[Project] = []
        self.current_project = Project("Unsorted")

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

    async def add_file(
        self, file_path: str, content: str, folder: bool = False
    ) -> None:
        new_project_name = os.path.basename(os.path.dirname(file_path))

        if self.current_project.name.lower() != new_project_name.lower():
            if (
                self.current_project.name.lower() != "unsorted"
                and self.current_project not in self.projects
            ):
                self.projects.append(self.current_project)
                logger.info(
                    f"Archived project '{self.current_project.name}' to projects list."
                )

            if not self.current_project.folder_structure and not folder:
                # === ADAPTED FOR NEW TUI ===
                # Old interactive yes_no_prompt removed (no manager.ui).
                # Automatically generate structure (non-interactive behaviour).
                logger.info(
                    "Automatically generating folder structure (new TUI - no interactive prompt)"
                )
                new_project = Project(new_project_name)
                try:
                    folder_path = os.path.dirname(file_path)
                    structure = self.file_utils.generate_structure(
                        folder_path, folder_path
                    )
                    new_project.folder_structure = structure
                    logger.info(
                        f"Generated new folder structure for project '{new_project_name}'."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate structure for project '{new_project_name}': {e}"
                    )

                self.current_project = new_project

        # Compute embedding for file path + content
        combined_content = f"Path: {file_path}\nContent: {content}"
        file_embedding = await self.fetch_embedding(combined_content)

        self.current_project._index_content(
            file_path, content, file_embedding, content_type="file"
        )

    async def add_terminal_output(
        self, command: str, output: str, summary: str
    ) -> None:
        terminal_content = f"Command: {command}\nOutput: {output}\nSummary: {summary}"
        terminal_embedding = await self.fetch_embedding(terminal_content)

        terminal_id = f"terminal_{hash(command + datetime.now().isoformat())}"
        self.current_project._index_content(
            terminal_id, terminal_content, terminal_embedding, content_type="terminal"
        )

        logger.info(f"Stored terminal output for command: {command}")

    def add_folder_structure(self, structure: dict) -> None:
        if self.current_project.folder_structure:
            if self.current_project not in self.projects:
                self.projects.append(self.current_project)
                logger.info(
                    f"Archived project '{self.current_project.name}' to projects list."
                )

        if (
            not self.current_project.name
            or self.current_project.name.lower() == "unsorted"
        ):
            if isinstance(structure, dict) and len(structure) == 1:
                new_name = list(structure.keys())[0]
                self.current_project.name = new_name
                logger.info(
                    f"Assigned new project name '{new_name}' from folder structure."
                )

        self.current_project.folder_structure = structure
        logger.info(
            f"Folder structure updated for project '{self.current_project.name}'."
        )

    def format_structure(self, folder_structure: dict) -> str:
        def format_substructure(substructure, indent=0):
            formatted = ""
            for key, value in substructure.items():
                if isinstance(value, dict):
                    formatted += " " * indent + f"{key}/\n"
                    formatted += format_substructure(value, indent + 4)
                else:
                    formatted += " " * indent + f"-- {value}\n"
            return formatted

        return format_substructure(folder_structure)

    def find_project_structure(self, query: str) -> Project | None:
        for project in self.projects:
            if project.name.lower() in query.lower():
                logger.info(f"Found project structure for '{project.name}' in query")
                return project
        logger.info("No matching project found in query")
        return None

    def extract_file_name_from_query(self, query: str) -> str | None:
        file_pattern = r"([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)*/[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)"
        match = re.search(file_pattern, query)
        return match.group(0) if match else None

    def extract_folder_from_query(self, query: str) -> str | None:
        folder_pattern = r"([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)+)"
        match = re.search(folder_pattern, query)
        if match:
            candidate = match.group(0)
            if not re.search(r"\.[a-zA-Z0-9]+$", candidate):
                return candidate
        return None

    async def get_relevant_content(
        self,
        query: str,
        content_type: Optional[str] = None,
        top_k: int = 1,
        similarity_threshold: float = CONT_THR,
    ) -> list | None:
        query_embedding = await self.fetch_embedding(query)
        query_embedding = np.array(query_embedding)  # Ensure it is a numpy array
        scores = []

        file_name = (
            self.extract_file_name_from_query(query)
            if content_type in (None, "file")
            else None
        )

        for identifier, info in self.current_project.file_embeddings.items():
            if content_type and info.get("type") != content_type:
                continue

            if file_name and info.get("type") == "file":
                if file_name.lower() in info.get("identifier", "").lower():
                    scores.append((identifier, 1.0))
                    logger.info(f"Added file '{identifier}' to context (Exact match).")
                    continue

            # Ensure that both embeddings are numpy arrays
            similarity = cosine_similarity(
                np.array(query_embedding), np.array(info["embedding"])
            )
            if similarity >= similarity_threshold:
                scores.append((identifier, similarity))
                logger.info(
                    f"Added file '{identifier}' to context (Similarity: {similarity})."
                )

        if not scores:
            logger.info(
                "No relevant content in current project; searching across all projects."
            )
            for project in self.projects:
                for identifier, info in project.file_embeddings.items():
                    if content_type and info.get("type") != content_type:
                        continue
                    similarity = cosine_similarity(
                        np.array(query_embedding), np.array(info["embedding"])
                    )
                    if similarity >= similarity_threshold:
                        scores.append((identifier, similarity))
                        logger.info(
                            f"Added file '{identifier}' from project '{project.name}' (Similarity: {similarity})."
                        )

        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            selected_ids = [id for id, _ in scores[:top_k]]
            results = []
            for id in selected_ids:
                if (
                    id in self.current_project.file_embeddings
                    and "content" in self.current_project.file_embeddings[id]
                ):
                    results.append(
                        (id, self.current_project.file_embeddings[id]["content"])
                    )
                else:
                    content = await _read_file(id)
                    results.append((id, content))
            return results

        logger.info("No matching content found.")
        return None

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

    async def generate_prompt(self, query: str, num_messages: int = NUM_MSG) -> list:
        embedding = await self.fetch_embedding(query)

        current_topic = await self._match_topic(embedding)
        if current_topic:
            await self.switch_topic(current_topic)

        project = self.find_project_structure(query)
        if project:
            self.current_project = project

        relevant_content = await self.get_relevant_content(query)
        content_references = ""

        if relevant_content:
            if self.current_project.folder_structure:
                content_references += f"Folder structure:\n{self.format_structure(self.current_project.folder_structure)}\n"
            for identifier, content in relevant_content:
                content_type = self.current_project.file_embeddings.get(
                    identifier, {}
                ).get("type", "content")
                label = (
                    "Referenced File"
                    if content_type == "file"
                    else "Referenced Terminal Output"
                    if content_type == "terminal"
                    else "Referenced Content"
                )
                content_references += f"\n[{label}: {identifier}]\n{content}\n"

        prompt = (
            f"{content_references}\nUser query: {query}"
            if content_references
            else query
        )

        await self.add_message("user", prompt, embedding)
        return self.current_topic.history[-num_messages:]

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
