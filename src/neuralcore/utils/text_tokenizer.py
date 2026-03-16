from typing import List
from tokenizers import Tokenizer


class TextTokenizer:
    def __init__(self, tokenizer_source: str):
        """
        Initialize the tokenizer.

        Args:
            tokenizer_source (str): Hugging Face repo ID or local path to tokenizer.json
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_source) if tokenizer_source.endswith(".json") else Tokenizer.from_pretrained(tokenizer_source)

    def split_text_into_chunks(
        self, text: str, max_tokens: int = 900, overlap: int = 100
    ) -> List[str]:
        """Split text into chunks of roughly max_tokens with overlap."""
        if not text.strip():
            return []

        tokens = self.tokenizer.encode(text).ids
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start_idx += max_tokens - overlap

        return chunks

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text).ids)

    def count_message_tokens(self, messages: List[dict]) -> int:
        """
        Count tokens in a list of messages (for chat models).
        Handles both string and multimodal content.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
        return total