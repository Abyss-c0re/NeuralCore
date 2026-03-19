from typing import List, Optional
from tokenizers import Tokenizer
from neuralcore.utils.config import get_loader


class TextTokenizer:
    _instance: Optional["TextTokenizer"] = None
    _initialized: bool = False

    _tokenizer_source: Optional[str]
    _client_name: Optional[str]

    def __new__(
        cls, tokenizer_source: Optional[str] = None, client_name: Optional[str] = None
    ):
        if cls._instance is None:
            loader = get_loader()

            client_name = client_name or "main"
            cfg = loader.get_client_config(client_name)

            resolved_source = tokenizer_source or cfg.get("tokenizer")

            if not resolved_source:
                raise ValueError(
                    f"Tokenizer source not provided and not found in config for client '{client_name}'"
                )

            instance = super().__new__(cls)

            # now type checker is happy
            instance._tokenizer_source = resolved_source
            instance._client_name = client_name

            cls._instance = instance

        return cls._instance

    def __init__(
        self, tokenizer_source: Optional[str] = None, client_name: Optional[str] = None
    ):
        if TextTokenizer._initialized:
            return

        tokenizer_source = getattr(self, "_tokenizer_source", None)

        if not tokenizer_source:
            raise ValueError("Tokenizer source missing during initialization")

        if tokenizer_source.endswith(".json"):
            self.tokenizer = Tokenizer.from_file(tokenizer_source)
        else:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_source)

        TextTokenizer._initialized = True

    @classmethod
    def get_instance(cls) -> "TextTokenizer":
        if cls._instance is None:
            raise ValueError("Tokenizer has not been initialized yet")
        return cls._instance

    # ----------------------
    # Methods
    # ----------------------

    def split_text_into_chunks(
        self, text: str, max_tokens: int = 500, overlap: int = 100
    ) -> List[str]:
        if not text or not text.strip():
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
        if not text or not text.strip():
            return 0
        return len(self.tokenizer.encode(text).ids)

    def count_message_tokens(self, messages: List[dict]) -> int:
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
