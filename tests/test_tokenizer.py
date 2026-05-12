"""Unit tests for neuralcore.utils.text_tokenizer -- TextTokenizer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TOKENIZER_PATH = str(PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json")


def _reset():
    from neuralcore.utils.text_tokenizer import TextTokenizer

    TextTokenizer._instance = None
    TextTokenizer._initialized = False
    import neuralcore.utils.config as cfg_mod

    cfg_mod.loader = None


class TestTextTokenizer:
    """Test TextTokenizer singleton and methods."""

    def test_init_from_file(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        assert tok.tokenizer is not None

    def test_singleton_pattern(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        t1 = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        t2 = TextTokenizer.get_instance()
        assert t1 is t2

    def test_count_tokens(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        count = tok.count_tokens("Hello world, this is a test.")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        assert tok.count_tokens("") == 0
        assert tok.count_tokens("   ") == 0

    def test_split_text_into_chunks(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        long_text = "word " * 2000
        chunks = tok.split_text_into_chunks(long_text, max_tokens=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_split_short_text_single_chunk(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        chunks = tok.split_text_into_chunks("Short text", max_tokens=500)
        assert len(chunks) == 1

    def test_split_empty_text(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        assert tok.split_text_into_chunks("") == []
        assert tok.split_text_into_chunks("   ") == []

    def test_count_message_tokens(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = tok.count_message_tokens(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_message_tokens_with_list_content(self):
        _reset()
        from neuralcore.utils.config import ConfigLoader

        ConfigLoader(
            cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
            app_root=PROJECT_ROOT,
        )
        from neuralcore.utils.text_tokenizer import TextTokenizer

        tok = TextTokenizer(tokenizer_source=TOKENIZER_PATH)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello world"}]},
        ]
        count = tok.count_message_tokens(messages)
        assert count > 0
