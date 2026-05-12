"""Unit tests for neuralcore.utils.formatting."""

import sys

from pathlib import Path
from neuralcore.utils.formatting import (
    _tokenize,
    map_type_to_json,
    safe_parse_json,
    safe_json_dumps,
    prepare_chat_messages,
    is_valid_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_special_chars(self):
        tokens = _tokenize("read_file write-file")
        assert "read_file" in tokens
        assert "write-file" in tokens

    def test_empty(self):
        assert _tokenize("") == []


class TestMapTypeToJson:
    def test_basic_types(self):
        assert map_type_to_json(str) == "string"
        assert map_type_to_json(int) == "integer"
        assert map_type_to_json(float) == "number"
        assert map_type_to_json(bool) == "boolean"
        assert map_type_to_json(list) == "array"
        assert map_type_to_json(dict) == "object"

    def test_missing_defaults_to_string(self):
        from inspect import _empty

        assert map_type_to_json(_empty) == "string"


class TestSafeParseJson:
    def test_valid_json(self):
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_embedded_json(self):
        result = safe_parse_json('Some text {"a": 1} more text')
        assert result is not None
        assert result["a"] == 1

    def test_no_json(self):
        result = safe_parse_json("no json here")
        assert result is None

    def test_list_json(self):
        result = safe_parse_json('[{"x": 10}]')
        assert result is not None
        assert result.get("__root__") == [{"x": 10}]


class TestSafeJsonDumps:
    def test_dict(self):
        assert '"key"' in safe_json_dumps({"key": "value"})

    def test_non_serializable(self):
        result = safe_json_dumps(object())
        assert isinstance(result, str)


class TestPrepareChatMessages:
    def test_from_string(self):
        result = prepare_chat_messages("Hello")
        assert isinstance(result, list)
        assert any(m["role"] == "user" for m in result)

    def test_from_list(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = prepare_chat_messages(msgs)
        assert result[0]["role"] == "user"

    def test_with_system_prompt(self):
        result = prepare_chat_messages("Hello", system_prompt="System msg")
        assert any(m["role"] == "system" for m in result)


class TestIsValidJson:
    def test_valid(self):
        assert is_valid_json('{"a": 1}') is True
        assert is_valid_json("[1, 2, 3]") is True

    def test_invalid(self):
        assert is_valid_json("not json") is False
        assert is_valid_json('{"a": }') is False
        assert is_valid_json("") is False
