"""Tests for LLM client modules."""

import pytest

from namunify.llm.base import BaseLLMClient


class TestBaseLLMClient:
    """Tests for BaseLLMClient."""

    def test_extract_json_from_markdown_block(self):
        """Test extracting JSON from markdown code blocks."""
        response = '''Here is the result:
```json
{"a": "userName", "b": "processData"}
```
That's all.'''

        result = BaseLLMClient.extract_json_from_response(response)

        assert result == {"a": "userName", "b": "processData"}

    def test_extract_json_from_plain_block(self):
        """Test extracting JSON from plain code blocks."""
        response = '''Here is the result:
```
{"x": "value1", "y": "value2"}
```'''

        result = BaseLLMClient.extract_json_from_response(response)

        assert result == {"x": "value1", "y": "value2"}

    def test_extract_json_direct(self):
        """Test extracting JSON directly from response."""
        response = '{"foo": "bar", "baz": "qux"}'

        result = BaseLLMClient.extract_json_from_response(response)

        assert result == {"foo": "bar", "baz": "qux"}

    def test_extract_json_embedded(self):
        """Test extracting JSON embedded in text."""
        response = 'The result is {"a": 1, "b": 2} as you can see.'

        result = BaseLLMClient.extract_json_from_response(response)

        assert result == {"a": 1, "b": 2}

    def test_extract_json_with_line_numbers(self):
        """Test extracting JSON with line number keys."""
        response = '''```json
{
  "a:10": "tempValue",
  "b:25": "counter"
}
```'''

        result = BaseLLMClient.extract_json_from_response(response)

        assert "a:10" in result
        assert result["a:10"] == "tempValue"

    def test_extract_json_invalid(self):
        """Test error handling for invalid JSON."""
        response = "This is not JSON at all."

        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            BaseLLMClient.extract_json_from_response(response)

    def test_extract_json_nested(self):
        """Test extracting nested JSON."""
        response = '''```json
{
  "a": {
    "nested": "value"
  }
}
```'''

        result = BaseLLMClient.extract_json_from_response(response)

        assert result == {"a": {"nested": "value"}}


class TestOpenAIClient:
    """Tests for OpenAI client (mocked)."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test OpenAI client initialization."""
        from namunify.llm.openai_client import OpenAIClient

        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o",
        )

        assert client.api_key == "test-key"
        assert client.model == "gpt-4o"
        assert client.max_tokens == 4096

        await client.close()


class TestAnthropicClient:
    """Tests for Anthropic client (mocked)."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test Anthropic client initialization."""
        from namunify.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key="test-key",
            model="claude-sonnet-4-6-20250514",
        )

        assert client.api_key == "test-key"
        assert client.model == "claude-sonnet-4-6-20250514"

        await client.close()
