"""Anthropic LLM client implementation."""

from typing import Optional

from anthropic import AsyncAnthropic

from namunify.llm.base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6-20250514",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        super().__init__(api_key, model, base_url, max_tokens, temperature)
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to Anthropic.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt

        Returns:
            The LLM's response text
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)

        # Extract text from response
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        if not text_content:
            raise ValueError("Anthropic returned empty response")
        return text_content

    async def close(self) -> None:
        """Close the Anthropic client."""
        await self._client.close()
