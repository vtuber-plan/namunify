"""OpenAI LLM client implementation."""

from typing import Optional

from openai import AsyncOpenAI

from namunify.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        super().__init__(api_key, model, base_url, max_tokens, temperature)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to OpenAI.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt

        Returns:
            The LLM's response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned empty response")
        return content

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()
