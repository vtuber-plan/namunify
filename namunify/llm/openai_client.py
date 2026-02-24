"""OpenAI LLM client implementation."""

import asyncio
import random
from typing import Any, Optional

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

        max_attempts = 6
        response = None
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc) or attempt >= max_attempts:
                    raise
                delay_seconds = self._compute_retry_delay_seconds(exc, attempt)
                await asyncio.sleep(delay_seconds)

        if response is None:
            raise ValueError(f"OpenAI-compatible API call failed after retries: {last_error}")

        content = self._extract_content_from_response(response)
        if content is None or not content.strip():
            error_detail = self._describe_unusable_response(response)
            raise ValueError(
                "OpenAI-compatible API returned no usable content. "
                f"response_type={type(response).__name__}. {error_detail}"
            )
        return content

    @staticmethod
    def _extract_content_from_response(response: Any) -> Optional[str]:
        """Extract text content from a variety of OpenAI-compatible response formats."""
        if response is None:
            return None

        # Standard Chat Completions format.
        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                message_content = getattr(message, "content", None)
                extracted = OpenAIClient._normalize_message_content(message_content)
                if extracted:
                    return extracted

            # Some compatible providers may put text directly on choice.
            direct_text = getattr(first_choice, "text", None)
            if isinstance(direct_text, str) and direct_text.strip():
                return direct_text

        # Fallback: OpenAI "responses" style attribute.
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        return None

    @staticmethod
    def _normalize_message_content(content: Any) -> Optional[str]:
        """Normalize message content that may be either string or structured blocks."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value:
                        text_parts.append(text_value)
                        continue
                    nested_text = item.get("content")
                    if isinstance(nested_text, str) and nested_text:
                        text_parts.append(nested_text)
                        continue
                else:
                    text_value = getattr(item, "text", None)
                    if isinstance(text_value, str) and text_value:
                        text_parts.append(text_value)
                        continue

            if text_parts:
                return "".join(text_parts)

        return None

    @staticmethod
    def _describe_unusable_response(response: Any) -> str:
        """Build an actionable diagnostic string from an unusable response."""
        try:
            if hasattr(response, "model_dump"):
                dumped = response.model_dump()
                status = dumped.get("status")
                msg = dumped.get("msg")
                body = dumped.get("body")
                details = []
                if status is not None:
                    details.append(f"status={status}")
                if isinstance(msg, str) and msg.strip():
                    details.append(f"msg={msg.strip()}")
                if body is not None:
                    details.append(f"body={body}")
                if details:
                    return "provider_details: " + ", ".join(details)
        except Exception:
            pass

        return "provider did not include parseable error details"

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Whether an OpenAI-compatible request error should be retried."""
        message = str(exc).lower()
        indicators = (
            "error code: 429",
            "status code: 429",
            "too many requests",
            "throttling",
            "rate limit",
            "tpm",
            "rpm",
            "timeout",
            "timed out",
            "connection error",
            "temporarily unavailable",
            "service unavailable",
            "try again later",
        )
        return any(indicator in message for indicator in indicators)

    @staticmethod
    def _compute_retry_delay_seconds(exc: Exception, attempt: int) -> float:
        """Compute exponential backoff with jitter for retries."""
        # Honor provider hint if available.
        retry_after = OpenAIClient._extract_retry_after_seconds(exc)
        if retry_after is not None:
            return min(120.0, max(1.0, float(retry_after)))

        # 1.5, 3, 6, 12, 24, 48 (+ jitter)
        base = min(60.0, 1.5 * (2 ** (attempt - 1)))
        jitter = random.uniform(0, 0.8)
        return base + jitter

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
        """Try reading Retry-After from SDK exception response headers."""
        response = getattr(exc, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()
