"""Base LLM client interface."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the LLM.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt

        Returns:
            The LLM's response text
        """
        pass

    @staticmethod
    def extract_json_from_response(response: str) -> dict[str, str]:
        """Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Try to extract from markdown code block first
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")

    async def rename_symbols(
        self,
        context: str,
        symbols: list[str],
        snippets: Optional[dict[str, str]] = None,
        symbol_lines: Optional[dict[str, int]] = None,
    ) -> dict[str, str]:
        """Rename obfuscated symbols using the LLM.

        Args:
            context: Full code context with line numbers
            symbols: List of symbols to rename
            snippets: Optional dict mapping symbols to their usage snippets
            symbol_lines: Optional dict mapping symbols to their line numbers

        Returns:
            Dict mapping old names (or "name:line" format) to new names
        """
        from namunify.config import PROMPTS

        if len(symbols) == 1:
            symbol = symbols[0]
            snippet = snippets.get(symbol, "") if snippets else ""
            line_info = f" (defined at line {symbol_lines.get(symbol)})" if symbol_lines and symbol in symbol_lines else ""
            prompt = PROMPTS["rename_single_symbol"].format(
                context=context,
                symbol=symbol + line_info,
                snippet=snippet,
            )
        else:
            # Build symbols list with line numbers
            symbols_list = []
            for s in symbols:
                if symbol_lines and s in symbol_lines:
                    symbols_list.append(f"- {s} (line {symbol_lines[s]})")
                else:
                    symbols_list.append(f"- {s}")
            symbols_str = "\n".join(symbols_list)

            prompt = PROMPTS["rename_symbols"].format(
                context=context,
                symbols=symbols_str,
            )

        response = await self.complete(prompt)
        return self.extract_json_from_response(response)

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass
