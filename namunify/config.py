"""Configuration management for namunify."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env from multiple locations
# 1. Current working directory
load_dotenv()
# 2. Project directory (where this package is installed)
_package_dir = Path(__file__).parent
load_dotenv(_package_dir.parent / ".env")
# 3. Home directory config
load_dotenv(Path.home() / ".config" / "namunify" / ".env")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Config(BaseSettings):
    """Configuration for namunify."""

    # LLM Settings
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")
    llm_model: str = Field(default="gpt-4o", description="Model name to use")
    llm_api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for API (for custom endpoints)")
    llm_max_tokens: int = Field(default=4096, description="Maximum tokens for LLM response")
    llm_temperature: float = Field(default=0.3, description="Temperature for LLM generation")
    llm_concurrency: int = Field(default=1, ge=1, description="Number of concurrent LLM rename requests")

    # Processing Settings
    max_context_size: int = Field(default=32000, description="Maximum context size for LLM (in characters)")
    max_context_lines: int = Field(default=200, description="Maximum lines of context to send to LLM")
    max_symbols_per_batch: int = Field(default=50, description="Maximum symbols to rename in one batch")
    program_batching_enabled: bool = Field(default=True, description="Allow strict batching for top-level program scope")
    program_max_symbols_per_batch: int = Field(default=10, description="Maximum top-level symbols per LLM call")
    program_variable_max_assignment_chars: int = Field(
        default=120,
        description="Max chars for top-level var/let/const assignment to be batch-eligible",
    )
    program_variable_max_assignment_lines: int = Field(
        default=2,
        description="Max lines for top-level var/let/const assignment to be batch-eligible",
    )
    program_function_max_chars: int = Field(
        default=600,
        description="Max chars for top-level function declaration to be batch-eligible",
    )
    program_function_max_lines: int = Field(
        default=20,
        description="Max lines for top-level function declaration to be batch-eligible",
    )
    local_scope_merge_enabled: bool = Field(
        default=True,
        description="Merge small function-local block/catch scopes into one batch",
    )
    local_scope_merge_function_max_chars: int = Field(
        default=1200,
        description="Max chars for a function to allow local block/catch scope merge",
    )
    local_scope_merge_function_max_lines: int = Field(
        default=40,
        description="Max lines for a function to allow local block/catch scope merge",
    )
    debug_scope_details: bool = Field(
        default=False,
        description="Include full scope details in debug logs (very verbose)",
    )
    context_padding: int = Field(default=200, description="Lines of context around symbol (legacy, use max_context_lines)")
    enable_uniquify: bool = Field(default=True, description="Apply binding-name uniquification before analysis")
    uniquify_timeout_seconds: int = Field(
        default=300,
        description="Stall timeout without JS progress events during binding-name uniquification",
    )
    beautify_after_each_rename: bool = Field(
        default=False,
        description="Beautify each intermediate generate step for more stable formatting",
    )
    retain_lines_during_generate: bool = Field(
        default=True,
        description="Preserve line layout during AST generation to reduce line drift",
    )

    # Output Settings
    output_dir: Optional[Path] = Field(default=None, description="Output directory for deobfuscated files")
    prettier_format: bool = Field(default=True, description="Apply prettier formatting to output")
    preserve_comments: bool = Field(default=True, description="Preserve comments in output")

    # Webcrack Settings
    unpack_webpack: bool = Field(default=True, description="Unpack webpack bundles")
    webcrack_output_dir: Optional[Path] = Field(default=None, description="Output dir for webcrack")

    model_config = {
        "env_prefix": "NAMUNIFY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate and load API key from environment if not provided."""
        if v:
            return v

        # Try to load from environment based on provider
        provider = info.data.get("llm_provider", LLMProvider.OPENAI)
        if provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        return v

    @field_validator("llm_model", mode="before")
    @classmethod
    def set_default_model(cls, v: Optional[str], info) -> str:
        """Set default model based on provider if not specified."""
        if v:
            return v

        provider = info.data.get("llm_provider", LLMProvider.OPENAI)
        if provider == LLMProvider.OPENAI:
            return "gpt-4o"
        elif provider == LLMProvider.ANTHROPIC:
            return "claude-sonnet-4-6-20250514"
        return v


# LLM Prompt templates
PROMPTS = {
    "rename_symbols": """You are a JavaScript code analysis expert. Your task is to rename obfuscated variable names to meaningful, descriptive names.

Given the following JavaScript code context and a list of symbols to rename, provide a JSON mapping from old names to new meaningful names.

Rules:
1. Analyze the context carefully to understand what each variable represents
2. Use descriptive, camelCase names that reflect the variable's purpose
3. If multiple symbols have the same name, use "varName:lineNumber" format as key to distinguish them
4. Consider the variable's usage patterns, assigned values, and how it's used
5. Output MUST be a valid JSON object wrapped in a markdown code block

Context (with line numbers):
```
{context}
```

Symbols to rename (with line numbers):
{symbols}

Example output format:
```json
{{
  "a": "userCount",
  "b": "processData",
  "a:45": "tempValue"
}}
```

Note: In the example above, "a" and "a:45" are two different variables with the same name at different lines.

Now provide the renaming mapping:""",

    "rename_single_symbol": """You are a JavaScript code analysis expert. Your task is to rename an obfuscated variable name to a meaningful, descriptive name.

Context (with line numbers):
```
{context}
```

Symbol to rename: {symbol}
Snippet where it appears:
```
{snippet}
```

Provide a single meaningful name for this variable. Output format:
```json
{{
  "{symbol_key}": "meaningfulName"
}}
```""",
}
