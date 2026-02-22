"""Namunify - JavaScript deobfuscation tool using LLM for variable renaming."""

__version__ = "0.1.0"
__author__ = "namunify"

from namunify.config import Config
from namunify.core.analyzer import analyze_identifiers
from namunify.core.parser import parse_javascript
from namunify.core.generator import generate_code

__all__ = [
    "__version__",
    "Config",
    "analyze_identifiers",
    "parse_javascript",
    "generate_code",
]
