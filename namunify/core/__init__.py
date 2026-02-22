"""Core deobfuscation functionality."""

from namunify.core.analyzer import analyze_identifiers, ScopeInfo, IdentifierInfo
from namunify.core.parser import parse_javascript
from namunify.core.generator import generate_code
from namunify.core.webcrack import unpack_webpack

__all__ = [
    "analyze_identifiers",
    "parse_javascript",
    "generate_code",
    "unpack_webpack",
    "ScopeInfo",
    "IdentifierInfo",
]
