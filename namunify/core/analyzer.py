"""Identifier analysis and grouping for deobfuscation."""

import re
from dataclasses import dataclass, field
from typing import Optional

from namunify.core.parser import (
    BindingIdentifier,
    ParseResult,
    Position,
    Range,
    Scope,
)


@dataclass
class IdentifierInfo:
    """Information about an identifier to be renamed."""
    name: str
    line: int
    column: int
    scope_id: str
    binding_type: str
    is_obfuscated: bool
    snippet: str = ""
    context_before: str = ""
    context_after: str = ""


@dataclass
class ScopeInfo:
    """Information about a scope with its renamable identifiers."""
    scope_id: str
    scope_type: str
    range: Range
    identifiers: list[IdentifierInfo] = field(default_factory=list)
    parent_scope_id: Optional[str] = None
    child_scope_ids: list[str] = field(default_factory=list)
    merged: bool = False


def is_obfuscated_name(name: str) -> bool:
    """Check if a name appears to be obfuscated.

    Considers names as obfuscated if they are:
    - Single character (a, b, c, etc.)
    - Two character combinations (aa, ab, etc.)
    - Mix of random letters and numbers (x1, a2b)
    - Common obfuscation patterns
    """
    if not name:
        return False

    # Common meaningful short names to preserve
    preserved_names = {
        "i", "j", "k",  # Loop counters (often meaningful)
        "x", "y", "z",  # Coordinates
        "id", "db", "io", "ui", "os",  # Common abbreviations
        "fs", "ls", "rm", "cp",  # Unix commands
        "up", "on", "in", "to",  # Common words
    }

    if name.lower() in preserved_names:
        return False

    # Single letter (except preserved)
    if len(name) == 1:
        return True

    # Two characters without clear meaning
    if len(name) == 2:
        # Check if it's a common abbreviation
        common_abbr = {"id", "io", "ui", "db", "fn", "cb", "ev", "el", "tx", "rx"}
        if name.lower() in common_abbr:
            return False
        return True

    # Mix of single letters and numbers (e.g., a1, b2, x1)
    if re.match(r'^[a-zA-Z]\d+$', name):
        return True

    # All single character repetitions or patterns (aaa, abc, xyz)
    if len(name) <= 3 and re.match(r'^[a-zA-Z]+$', name):
        # Check if it's a meaningful word or abbreviation
        if name.lower() in {"set", "get", "map", "key", "val", "arr", "obj", "str", "num", "err", "req", "res"}:
            return False
        # Short random letter combinations
        if len(set(name.lower())) <= 2:
            return True

    # Names with underscores but very short (a_b, x_1)
    if "_" in name and len(name) <= 5:
        return True

    # Hex-like names (a1b2, x9f3)
    if re.match(r'^[a-f0-9]+$', name.lower()) and len(name) >= 4:
        return True

    return False


def analyze_identifiers(
    parse_result: ParseResult,
    max_context_lines: int = 500,
    max_symbols_per_scope: int = 50,
) -> list[ScopeInfo]:
    """Analyze and group identifiers for renaming.

    This function:
    1. Filters identifiers to only obfuscated ones
    2. Groups them by scope
    3. Merges small nested scopes into parent scopes
    4. Prepares context for each scope

    Args:
        parse_result: Result of parsing JavaScript code
        max_context_lines: Maximum lines of context around a scope
        max_symbols_per_scope: Maximum symbols before splitting

    Returns:
        List of ScopeInfo objects ready for LLM processing
    """
    lines = parse_result.lines
    scopes = parse_result.scopes

    # Build scope tree relationships
    scope_infos: dict[str, ScopeInfo] = {}

    for scope_id, scope in scopes.items():
        # Filter to obfuscated identifiers
        obfuscated_bindings = [
            b for b in scope.bindings
            if is_obfuscated_name(b.name)
        ]

        if not obfuscated_bindings:
            continue

        identifiers = []
        for binding in obfuscated_bindings:
            info = IdentifierInfo(
                name=binding.name,
                line=binding.range.start.row,
                column=binding.range.start.column,
                scope_id=scope_id,
                binding_type=binding.binding_type,
                is_obfuscated=True,
            )
            identifiers.append(info)

        scope_info = ScopeInfo(
            scope_id=scope_id,
            scope_type=scope.scope_type,
            range=scope.range,
            identifiers=identifiers,
            parent_scope_id=scope.parent_id,
            child_scope_ids=scope.children.copy(),
        )
        scope_infos[scope_id] = scope_info

    # Merge small nested scopes into parents
    _merge_nested_scopes(scope_infos)

    # Prepare context for each scope
    for scope_info in scope_infos.values():
        if scope_info.merged:
            continue
        _prepare_scope_context(scope_info, lines, max_context_lines)

    # Split large scopes if needed
    result = []
    for scope_info in scope_infos.values():
        if scope_info.merged:
            continue

        if len(scope_info.identifiers) > max_symbols_per_scope:
            # Split into smaller chunks
            chunks = _split_large_scope(scope_info, max_symbols_per_scope)
            result.extend(chunks)
        else:
            result.append(scope_info)

    return result


def _merge_nested_scopes(scope_infos: dict[str, ScopeInfo]) -> None:
    """Merge small nested scopes into their parent scopes.

    If a child scope is completely contained within a parent scope
    and the parent scope only contains variable/constant bindings,
    merge the child's identifiers into the parent.
    """
    # Build parent-child relationships
    for scope_info in scope_infos.values():
        if scope_info.parent_scope_id and scope_info.parent_scope_id in scope_infos:
            parent = scope_infos[scope_info.parent_scope_id]
            if scope_info.scope_id not in parent.child_scope_ids:
                parent.child_scope_ids.append(scope_info.scope_id)

    # Process scopes from innermost to outermost
    scopes_to_process = sorted(
        scope_infos.values(),
        key=lambda s: len(s.child_scope_ids),
        reverse=True,
    )

    for scope_info in scopes_to_process:
        if scope_info.merged:
            continue

        # Check if this scope should be merged into parent
        if scope_info.parent_scope_id and scope_info.parent_scope_id in scope_infos:
            parent = scope_infos[scope_info.parent_scope_id]

            # Only merge if parent is a function/class/program scope
            # and child is a block scope
            if (parent.scope_type in ("function", "method", "class", "program") and
                scope_info.scope_type in ("block", "catch")):
                # Merge identifiers
                parent.identifiers.extend(scope_info.identifiers)
                # Expand parent range to include child
                if scope_info.range.start < parent.range.start:
                    parent.range.start = scope_info.range.start
                if scope_info.range.end > parent.range.end:
                    parent.range.end = scope_info.range.end
                # Mark child as merged
                scope_info.merged = True


def _prepare_scope_context(
    scope_info: ScopeInfo,
    lines: list[str],
    max_context_lines: int,
) -> None:
    """Prepare context for a scope."""
    start_line = max(0, scope_info.range.start.row - 10)
    end_line = min(len(lines), scope_info.range.end.row + 10)

    # Limit context size
    if end_line - start_line > max_context_lines:
        # Take start, end, and lines around each identifier
        context_lines = set(range(start_line, start_line + 50))
        context_lines.update(range(end_line - 50, end_line))

        for identifier in scope_info.identifiers:
            id_line = identifier.line
            context_lines.update(range(
                max(start_line, id_line - 5),
                min(end_line, id_line + 5),
            ))

        selected_lines = sorted(context_lines)
    else:
        selected_lines = list(range(start_line, end_line))

    # Build context string with line numbers
    context_parts = []
    for line_num in selected_lines:
        line_content = lines[line_num]
        context_parts.append(f"{line_num + 1:6d} | {line_content}")

    scope_info.identifiers[0].context_before = "\n".join(context_parts) if scope_info.identifiers else ""

    # Prepare snippets for each identifier
    for identifier in scope_info.identifiers:
        id_line = identifier.line
        snippet_start = max(0, id_line - 3)
        snippet_end = min(len(lines), id_line + 4)
        snippet_lines = []
        for i in range(snippet_start, snippet_end):
            marker = " >>> " if i == id_line else "     "
            snippet_lines.append(f"{marker}{i + 1:6d} | {lines[i]}")
        identifier.snippet = "\n".join(snippet_lines)


def _split_large_scope(
    scope_info: ScopeInfo,
    max_symbols: int,
) -> list[ScopeInfo]:
    """Split a large scope into smaller chunks."""
    chunks = []

    for i in range(0, len(scope_info.identifiers), max_symbols):
        chunk_identifiers = scope_info.identifiers[i:i + max_symbols]
        chunk = ScopeInfo(
            scope_id=f"{scope_info.scope_id}_chunk_{i // max_symbols}",
            scope_type=scope_info.scope_type,
            range=scope_info.range,
            identifiers=chunk_identifiers,
        )

        # Prepare context for this chunk
        if chunk_identifiers:
            chunk_identifiers[0].context_before = scope_info.identifiers[0].context_before

        chunks.append(chunk)

    return chunks


def format_context_with_line_numbers(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    """Format a range of lines with line numbers."""
    result = []
    for i in range(start_line, end_line):
        if 0 <= i < len(lines):
            result.append(f"{i + 1:6d} | {lines[i]}")
    return "\n".join(result)


def get_renamable_symbols(parse_result: ParseResult) -> list[str]:
    """Get list of all renamable (obfuscated) symbol names."""
    symbols = set()
    for binding in parse_result.all_bindings:
        if is_obfuscated_name(binding.name):
            symbols.add(binding.name)
    return sorted(symbols)
