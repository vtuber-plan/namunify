"""Identifier analysis and grouping for deobfuscation."""

import re
from dataclasses import dataclass, field
from typing import Optional

from namunify.core.parser import (
    ParseResult,
    Range,
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

    # If name was uniquified (e.g., foo__u12), evaluate obfuscation on the base
    # so uniquified short names can still be renamed by LLM.
    was_uniquified = False
    uniquified_match = re.match(r"^(.+)__u\d+$", name)
    if uniquified_match:
        base_name = uniquified_match.group(1)
        if base_name:
            name = base_name
            was_uniquified = True

    # Common meaningful short names to preserve
    preserved_names = {
        "i", "j", "k",  # Loop counters (often meaningful)
        "x", "y", "z",  # Coordinates
        "id", "db", "io", "ui", "os",  # Common abbreviations
        "fs", "ls", "rm", "cp",  # Unix commands
        "up", "on", "in", "to",  # Common words
    }

    # Keep preserving common short names for original source.
    # For uniquified names (e.g., i__u6), we intentionally do not preserve,
    # so they can still be renamed in deobfuscation.
    if not was_uniquified and name.lower() in preserved_names:
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
    program_batching_enabled: bool = True,
    program_max_symbols_per_batch: int = 30,
    program_variable_max_assignment_chars: int = 120,
    program_variable_max_assignment_lines: int = 2,
    program_function_max_chars: int = 600,
    program_function_max_lines: int = 20,
) -> list[ScopeInfo]:
    """Analyze and group identifiers for renaming.

    This function:
    1. Filters identifiers to only obfuscated ones
    2. Groups them by scope
    3. Prepares context for each scope
    4. Applies batching constraints by scope type

    Args:
        parse_result: Result of parsing JavaScript code
        max_context_lines: Maximum lines of context around a scope
        max_symbols_per_scope: Maximum symbols before splitting
        program_batching_enabled: Whether to allow strict batching in program scope
        program_max_symbols_per_batch: Max symbols per program-level batch
        program_variable_max_assignment_chars: Max chars for top-level var/let/const assignment
        program_variable_max_assignment_lines: Max lines for top-level var/let/const assignment
        program_function_max_chars: Max chars for top-level function declaration
        program_function_max_lines: Max lines for top-level function declaration

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

        if scope_info.scope_type == "program":
            result.extend(
                _split_program_scope_with_constraints(
                    scope_info=scope_info,
                    scopes=scopes,
                    lines=lines,
                    allow_batching=program_batching_enabled,
                    max_symbols=max_symbols_per_scope,
                    program_max_symbols=program_max_symbols_per_batch,
                    variable_max_chars=program_variable_max_assignment_chars,
                    variable_max_lines=program_variable_max_assignment_lines,
                    function_max_chars=program_function_max_chars,
                    function_max_lines=program_function_max_lines,
                )
            )
            continue

        if _can_batch_in_scope(scope_info.scope_type):
            if len(scope_info.identifiers) > max_symbols_per_scope:
                # Split into smaller chunks within the same scope range
                chunks = _split_large_scope(scope_info, max_symbols_per_scope)
                result.extend(chunks)
            else:
                result.append(scope_info)
            continue

        # Non function/class scope: never batch multiple symbols together.
        # This explicitly prevents program-level batching.
        if len(scope_info.identifiers) > 1:
            result.extend(_split_scope_to_singletons(scope_info))
        else:
            result.append(scope_info)

    result.sort(key=_scope_sort_key)
    return result


def _can_batch_in_scope(scope_type: str) -> bool:
    """Whether this scope type can contain multi-symbol batches.

    We only forbid batching at global program scope.
    """
    return scope_type != "program"


def _split_program_scope_with_constraints(
    scope_info: ScopeInfo,
    scopes: dict,
    lines: list[str],
    allow_batching: bool,
    max_symbols: int,
    program_max_symbols: int,
    variable_max_chars: int,
    variable_max_lines: int,
    function_max_chars: int,
    function_max_lines: int,
) -> list[ScopeInfo]:
    """Split program scope with strict batching rules for top-level symbols."""
    if not scope_info.identifiers:
        return []

    if not allow_batching:
        return _split_scope_to_singletons(scope_info) if len(scope_info.identifiers) > 1 else [scope_info]

    eligible: list[IdentifierInfo] = []
    ineligible: list[IdentifierInfo] = []

    for identifier in scope_info.identifiers:
        if _is_program_batch_eligible(
            identifier=identifier,
            scope_info=scope_info,
            scopes=scopes,
            lines=lines,
            variable_max_chars=variable_max_chars,
            variable_max_lines=variable_max_lines,
            function_max_chars=function_max_chars,
            function_max_lines=function_max_lines,
        ):
            eligible.append(identifier)
        else:
            ineligible.append(identifier)

    chunks: list[ScopeInfo] = []

    # Keep a strict cap for top-level batching.
    effective_max = max(1, min(max_symbols, program_max_symbols))
    for i in range(0, len(eligible), effective_max):
        batch_identifiers = eligible[i:i + effective_max]
        chunk = ScopeInfo(
            scope_id=f"{scope_info.scope_id}_program_batch_{i // effective_max}",
            scope_type=scope_info.scope_type,
            range=scope_info.range,
            identifiers=batch_identifiers,
            parent_scope_id=scope_info.parent_scope_id,
            child_scope_ids=scope_info.child_scope_ids.copy(),
        )
        if batch_identifiers and scope_info.identifiers:
            batch_identifiers[0].context_before = scope_info.identifiers[0].context_before
        chunks.append(chunk)

    if ineligible:
        singleton_scope = ScopeInfo(
            scope_id=f"{scope_info.scope_id}_program_ineligible",
            scope_type=scope_info.scope_type,
            range=scope_info.range,
            identifiers=ineligible,
            parent_scope_id=scope_info.parent_scope_id,
            child_scope_ids=scope_info.child_scope_ids.copy(),
        )
        chunks.extend(_split_scope_to_singletons(singleton_scope))

    return chunks


def _is_program_batch_eligible(
    identifier: IdentifierInfo,
    scope_info: ScopeInfo,
    scopes: dict,
    lines: list[str],
    variable_max_chars: int,
    variable_max_lines: int,
    function_max_chars: int,
    function_max_lines: int,
) -> bool:
    """Whether a program-scope identifier is safe to include in top-level batching."""
    if identifier.binding_type == "function":
        return _is_small_top_level_function(
            identifier=identifier,
            scope_info=scope_info,
            scopes=scopes,
            lines=lines,
            max_chars=function_max_chars,
            max_lines=function_max_lines,
        )

    if identifier.binding_type == "variable":
        return _is_small_top_level_assignment(
            identifier=identifier,
            lines=lines,
            max_chars=variable_max_chars,
            max_lines=variable_max_lines,
        )

    # Classes/others are conservative: keep as singleton.
    return False


def _is_small_top_level_function(
    identifier: IdentifierInfo,
    scope_info: ScopeInfo,
    scopes: dict,
    lines: list[str],
    max_chars: int,
    max_lines: int,
) -> bool:
    """Allow batching only for short function declarations in program scope."""
    scope_obj = scopes.get(scope_info.scope_id)
    if scope_obj is None:
        return False

    target_scope = None
    for child_id in scope_obj.children:
        child = scopes.get(child_id)
        if child is None:
            continue
        if child.scope_type not in {"function", "arrow"}:
            continue
        if child.range.start.row != identifier.line:
            continue
        target_scope = child
        break

    if target_scope is None:
        return False

    start_line = max(0, target_scope.range.start.row)
    end_line = min(len(lines) - 1, target_scope.range.end.row)
    if end_line < start_line:
        return False

    line_count = end_line - start_line + 1
    if line_count > max_lines:
        return False

    char_count = sum(len(lines[i]) + 1 for i in range(start_line, end_line + 1))
    return char_count <= max_chars


def _is_small_top_level_assignment(
    identifier: IdentifierInfo,
    lines: list[str],
    max_chars: int,
    max_lines: int,
) -> bool:
    """Allow batching only for short, closed top-level var/let/const assignments."""
    if not (0 <= identifier.line < len(lines)):
        return False

    head_line = lines[identifier.line]
    if not re.search(r"\b(?:var|let|const)\b", head_line):
        return False

    statement_lines = []
    end_found = False
    scan_limit = max(1, max_lines + 2)
    for i in range(identifier.line, min(len(lines), identifier.line + scan_limit)):
        statement_lines.append(lines[i])
        if ";" in lines[i]:
            end_found = True
            break

    if not end_found:
        return False

    line_count = len(statement_lines)
    if line_count > max_lines:
        return False

    statement_text = "\n".join(statement_lines)
    if "=" not in statement_text:
        return False

    # Keep assignment expressions simple for safer batch prompts.
    if "function" in statement_text or "=>" in statement_text:
        return False

    return len(statement_text) <= max_chars


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


def _split_scope_to_singletons(scope_info: ScopeInfo) -> list[ScopeInfo]:
    """Split scope so each chunk contains exactly one identifier."""
    chunks = []
    for i, identifier in enumerate(scope_info.identifiers):
        if not identifier.context_before and scope_info.identifiers:
            identifier.context_before = scope_info.identifiers[0].context_before

        chunk = ScopeInfo(
            scope_id=f"{scope_info.scope_id}_single_{i}",
            scope_type=scope_info.scope_type,
            range=scope_info.range,
            identifiers=[identifier],
            parent_scope_id=scope_info.parent_scope_id,
            child_scope_ids=scope_info.child_scope_ids.copy(),
        )
        chunks.append(chunk)

    return chunks


def _scope_sort_key(scope_info: ScopeInfo) -> tuple[int, int, int, str]:
    """Sort smaller scopes first, then smaller batches first.

    Order:
    1. Scope span in lines (small -> large)
    2. Identifier count in batch (small -> large)
    3. Scope start line (top -> bottom)
    4. Scope id (stable tie-breaker)
    """
    span_lines = max(1, scope_info.range.end.row - scope_info.range.start.row + 1)
    identifier_count = len(scope_info.identifiers)
    start_line = scope_info.range.start.row
    stable_id = scope_info.scope_id
    return (span_lines, identifier_count, start_line, stable_id)


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
