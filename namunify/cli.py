"""CLI interface for namunify."""

import asyncio
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from namunify import __version__
from namunify.config import Config, LLMProvider
from namunify.core import analyze_identifiers, parse_javascript, unpack_webpack
from namunify.core.generator import CodeGenerator, uniquify_binding_names
from namunify.core.webcrack import beautify_js_file
from namunify.llm import AnthropicClient, OpenAIClient
from namunify.plugins import BeautifyPlugin, PluginChain
from namunify.state import StateManager, ProcessingState, ask_resume

console = Console()

# Debug logger
debug_logger = None
debug_log_file = None

# State manager (global for process_file access)
state_manager = StateManager()
_PREPROCESS_CACHE_KEY = "__preprocess_cache__"


def setup_debug_logger(log_path: Optional[Path] = None) -> logging.Logger:
    """Setup debug logger for detailed logging."""
    global debug_logger, debug_log_file

    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"namunify_debug_{timestamp}.log")

    debug_log_file = log_path

    logger = logging.getLogger("namunify_debug")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Detailed format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    debug_logger = logger
    return logger


def debug_log(level: str, message: str, data: dict = None):
    """Log debug message with optional structured data."""
    global debug_logger
    if debug_logger is None:
        return

    log_func = getattr(debug_logger, level.lower(), debug_logger.info)

    if data:
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        log_func(f"{message}\n{data_str}")
    else:
        log_func(message)


def create_llm_client(config: Config):
    """Create LLM client based on configuration."""
    if config.llm_provider == LLMProvider.OPENAI:
        return OpenAIClient(
            api_key=config.llm_api_key,
            model=config.llm_model,
            base_url=config.llm_base_url,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
    elif config.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(
            api_key=config.llm_api_key,
            model=config.llm_model,
            base_url=config.llm_base_url,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _find_identifier_occurrence_lines(lines: list[str], identifier: str) -> list[int]:
    """Find likely identifier occurrence lines (0-indexed)."""
    if not identifier:
        return []

    # Avoid matching member access like obj.foo by excluding dot before identifier.
    pattern = re.compile(rf"(?<![A-Za-z0-9_$.]){re.escape(identifier)}(?![A-Za-z0-9_$])")
    return [idx for idx, line in enumerate(lines) if pattern.search(line)]


def _clip_line_content(content: str, max_chars: int = 500) -> str:
    """Clip very long source lines to keep prompt size bounded."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    head = max_chars - 3
    if head <= 0:
        return "..."
    return content[:head] + "..."


def _format_numbered_line(line_no: int, content: str, max_line_chars: int = 500) -> str:
    """Format a source line with line number and clipping."""
    return f"{line_no + 1:6d} | {_clip_line_content(content, max_line_chars)}"


def _truncate_text_middle(text: str, max_chars: int) -> str:
    """Truncate long text while preserving both head and tail."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n... [truncated] ...\n"
    keep = max_chars - len(marker)
    if keep <= 32:
        return text[:max_chars]

    head_keep = int(keep * 0.7)
    tail_keep = keep - head_keep
    return text[:head_keep] + marker + text[-tail_keep:]


def _apply_prompt_size_limits(
    context: str,
    snippets: dict[str, str],
    max_context_chars: int,
    max_snippet_chars: int,
) -> tuple[str, dict[str, str]]:
    """Apply size limits to context and snippets before sending to LLM."""
    limited_context = _truncate_text_middle(context, max_context_chars)
    limited_snippets = {
        name: _truncate_text_middle(snippet, max_snippet_chars)
        for name, snippet in snippets.items()
    }
    return limited_context, limited_snippets


def _is_input_too_long_error(exc: Exception) -> bool:
    """Check whether exception indicates request input is too long."""
    message = str(exc).lower()
    indicators = (
        "range of input length",
        "input length",
        "context length",
        "maximum context length",
        "token limit",
    )
    return any(indicator in message for indicator in indicators)


def _is_fatal_llm_response_error(exc: Exception) -> bool:
    """Whether an LLM response error is fatal and should stop processing."""
    message = str(exc).lower()
    indicators = (
        "openai-compatible api returned no usable content",
        "openai-compatible api returned unexpected response",
    )
    return any(indicator in message for indicator in indicators)


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Whether an LLM call failure should be retried instead of skipping symbols."""
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


def _parse_line_specific_key(key: str) -> Optional[tuple[str, int]]:
    """Parse a rename key in `name:line` format."""
    if not key:
        return None
    name, sep, line_text = key.rpartition(":")
    if not sep or not name or not line_text.isdigit():
        return None
    return name, int(line_text)


def _count_binding_names(parse_result) -> dict[str, int]:
    """Count declaration bindings by name in parse result."""
    counts: dict[str, int] = {}
    for binding in parse_result.all_bindings:
        counts[binding.name] = counts.get(binding.name, 0) + 1
    return counts


def _compute_sha256(text: str) -> str:
    """Compute SHA-256 hex digest for a text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_preprocess_cache(state: Optional[ProcessingState]) -> dict:
    """Read preprocess cache payload from processing state."""
    if state is None or not isinstance(state.config, dict):
        return {}
    cache = state.config.get(_PREPROCESS_CACHE_KEY)
    return cache if isinstance(cache, dict) else {}


def _existing_path(path_text: Optional[str]) -> Optional[Path]:
    """Return path only when text is valid and file exists."""
    if not path_text or not isinstance(path_text, str):
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    return path


def _normalize_rename_keys_for_unique_bindings(
    renames: dict[str, str],
    binding_name_counts: dict[str, int],
) -> dict[str, str]:
    """Normalize `name:line` keys to `name` when the binding name is globally unique.

    This avoids line-drift mismatches when source formatting changes across iterations.
    """
    normalized: dict[str, str] = {}

    # Keep non line-specific keys first.
    for key, value in renames.items():
        if _parse_line_specific_key(key) is None:
            normalized[key] = value

    # Then fold line-specific keys into plain symbol keys when safe.
    for key, value in renames.items():
        parsed = _parse_line_specific_key(key)
        if parsed is None:
            continue
        name, _line = parsed
        if binding_name_counts.get(name, 0) == 1:
            normalized.setdefault(name, value)
        else:
            normalized[key] = value

    return normalized


def _line_window(center: int, radius: int, total_lines: int) -> list[int]:
    """Get a bounded line window around a center line."""
    if total_lines <= 0:
        return []
    start = max(0, center - radius)
    end = min(total_lines, center + radius + 1)
    return list(range(start, end))


def _format_line_block(lines: list[str], line_numbers: list[int]) -> str:
    """Format selected line numbers with 1-based line labels."""
    return "\n".join(_format_numbered_line(line_no, lines[line_no]) for line_no in line_numbers)


def _build_global_symbol_context(
    lines: list[str],
    symbol: str,
    declaration_line: int,
    max_context_lines: int,
    reference_lines: Optional[list[int]] = None,
    declaration_window: int = 12,
    reference_window: int = 2,
    max_reference_points: int = 10,
) -> str:
    """Build focused context for a global symbol using declaration + reference snippets."""
    total_lines = len(lines)
    if total_lines == 0:
        return ""

    decl_line = min(max(0, declaration_line), total_lines - 1)
    if reference_lines is None:
        all_occurrences = _find_identifier_occurrence_lines(lines, symbol)
        selected_reference_lines = [line for line in all_occurrences if line != decl_line][:max_reference_points]
    else:
        selected_reference_lines = [line for line in reference_lines if line != decl_line][:max_reference_points]

    selected_set: set[int] = set()
    selected_ordered: list[int] = []

    def add_lines(candidates: list[int]) -> None:
        for line_no in candidates:
            if line_no in selected_set:
                continue
            if len(selected_ordered) >= max_context_lines:
                return
            selected_set.add(line_no)
            selected_ordered.append(line_no)

    add_lines(_line_window(decl_line, declaration_window, total_lines))
    for ref_line in selected_reference_lines:
        add_lines(_line_window(ref_line, reference_window, total_lines))
        if len(selected_ordered) >= max_context_lines:
            break

    selected_ordered.sort()
    reference_lines_1_based = (
        ", ".join(str(line + 1) for line in selected_reference_lines)
        if selected_reference_lines
        else "none"
    )

    header = (
        f"Global symbol: {symbol}\n"
        f"Declaration line: {decl_line + 1}\n"
        f"Reference lines: {reference_lines_1_based}\n"
    )
    return header + _format_line_block(lines, selected_ordered)


def _build_global_symbol_snippet(
    lines: list[str],
    symbol: str,
    declaration_line: int,
    reference_lines: Optional[list[int]] = None,
    snippet_radius: int = 3,
    max_reference_snippets: int = 3,
) -> str:
    """Build snippet containing declaration and several reference snippets for a global symbol."""
    total_lines = len(lines)
    if total_lines == 0:
        return ""

    decl_line = min(max(0, declaration_line), total_lines - 1)
    if reference_lines is None:
        occurrences = _find_identifier_occurrence_lines(lines, symbol)
        ref_lines = [line for line in occurrences if line != decl_line][:max_reference_snippets]
    else:
        ref_lines = [line for line in reference_lines if line != decl_line][:max_reference_snippets]

    parts = []
    decl_block_lines = _line_window(decl_line, snippet_radius, total_lines)
    decl_block = []
    for line_no in decl_block_lines:
        marker = " >>> " if line_no == decl_line else "     "
        decl_block.append(f"{marker}{line_no + 1:6d} | {lines[line_no]}")
    parts.append("Declaration:\n" + "\n".join(decl_block))

    for idx, ref_line in enumerate(ref_lines, start=1):
        ref_block_lines = _line_window(ref_line, snippet_radius, total_lines)
        ref_block = []
        for line_no in ref_block_lines:
            marker = " >>> " if line_no == ref_line else "     "
            ref_block.append(f"{marker}{line_no + 1:6d} | {lines[line_no]}")
        parts.append(f"Reference {idx} (line {ref_line + 1}):\n" + "\n".join(ref_block))

    return "\n\n".join(parts)


def _get_binding_reference_lines(
    parse_result,
    symbol: str,
    declaration_line: int,
) -> Optional[list[int]]:
    """Get reference lines for a specific binding from AST parse result."""
    for binding in parse_result.all_bindings:
        if binding.name != symbol:
            continue
        if binding.range.start.row != declaration_line:
            continue

        if not binding.reference_lines:
            return None
        return sorted({line for line in binding.reference_lines if line != declaration_line})

    return None


async def process_file(
    file_path: Path,
    config: Config,
    output_path: Optional[Path] = None,
    resume: bool = True,
) -> dict:
    """Process a single JavaScript file.

    Args:
        file_path: Path to JavaScript file
        config: Configuration
        output_path: Optional output path
        resume: Whether to resume from checkpoint if available

    Returns:
        Processing statistics
    """
    debug_log("info", f"=" * 80)
    debug_log("info", f"Starting processing file: {file_path}")

    stats = {
        "file": str(file_path),
        "symbols_found": 0,
        "symbols_renamed": 0,
        "scopes_processed": 0,
    }

    llm_client = None
    processing_state: Optional[ProcessingState] = None
    start_scope_idx = 0

    try:
        # Read original source code first
        source_code = file_path.read_text(encoding="utf-8")
        source_code_hash = _compute_sha256(source_code)
        preprocess_input_file = file_path
        beautified_file = file_path
        resumed_from_checkpoint = False

        # Check for existing checkpoint state before expensive preprocessing
        if resume and state_manager.has_state(file_path):
            processing_state = state_manager.load_state(file_path)
            if processing_state and ask_resume(processing_state, state_manager):
                resumed_from_checkpoint = True
                start_scope_idx = processing_state.processed_scopes
                debug_log("info", "Resume checkpoint accepted", {
                    "start_scope_idx": start_scope_idx,
                    "processed_scopes": processing_state.processed_scopes,
                    "total_scopes": processing_state.total_scopes,
                })
            else:
                # User chose not to resume, start fresh
                state_manager.clear_state(file_path)
                processing_state = None

        preprocess_cache = _extract_preprocess_cache(processing_state)
        if resumed_from_checkpoint and preprocess_cache:
            saved_hash = preprocess_cache.get("source_sha256")
            if isinstance(saved_hash, str) and saved_hash and saved_hash != source_code_hash:
                console.print("[yellow]Input changed since checkpoint; starting fresh (state cleared).[/yellow]")
                state_manager.clear_state(file_path)
                processing_state = None
                resumed_from_checkpoint = False
                start_scope_idx = 0
                preprocess_cache = {}
            saved_enable_uniquify = preprocess_cache.get("enable_uniquify")
            if (
                resumed_from_checkpoint
                and isinstance(saved_enable_uniquify, bool)
                and saved_enable_uniquify != config.enable_uniquify
            ):
                console.print(
                    "[yellow]Uniquify setting changed since checkpoint; starting fresh (state cleared).[/yellow]"
                )
                state_manager.clear_state(file_path)
                processing_state = None
                resumed_from_checkpoint = False
                start_scope_idx = 0
                preprocess_cache = {}

        cache_matches_input = (
            bool(preprocess_cache)
            and preprocess_cache.get("source_sha256") == source_code_hash
            and preprocess_cache.get("enable_uniquify") == config.enable_uniquify
        )
        cached_uniquified_file = (
            _existing_path(preprocess_cache.get("uniquified_file"))
            if cache_matches_input else None
        )
        cached_beautified_file = (
            _existing_path(preprocess_cache.get("beautified_file"))
            if cache_matches_input else None
        )
        uniquified_file_used: Optional[Path] = None
        used_cached_final_preprocess = False

        if cached_beautified_file is not None:
            if config.enable_uniquify:
                console.print(f"[blue]Uniquifying variable names[/blue] {file_path} [dim](cached)[/dim]")
                if cached_uniquified_file is not None:
                    preprocess_input_file = cached_uniquified_file
                    uniquified_file_used = cached_uniquified_file
            console.print(f"[blue]Beautifying[/blue] {preprocess_input_file} [dim](cached)[/dim]")
            beautified_file = cached_beautified_file
            source_code = cached_beautified_file.read_text(encoding="utf-8")
            used_cached_final_preprocess = True
            debug_log("info", "Using cached preprocess artifacts", {
                "uniquified_file": str(cached_uniquified_file) if cached_uniquified_file else None,
                "beautified_file": str(cached_beautified_file),
            })

        # Ensure variable bindings with the same name in different scopes are unique
        if not used_cached_final_preprocess and config.enable_uniquify:
            if cached_uniquified_file is not None:
                console.print(f"[blue]Uniquifying variable names[/blue] {file_path} [dim](cached)[/dim]")
                preprocess_input_file = cached_uniquified_file
                uniquified_file_used = cached_uniquified_file
                source_code = cached_uniquified_file.read_text(encoding="utf-8")
                debug_log("info", "Using cached uniquified source", {
                    "cached_file": str(cached_uniquified_file),
                })
            else:
                console.print(f"[blue]Uniquifying variable names[/blue] {file_path}")
                uniquified_file = file_path.with_suffix(".uniquified.js")
                progress_started_logged = False
                uniquify_pbar: Optional[tqdm] = None
                last_uniquify_renamed = 0

                def ensure_uniquify_pbar(total_bindings: int) -> None:
                    nonlocal uniquify_pbar
                    if uniquify_pbar is not None or total_bindings <= 0:
                        return
                    uniquify_pbar = tqdm(
                        total=total_bindings,
                        desc="Uniquifying bindings",
                        unit="binding",
                        ncols=100,
                    )

                def on_uniquify_progress(event: dict) -> None:
                    nonlocal progress_started_logged, last_uniquify_renamed
                    event_type = event.get("event")
                    event_source = event.get("source")

                    if event_type == "started" and event_source == "python" and not progress_started_logged:
                        progress_started_logged = True
                        debug_log("info", "Binding-name uniquification started", {
                            "file": str(file_path),
                            "timeout_seconds": event.get("timeout_seconds"),
                            "source_length": event.get("source_length"),
                        })
                        return

                    if event_type == "started" and event_source == "js":
                        total = event.get("totalBindingsToRename")
                        if isinstance(total, int):
                            ensure_uniquify_pbar(total)
                        debug_log("debug", "Binding-name uniquification JS started", event)
                        return

                    if event_type == "heartbeat":
                        elapsed = event.get("elapsed_seconds")
                        timeout = event.get("timeout_seconds")
                        if isinstance(elapsed, (int, float)) and isinstance(timeout, (int, float)):
                            debug_log("debug", "Binding-name uniquification heartbeat", {
                                "elapsed_seconds": elapsed,
                                "timeout_seconds": timeout,
                            })
                        return

                    if event_type == "rename_progress":
                        renamed = event.get("renamedBindings")
                        total = event.get("totalBindingsToRename")
                        if isinstance(renamed, int) and isinstance(total, int) and total > 0:
                            ensure_uniquify_pbar(total)
                            current = max(0, min(renamed, total))
                            delta = current - last_uniquify_renamed
                            if delta > 0 and uniquify_pbar is not None:
                                uniquify_pbar.update(delta)
                            last_uniquify_renamed = max(last_uniquify_renamed, current)
                            debug_log("debug", "Binding-name rename progress", {
                                "renamed_bindings": renamed,
                                "total_bindings_to_rename": total,
                                "processed_groups": event.get("processedGroups"),
                                "total_duplicate_groups": event.get("totalDuplicateGroups"),
                                "current_group_name": event.get("currentGroupName"),
                            })
                        return

                    if event_type == "completed":
                        elapsed = event.get("elapsed_seconds")
                        if event_source == "python":
                            debug_log("info", "Binding-name uniquification completed", {
                                "elapsed_seconds": elapsed,
                                "output_path": event.get("output_path"),
                            })
                        else:
                            total = event.get("totalBindingsToRename")
                            renamed = event.get("renamedBindings")
                            if isinstance(total, int) and total > 0:
                                ensure_uniquify_pbar(total)
                                current = max(0, min(renamed if isinstance(renamed, int) else total, total))
                                delta = current - last_uniquify_renamed
                                if delta > 0 and uniquify_pbar is not None:
                                    uniquify_pbar.update(delta)
                                last_uniquify_renamed = max(last_uniquify_renamed, current)
                            debug_log("debug", "Binding-name uniquification JS completed", event)
                        return

                    if event_type in {"timeout", "failed", "error"}:
                        debug_log("warning", "Binding-name uniquification status", event)

                uniquified_source = uniquify_binding_names(
                    source_code,
                    output_path=uniquified_file,
                    timeout_seconds=config.uniquify_timeout_seconds,
                    progress_callback=on_uniquify_progress,
                )
                if uniquify_pbar is not None:
                    uniquify_pbar.close()
                if uniquified_source != source_code:
                    debug_log("info", f"Applied binding-name uniquification: {uniquified_file}")
                    source_code = uniquified_source
                else:
                    debug_log("info", f"Binding-name uniquification produced no changes: {uniquified_file}")
                if uniquified_file.exists():
                    preprocess_input_file = uniquified_file
                    uniquified_file_used = uniquified_file
        elif not used_cached_final_preprocess:
            debug_log("info", "Binding-name uniquification disabled by configuration")

        # Beautify after uniquification so formatting reflects final identifier names
        if used_cached_final_preprocess:
            debug_log("info", f"Beautified file: {beautified_file}")
        elif cached_beautified_file is not None:
            console.print(f"[blue]Beautifying[/blue] {preprocess_input_file} [dim](cached)[/dim]")
            beautified_file = cached_beautified_file
            debug_log("info", "Using cached beautified source", {
                "cached_file": str(cached_beautified_file),
            })
        else:
            console.print(f"[blue]Beautifying[/blue] {preprocess_input_file}")
            beautified_file = beautify_js_file(preprocess_input_file)
            debug_log("info", f"Beautified file: {beautified_file}")

        preprocess_cache_payload = {
            "source_sha256": source_code_hash,
            "enable_uniquify": config.enable_uniquify,
            "uniquified_file": (
                str(uniquified_file_used.resolve())
                if uniquified_file_used is not None and uniquified_file_used.exists()
                else ""
            ),
            "beautified_file": str(beautified_file.resolve()) if beautified_file.exists() else "",
        }

        if processing_state is not None and isinstance(processing_state.config, dict):
            processing_state.config[_PREPROCESS_CACHE_KEY] = preprocess_cache_payload
            state_manager.save_state(processing_state)
            debug_log("debug", "Updated checkpoint preprocess cache", preprocess_cache_payload)

        # Read source code (from beautified file if available)
        source_code = beautified_file.read_text(encoding="utf-8")
        debug_log("debug", f"Source code length: {len(source_code)} chars")

        generator = CodeGenerator(source_code)

        # Setup plugin chain
        plugin_chain = PluginChain()
        if config.prettier_format:
            plugin_chain.add_plugin(BeautifyPlugin())

        # Parse JavaScript
        console.print(f"[blue]Parsing[/blue] {file_path}")
        parse_result = parse_javascript(source_code)
        initial_binding_name_counts = _count_binding_names(parse_result)
        debug_log("info", f"Parsed {len(parse_result.scopes)} scopes, {len(parse_result.all_bindings)} bindings")

        # Analyze identifiers
        scope_infos = analyze_identifiers(
            parse_result,
            max_context_lines=config.context_padding,
            max_symbols_per_scope=config.max_symbols_per_batch,
            program_batching_enabled=config.program_batching_enabled,
            program_max_symbols_per_batch=config.program_max_symbols_per_batch,
            program_variable_max_assignment_chars=config.program_variable_max_assignment_chars,
            program_variable_max_assignment_lines=config.program_variable_max_assignment_lines,
            program_function_max_chars=config.program_function_max_chars,
            program_function_max_lines=config.program_function_max_lines,
        )

        total_symbols = sum(len(s.identifiers) for s in scope_infos)
        stats["symbols_found"] = total_symbols
        stats["scopes_processed"] = len(scope_infos)

        debug_log("info", f"Found {total_symbols} obfuscated symbols in {len(scope_infos)} scopes")
        if config.debug_scope_details:
            debug_log("debug", "Scope details", {
                "scope_details": [
                    {
                        "scope_id": s.scope_id,
                        "scope_type": s.scope_type,
                        "range": f"lines {s.range.start.row}-{s.range.end.row}",
                        "symbol_count": len(s.identifiers),
                        "symbols": [id.name for id in s.identifiers],
                    }
                    for s in scope_infos
                ]
            })

        console.print(f"[green]Found {total_symbols} obfuscated symbols in {len(scope_infos)} scopes[/green]")

        if total_symbols == 0:
            console.print("[yellow]No obfuscated symbols found[/yellow]")
            return stats

        if resumed_from_checkpoint and processing_state is not None:
            # Apply already processed renames
            normalized_saved_renames = _normalize_rename_keys_for_unique_bindings(
                processing_state.all_renames,
                initial_binding_name_counts,
            )
            generator.apply_renames(normalized_saved_renames)
            processing_state.all_renames = normalized_saved_renames
            stats["symbols_renamed"] = len(normalized_saved_renames)
            console.print(f"[green]Resuming from scope {start_scope_idx + 1}[/green]")
            debug_log("info", "Resuming from checkpoint", {
                "start_scope_idx": start_scope_idx,
                "already_renamed_count": len(normalized_saved_renames),
            })

        # Create or get processing state
        if processing_state is None:
            config_dict = {
                "llm_provider": str(config.llm_provider),
                "llm_model": config.llm_model,
                "llm_base_url": config.llm_base_url,
                "max_symbols_per_batch": config.max_symbols_per_batch,
                "program_batching_enabled": config.program_batching_enabled,
                "program_max_symbols_per_batch": config.program_max_symbols_per_batch,
                "program_variable_max_assignment_chars": config.program_variable_max_assignment_chars,
                "program_variable_max_assignment_lines": config.program_variable_max_assignment_lines,
                "program_function_max_chars": config.program_function_max_chars,
                "program_function_max_lines": config.program_function_max_lines,
                "context_padding": config.context_padding,
                _PREPROCESS_CACHE_KEY: preprocess_cache_payload,
            }
            processing_state = state_manager.create_state(
                input_file=file_path,
                output_file=output_path or file_path.with_suffix(".deobfuscated.js"),
                total_scopes=len(scope_infos),
                config=config_dict,
            )
            console.print(f"[dim]Checkpoint state saved to .namunify_state/[/dim]")

        # Create LLM client
        llm_client = create_llm_client(config)
        debug_log("info", f"Created LLM client", {
            "provider": str(config.llm_provider),
            "model": config.llm_model,
            "base_url": config.llm_base_url,
            "max_tokens": config.llm_max_tokens,
            "temperature": config.llm_temperature,
        })

        all_renames = dict(processing_state.all_renames) if processing_state.all_renames else {}

        # Track remaining symbols to rename (excluding already renamed)
        remaining_symbol_names = set()
        for scope_info in scope_infos[start_scope_idx:]:
            for id_info in scope_info.identifiers:
                if id_info.name not in all_renames:
                    remaining_symbol_names.add(id_info.name)

        # Process scopes iteratively, re-parsing after each rename
        pbar = tqdm(
            total=total_symbols,
            initial=len(all_renames),
            desc="Renaming symbols",
            unit="symbol",
            ncols=100,
        )
        consecutive_retryable_errors = 0

        while remaining_symbol_names:
            # Re-parse current source to get updated AST and positions
            current_source = generator.get_current_source()
            current_parse_result = parse_javascript(current_source)
            current_binding_name_counts = _count_binding_names(current_parse_result)
            current_scope_infos = analyze_identifiers(
                current_parse_result,
                max_context_lines=config.context_padding,
                max_symbols_per_scope=config.max_symbols_per_batch,
                program_batching_enabled=config.program_batching_enabled,
                program_max_symbols_per_batch=config.program_max_symbols_per_batch,
                program_variable_max_assignment_chars=config.program_variable_max_assignment_chars,
                program_variable_max_assignment_lines=config.program_variable_max_assignment_lines,
                program_function_max_chars=config.program_function_max_chars,
                program_function_max_lines=config.program_function_max_lines,
            )

            # Find the first scope with remaining symbols to rename
            found_scope = False
            for scope_info in current_scope_infos:
                scope_symbols = [id.name for id in scope_info.identifiers if id.name in remaining_symbol_names]
                if not scope_symbols:
                    continue

                found_scope = True
                symbols = scope_symbols
                symbol_lines = {id.name: id.line + 1 for id in scope_info.identifiers if id.name in remaining_symbol_names}

                # Build context from current source
                # Limit context size to avoid sending huge context to LLM
                MAX_CONTEXT_LINES = config.max_context_lines  # Max lines to send

                current_lines = current_source.split("\n")
                scope_line_count = scope_info.range.end.row - scope_info.range.start.row + 1
                global_context_meta: dict[str, object] = {}

                if scope_info.scope_type == "program" and len(symbols) == 1:
                    # Global symbol: use focused declaration context + reference snippets.
                    symbol = symbols[0]
                    declaration_line = symbol_lines.get(symbol, 1) - 1
                    ast_reference_lines = _get_binding_reference_lines(
                        current_parse_result,
                        symbol,
                        declaration_line,
                    )
                    context = _build_global_symbol_context(
                        current_lines,
                        symbol,
                        declaration_line,
                        max_context_lines=MAX_CONTEXT_LINES,
                        reference_lines=ast_reference_lines,
                    )
                    reference_preview_lines = ast_reference_lines or []
                    global_context_meta = {
                        "global_declaration_text": _clip_line_content(
                            current_lines[declaration_line],
                            300,
                        ) if 0 <= declaration_line < len(current_lines) else "",
                        "global_reference_texts": [
                            f"{line_no + 1}: {_clip_line_content(current_lines[line_no], 300)}"
                            for line_no in reference_preview_lines[:5]
                            if 0 <= declaration_line < len(current_lines)
                            and 0 <= line_no < len(current_lines)
                        ],
                    }
                elif scope_line_count <= MAX_CONTEXT_LINES:
                    # Small scope: include entire scope with padding
                    context_start = max(0, scope_info.range.start.row - 10)
                    context_end = min(len(current_lines), scope_info.range.end.row + 10)
                    context_lines = []
                    for i in range(context_start, context_end):
                        context_lines.append(_format_numbered_line(i, current_lines[i]))
                    context = "\n".join(context_lines)
                else:
                    # Large scope: only include context around each symbol
                    context_line_set = set()
                    for id_info in scope_info.identifiers:
                        if id_info.name not in remaining_symbol_names:
                            continue
                        id_line = id_info.line
                        # Add lines around each symbol
                        for i in range(max(0, id_line - 20), min(len(current_lines), id_line + 20)):
                            context_line_set.add(i)

                    # If still too many lines, limit further
                    if len(context_line_set) > MAX_CONTEXT_LINES:
                        # Prioritize lines closest to symbols
                        sorted_lines = sorted(context_line_set)
                        context_line_set = set(sorted_lines[:MAX_CONTEXT_LINES])

                    context_lines = []
                    for i in sorted(context_line_set):
                        context_lines.append(_format_numbered_line(i, current_lines[i]))
                    context = "\n".join(context_lines)

                # Build snippets dict
                snippets = {}
                for id_info in scope_info.identifiers:
                    if id_info.name not in remaining_symbol_names:
                        continue
                    id_line = id_info.line
                    if scope_info.scope_type == "program":
                        ast_reference_lines = _get_binding_reference_lines(
                            current_parse_result,
                            id_info.name,
                            id_line,
                        )
                        snippets[id_info.name] = _build_global_symbol_snippet(
                            current_lines,
                            id_info.name,
                            id_line,
                            reference_lines=ast_reference_lines,
                        )
                    else:
                        snippet_start = max(0, id_line - 3)
                        snippet_end = min(len(current_lines), id_line + 4)
                        snippet_lines = []
                        for i in range(snippet_start, snippet_end):
                            marker = " >>> " if i == id_line else "     "
                            snippet_lines.append(f"{marker}{_format_numbered_line(i, current_lines[i])}")
                        snippets[id_info.name] = "\n".join(snippet_lines)

                # Enforce max prompt size to avoid provider input-length errors.
                # `max_context_size` is char-based budget for context text.
                max_context_chars = max(2048, config.max_context_size)
                max_snippet_chars = max(1024, min(12000, max_context_chars // 2))
                context, snippets = _apply_prompt_size_limits(
                    context=context,
                    snippets=snippets,
                    max_context_chars=max_context_chars,
                    max_snippet_chars=max_snippet_chars,
                )

                debug_log("info", f"\n{'='*60}")
                debug_log("info", f"Processing scope: {scope_info.scope_id}", {
                    "scope_type": scope_info.scope_type,
                    "range": f"lines {scope_info.range.start.row}-{scope_info.range.end.row}",
                    "symbols_to_rename": symbols,
                    "symbol_lines": symbol_lines,
                    "context_length": len(context),
                    "context_preview": context[:500] + "..." if len(context) > 500 else context,
                    "already_renamed_count": len(all_renames),
                    "remaining_count": len(remaining_symbol_names),
                    **global_context_meta,
                })

                try:
                    # Call LLM to rename
                    try:
                        renames = await llm_client.rename_symbols(
                            context=context,
                            symbols=symbols,
                            snippets=snippets,
                            symbol_lines=symbol_lines,
                        )
                    except Exception as e:
                        if not _is_input_too_long_error(e):
                            raise

                        debug_log("warning", "LLM input too long, retrying with compact context", {
                            "error": str(e),
                            "symbols": symbols,
                            "context_length": len(context),
                        })
                        compact_context, compact_snippets = _apply_prompt_size_limits(
                            context=context,
                            snippets=snippets,
                            max_context_chars=min(8000, max_context_chars),
                            max_snippet_chars=min(2000, max_snippet_chars),
                        )
                        renames = await llm_client.rename_symbols(
                            context=compact_context,
                            symbols=symbols,
                            snippets=compact_snippets,
                            symbol_lines=symbol_lines,
                        )

                    debug_log("info", f"LLM response", {
                        "renames": renames,
                        "symbols_requested": symbols,
                        "symbols_returned": list(renames.keys()),
                    })

                    # Keep only renames that match current requested symbols.
                    # This prevents progress inflation when LLM returns unrelated keys.
                    filtered_renames: dict[str, str] = {}
                    resolved_symbols: set[str] = set()
                    for symbol in symbols:
                        if symbol in renames:
                            filtered_renames[symbol] = renames[symbol]
                            resolved_symbols.add(symbol)
                            continue

                        if symbol_lines and symbol in symbol_lines:
                            line_key = f"{symbol}:{symbol_lines[symbol]}"
                            if line_key in renames:
                                filtered_renames[line_key] = renames[line_key]
                                resolved_symbols.add(symbol)
                                continue

                        # Fallback for single-symbol mode: accept a single returned rename value
                        # even if the key is not exact (e.g., "symbolName").
                        if len(symbols) == 1 and len(renames) == 1:
                            only_value = next(iter(renames.values()))
                            if isinstance(only_value, str) and only_value.strip():
                                filtered_renames[symbol] = only_value.strip()
                                resolved_symbols.add(symbol)

                    if not filtered_renames:
                        debug_log("warning", "No usable renames returned for requested symbols", {
                            "symbols_requested": symbols,
                            "symbols_returned": list(renames.keys()),
                        })
                        # Avoid infinite loop on unusable model output.
                        for symbol in symbols:
                            remaining_symbol_names.discard(symbol)
                        pbar.update(len(symbols))
                        pbar.set_postfix_str(f"remaining={len(remaining_symbol_names)}")
                        continue

                    filtered_renames = _normalize_rename_keys_for_unique_bindings(
                        filtered_renames,
                        current_binding_name_counts,
                    )

                    # Apply validated renames
                    generator.apply_renames(filtered_renames)
                    stats["symbols_renamed"] += len(resolved_symbols)
                    all_renames.update(filtered_renames)

                    # Remove renamed symbols from remaining
                    for resolved_symbol in resolved_symbols:
                        remaining_symbol_names.discard(resolved_symbol)

                    # Save checkpoint state
                    state_manager._current_state.all_renames = all_renames.copy()
                    state_manager._current_state.processed_scopes = len(scope_infos) - len(remaining_symbol_names)
                    state_manager.save_state(state_manager._current_state)

                    debug_log("info", f"Applied {len(filtered_renames)} renames, checkpoint saved", {
                        "total_renamed": stats["symbols_renamed"],
                        "resolved_symbols_count": len(resolved_symbols),
                        "remaining_count": len(remaining_symbol_names),
                    })

                    # Update progress bar
                    pbar.update(len(resolved_symbols))
                    pbar.set_postfix_str(f"remaining={len(remaining_symbol_names)}")
                    consecutive_retryable_errors = 0

                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    debug_log("error", f"Error processing scope", {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": error_trace,
                        "symbols": symbols,
                    })
                    if _is_fatal_llm_response_error(e):
                        console.print(f"[red]Error: {e}[/red]")
                        raise
                    if _is_retryable_llm_error(e):
                        consecutive_retryable_errors += 1
                        retry_delay = min(60, 2 ** min(consecutive_retryable_errors, 6))
                        debug_log("warning", "Retryable LLM error, will retry same symbols", {
                            "error": str(e),
                            "symbols": symbols,
                            "retry_delay_seconds": retry_delay,
                            "consecutive_retryable_errors": consecutive_retryable_errors,
                        })
                        console.print(
                            f"[yellow]Retryable LLM error, retrying in {retry_delay}s: {e}[/yellow]"
                        )
                        if consecutive_retryable_errors >= 12:
                            raise RuntimeError(
                                "LLM rate limit/connection errors persisted for 12 retries; "
                                "please lower request rate or retry later."
                            ) from e
                        await asyncio.sleep(retry_delay)
                    else:
                        console.print(f"[red]Error: {e}[/red]")
                        # Skip these symbols and continue for non-retryable scope-level failures.
                        for s in symbols:
                            remaining_symbol_names.discard(s)

                break  # Exit scope loop after processing first scope with remaining symbols

            if not found_scope:
                # No more scopes with remaining symbols
                break

        # Close progress bar
        pbar.close()

        # Mark as completed
        state_manager.mark_completed()
        debug_log("info", f"Processing marked as completed")

        # Log all renames summary
        debug_log("info", f"\n{'='*80}")
        debug_log("info", f"All renames applied:", {
            "total_symbols_found": stats["symbols_found"],
            "total_symbols_renamed": stats["symbols_renamed"],
            "all_renames": all_renames,
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        debug_log("error", f"Fatal error processing file", {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_trace,
        })
        stats["error"] = str(e)
        console.print(f"[red]Error: {e}[/red]")
        raise
    finally:
        if llm_client:
            await llm_client.close()
            debug_log("info", "LLM client closed")

    # Determine output path
    if output_path is None:
        if config.output_dir:
            output_path = config.output_dir / file_path.name
        else:
            output_path = file_path.with_suffix(".deobfuscated.js")

    # Save output
    await generator.format_and_save(output_path, use_prettier=config.prettier_format)
    debug_log("info", f"Saved output to: {output_path}")

    debug_log("info", f"Processing complete", {"stats": stats})

    return stats


async def process_directory(
    dir_path: Path,
    config: Config,
    output_dir: Optional[Path] = None,
    resume: bool = True,
) -> list[dict]:
    """Process all JavaScript files in a directory.

    Args:
        dir_path: Path to directory
        config: Configuration
        output_dir: Optional output directory
        resume: Whether to resume from checkpoint if available

    Returns:
        List of processing statistics for each file
    """
    js_files = list(dir_path.rglob("*.js"))
    console.print(f"[blue]Found {len(js_files)} JavaScript files in {dir_path}[/blue]")

    debug_log("info", f"Processing directory: {dir_path}", {
        "js_files_count": len(js_files),
        "js_files": [str(f) for f in js_files[:10]],  # First 10 files
    })

    results = []
    for js_file in js_files:
        # Skip node_modules and minified files
        if "node_modules" in str(js_file):
            continue
        if ".min." in js_file.name:
            continue

        rel_path = js_file.relative_to(dir_path)
        out_path = output_dir / rel_path if output_dir else None

        try:
            result = await process_file(js_file, config, out_path, resume=resume)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error processing {js_file}: {e}[/red]")
            results.append({"file": str(js_file), "error": str(e)})

    return results


@click.group()
@click.version_option(version=__version__)
def main():
    """Namunify - JavaScript deobfuscation tool using LLM."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), help="Output file/directory path")
@click.option("--provider", type=click.Choice(["openai", "anthropic"]), default="openai", help="LLM provider")
@click.option("--model", help="LLM model name")
@click.option("--api-key", help="API key (or set NAMUNIFY_LLM_API_KEY env)")
@click.option("--base-url", help="Custom API base URL")
@click.option("--max-symbols", default=50, help="Max symbols per LLM call")
@click.option("--program-max-symbols", default=10, help="Max top-level symbols per LLM call")
@click.option(
    "--program-var-max-chars",
    default=120,
    help="Max chars for top-level var/let/const assignment to allow batching",
)
@click.option(
    "--program-var-max-lines",
    default=2,
    help="Max lines for top-level var/let/const assignment to allow batching",
)
@click.option(
    "--program-fn-max-chars",
    default=600,
    help="Max chars for top-level function declaration to allow batching",
)
@click.option(
    "--program-fn-max-lines",
    default=20,
    help="Max lines for top-level function declaration to allow batching",
)
@click.option("--context-padding", default=500, help="Lines of context around symbols")
@click.option("--no-prettier", is_flag=True, help="Disable prettier formatting")
@click.option("--no-uniquify", is_flag=True, help="Disable binding-name uniquification step")
@click.option("--no-program-batch", is_flag=True, help="Disable program-scope batching and force single-symbol top-level calls")
@click.option("--uniquify-timeout", default=300, help="Stall timeout without JS progress (seconds)")
@click.option("--unpack", is_flag=True, help="Unpack webpack bundle first")
@click.option("--install-webcrack", is_flag=True, help="Install webcrack if needed")
@click.option("--debug", is_flag=True, help="Enable debug logging to file")
@click.option("--debug-scope-details", is_flag=True, help="Include full scope details in debug logs (very verbose)")
@click.option("--debug-file", type=click.Path(path_type=Path), help="Debug log file path (default: namunify_debug_TIMESTAMP.log)")
@click.option("--no-resume", is_flag=True, help="Disable checkpoint resume, start fresh")
def deobfuscate(
    input_path: Path,
    output_path: Optional[Path],
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    max_symbols: int,
    program_max_symbols: int,
    program_var_max_chars: int,
    program_var_max_lines: int,
    program_fn_max_chars: int,
    program_fn_max_lines: int,
    context_padding: int,
    no_prettier: bool,
    no_uniquify: bool,
    no_program_batch: bool,
    uniquify_timeout: int,
    unpack: bool,
    install_webcrack: bool,
    debug: bool,
    debug_scope_details: bool,
    debug_file: Optional[Path],
    no_resume: bool,
):
    """Deobfuscate JavaScript code using LLM.

    INPUT_PATH can be a JavaScript file or directory containing JS files.

    Checkpoint/resume: Processing state is automatically saved to .namunify_state/
    directory. If interrupted, run the same command again to resume from checkpoint.
    Use --no-resume to start fresh and ignore existing checkpoints.
    """
    # Setup debug logger if requested
    if debug:
        setup_debug_logger(debug_file)
        console.print(f"[yellow]Debug logging enabled: {debug_log_file}[/yellow]")
        debug_log("info", "Debug logging started", {
            "input_path": str(input_path),
            "output_path": str(output_path) if output_path else None,
            "provider": provider,
            "model": model,
            "max_symbols": max_symbols,
            "program_max_symbols": program_max_symbols,
            "program_var_max_chars": program_var_max_chars,
            "program_var_max_lines": program_var_max_lines,
            "program_fn_max_chars": program_fn_max_chars,
            "program_fn_max_lines": program_fn_max_lines,
            "program_batching_enabled": not no_program_batch,
            "debug_scope_details": debug_scope_details,
            "context_padding": context_padding,
            "prettier_format": not no_prettier,
            "enable_uniquify": not no_uniquify,
            "uniquify_timeout_seconds": uniquify_timeout,
        })

    # Build config kwargs, only include non-None CLI args to let .env values be used
    config_kwargs = {
        "llm_provider": LLMProvider(provider),
        "max_symbols_per_batch": max_symbols,
        "program_batching_enabled": not no_program_batch,
        "program_max_symbols_per_batch": program_max_symbols,
        "program_variable_max_assignment_chars": program_var_max_chars,
        "program_variable_max_assignment_lines": program_var_max_lines,
        "program_function_max_chars": program_fn_max_chars,
        "program_function_max_lines": program_fn_max_lines,
        "debug_scope_details": debug_scope_details,
        "context_padding": context_padding,
        "prettier_format": not no_prettier,
        "enable_uniquify": not no_uniquify,
        "uniquify_timeout_seconds": uniquify_timeout,
    }

    # Only override .env values if CLI args are explicitly provided
    if model:
        config_kwargs["llm_model"] = model
    if api_key:
        config_kwargs["llm_api_key"] = api_key
    if base_url:
        config_kwargs["llm_base_url"] = base_url

    config = Config(**config_kwargs)

    debug_log("info", "Configuration loaded", {
        "llm_provider": str(config.llm_provider),
        "llm_model": config.llm_model,
        "llm_base_url": config.llm_base_url,
        "max_symbols_per_batch": config.max_symbols_per_batch,
        "program_batching_enabled": config.program_batching_enabled,
        "program_max_symbols_per_batch": config.program_max_symbols_per_batch,
        "program_variable_max_assignment_chars": config.program_variable_max_assignment_chars,
        "program_variable_max_assignment_lines": config.program_variable_max_assignment_lines,
        "program_function_max_chars": config.program_function_max_chars,
        "program_function_max_lines": config.program_function_max_lines,
        "debug_scope_details": config.debug_scope_details,
        "context_padding": config.context_padding,
        "prettier_format": config.prettier_format,
        "enable_uniquify": config.enable_uniquify,
        "uniquify_timeout_seconds": config.uniquify_timeout_seconds,
    })

    # Validate API key
    if not config.llm_api_key:
        env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        console.print(f"[red]Error: API key required. Set {env_var} environment variable or use --api-key[/red]")
        debug_log("error", "API key not found")
        raise SystemExit(1)

    async def run():
        results = []
        resume_enabled = not no_resume

        # Handle webpack unpacking
        current_input = input_path
        if unpack and input_path.is_file():
            unpacked_dir = await unpack_webpack(input_path, force_install=install_webcrack)
            if unpacked_dir:
                current_input = unpacked_dir
                debug_log("info", f"Unpacked webpack to: {unpacked_dir}")

        if current_input.is_file():
            result = await process_file(current_input, config, output_path, resume=resume_enabled)
            results.append(result)
        else:
            results = await process_directory(current_input, config, output_path, resume=resume_enabled)

        # Print summary
        table = Table(title="Processing Summary")
        table.add_column("File")
        table.add_column("Symbols Found")
        table.add_column("Symbols Renamed")
        table.add_column("Status")

        for r in results:
            status = "✓" if "error" not in r else "✗"
            table.add_row(
                r.get("file", "unknown"),
                str(r.get("symbols_found", 0)),
                str(r.get("symbols_renamed", 0)),
                status,
            )

        console.print(table)

        debug_log("info", "Processing complete", {"results": results})

        if debug:
            console.print(f"\n[yellow]Debug log saved to: {debug_log_file}[/yellow]")

    asyncio.run(run())


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), help="Output directory")
@click.option("--install", is_flag=True, help="Install webcrack if needed")
def unpack(input_path: Path, output_path: Optional[Path], install: bool):
    """Unpack a webpack bundle using webcrack."""

    async def run():
        result = await unpack_webpack(input_path, output_path, force_install=install)
        if result:
            console.print(f"[green]Unpacked to: {result}[/green]")
        else:
            console.print("[red]Unpacking failed[/red]")
            raise SystemExit(1)

    asyncio.run(run())


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
def analyze(input_path: Path):
    """Analyze JavaScript file and show obfuscated symbols."""
    source_code = input_path.read_text(encoding="utf-8")
    parse_result = parse_javascript(source_code)
    scope_infos = analyze_identifiers(parse_result)

    console.print(f"[blue]File:[/blue] {input_path}")
    console.print(f"[blue]Total scopes:[/blue] {len(scope_infos)}")

    for scope_info in scope_infos:
        console.print(f"\n[green]Scope {scope_info.scope_id}[/green] ({scope_info.scope_type})")
        console.print(f"  Range: lines {scope_info.range.start.row + 1} - {scope_info.range.end.row + 1}")

        for identifier in scope_info.identifiers:
            console.print(f"  - {identifier.name} (line {identifier.line + 1}, {identifier.binding_type})")


if __name__ == "__main__":
    main()
