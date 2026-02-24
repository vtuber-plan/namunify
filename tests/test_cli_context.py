"""Tests for CLI context building helpers."""

from namunify.cli import (
    _apply_prompt_size_limits,
    _build_global_symbol_context,
    _build_global_symbol_snippet,
    _count_binding_names,
    _find_identifier_occurrence_lines,
    _is_input_too_long_error,
    _is_retryable_llm_error,
    _normalize_rename_keys_for_unique_bindings,
    _parse_line_specific_key,
    _truncate_text_middle,
)
from namunify.core.parser import parse_javascript


class TestGlobalContextHelpers:
    """Tests for global-symbol context/snippet helpers."""

    def test_find_identifier_occurrence_lines_excludes_member_access(self):
        """Member access like obj.di should not be treated as identifier occurrence."""
        lines = [
            "var di = 1;",
            "obj.di = 2;",
            "console.log(di);",
            "const x = di + 1;",
        ]

        occurrences = _find_identifier_occurrence_lines(lines, "di")

        assert occurrences == [0, 2, 3]

    def test_build_global_symbol_context_includes_reference_lines(self):
        """Global context should include declaration info and reference line list."""
        lines = [f"const filler_{idx} = {idx};" for idx in range(1, 120)]
        lines[20] = "var di = create();"
        lines[58] = "use(di);"
        lines[95] = "return di;"

        context = _build_global_symbol_context(
            lines=lines,
            symbol="di",
            declaration_line=20,
            max_context_lines=40,
        )

        assert "Global symbol: di" in context
        assert "Declaration line: 21" in context
        assert "Reference lines: 59, 96" in context
        assert "    21 | var di = create();" in context
        assert "    59 | use(di);" in context

    def test_build_global_symbol_snippet_contains_reference_sections(self):
        """Snippet should include declaration and reference sections."""
        lines = [
            "var di = init();",
            "noop();",
            "console.log(di);",
            "other(di);",
        ]

        snippet = _build_global_symbol_snippet(lines, "di", declaration_line=0)

        assert "Declaration:" in snippet
        assert "Reference 1 (line 3):" in snippet

    def test_truncate_text_middle(self):
        """Long text should be truncated with marker."""
        text = "A" * 200 + "B" * 200
        truncated = _truncate_text_middle(text, 120)
        assert len(truncated) <= 120
        assert "[truncated]" in truncated

    def test_apply_prompt_size_limits(self):
        """Context and snippets should respect size limits."""
        context = "x" * 1000
        snippets = {"a": "y" * 800}
        limited_context, limited_snippets = _apply_prompt_size_limits(
            context=context,
            snippets=snippets,
            max_context_chars=300,
            max_snippet_chars=200,
        )

        assert len(limited_context) <= 300
        assert len(limited_snippets["a"]) <= 200

    def test_detect_input_too_long_error(self):
        """Length-related error message should be detected."""
        err = Exception("Range of input length should be [1, 131072]")
        assert _is_input_too_long_error(err) is True

    def test_detect_retryable_llm_error(self):
        """429/Throttling should be treated as retryable."""
        err = Exception("Error code: 429 - {'message': 'Throttling: TPM'}")
        assert _is_retryable_llm_error(err) is True

    def test_parse_line_specific_key(self):
        """Line-specific rename key should parse only for `name:line`."""
        assert _parse_line_specific_key("a__u1:42") == ("a__u1", 42)
        assert _parse_line_specific_key("a__u1") is None
        assert _parse_line_specific_key("a__u1:x") is None

    def test_normalize_line_specific_renames_for_unique_bindings(self):
        """Unique binding names should fold `name:line` into plain `name`."""
        code = """
var a__u1 = 1;
function f(a__u2) { return a__u2; }
"""
        parse_result = parse_javascript(code)
        counts = _count_binding_names(parse_result)
        normalized = _normalize_rename_keys_for_unique_bindings(
            {"a__u1:2": "firstValue", "a__u2:3": "paramValue"},
            counts,
        )

        assert normalized["a__u1"] == "firstValue"
        assert normalized["a__u2"] == "paramValue"
        assert "a__u1:2" not in normalized
        assert "a__u2:3" not in normalized

    def test_normalize_line_specific_renames_keeps_non_unique_name(self):
        """Non-unique names should keep line-specific keys."""
        code = """
var a = 1;
function f(a) { return a; }
"""
        parse_result = parse_javascript(code)
        counts = _count_binding_names(parse_result)
        normalized = _normalize_rename_keys_for_unique_bindings(
            {"a:2": "globalValue", "a:3": "paramValue"},
            counts,
        )

        assert normalized["a:2"] == "globalValue"
        assert normalized["a:3"] == "paramValue"
