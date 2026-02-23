"""Tests for CLI context building helpers."""

from namunify.cli import (
    _build_global_symbol_context,
    _build_global_symbol_snippet,
    _find_identifier_occurrence_lines,
)


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
