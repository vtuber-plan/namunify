"""Tests for parser module."""

import shutil

import pytest

from namunify.core.parser import (
    BindingIdentifier,
    ParseResult,
    Position,
    Range,
    Scope,
    parse_javascript,
)

NODE_AVAILABLE = shutil.which("node") is not None


class TestPosition:
    """Tests for Position class."""

    def test_position_comparison(self):
        """Test position comparison operators."""
        p1 = Position(row=1, column=5)
        p2 = Position(row=2, column=0)
        p3 = Position(row=1, column=10)

        assert p1 < p2
        assert p1 < p3
        assert p1 <= p1
        assert p2 > p1


class TestRange:
    """Tests for Range class."""

    def test_range_contains(self):
        """Test range contains check."""
        r1 = Range(start=Position(0, 0), end=Position(10, 0))
        r2 = Range(start=Position(2, 0), end=Position(5, 0))

        assert r1.contains(r2)
        assert not r2.contains(r1)

    def test_range_overlaps(self):
        """Test range overlap check."""
        r1 = Range(start=Position(0, 0), end=Position(10, 0))
        r2 = Range(start=Position(5, 0), end=Position(15, 0))
        r3 = Range(start=Position(20, 0), end=Position(30, 0))

        assert r1.overlaps(r2)
        assert not r1.overlaps(r3)


class TestParseJavaScript:
    """Tests for parse_javascript function."""

    def test_parse_simple_code(self):
        """Test parsing simple JavaScript code."""
        code = "var a = 1;"
        result = parse_javascript(code)

        assert isinstance(result, ParseResult)
        assert result.source_code == code
        assert len(result.lines) == 1

    def test_parse_multiline_code(self):
        """Test parsing multiline JavaScript code."""
        code = """var a = 1;
var b = 2;
function test() {
    return a + b;
}"""
        result = parse_javascript(code)

        assert len(result.lines) == 5
        assert "scope_0" in result.scopes

    def test_parse_with_function(self):
        """Test parsing code with function declarations."""
        code = """
function foo() {
    var x = 1;
    return x;
}
"""
        result = parse_javascript(code)

        # Should have at least program scope
        assert len(result.scopes) >= 1
        assert "scope_0" in result.scopes

    def test_parse_empty_code(self):
        """Test parsing empty code."""
        result = parse_javascript("")

        assert result.source_code == ""
        assert len(result.lines) == 1  # Empty string still has one empty line
        assert len(result.scopes) == 1  # Program scope

    def test_parse_arithmetic(self):
        """Test parsing arithmetic expressions."""
        code = "var result = (1 + 2) * 3 / 4;"
        result = parse_javascript(code)

        assert result.source_code == code

    def test_parse_object(self):
        """Test parsing object literals."""
        code = """
var obj = {
    name: "test",
    value: 123,
    method: function() {
        return this.value;
    }
};
"""
        result = parse_javascript(code)

        assert "scope_0" in result.scopes

    @pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is required for Babel reference extraction")
    def test_parse_binding_reference_lines(self):
        """Test parser captures reference lines for bindings when available."""
        code = "var xi = Object.getOwnPropertyNames;\nfunction run(obj) { return xi(obj); }"
        result = parse_javascript(code)

        xi_bindings = [
            b for b in result.all_bindings
            if b.name == "xi" and b.range.start.row == 0
        ]
        assert xi_bindings
        # When Babel path.scope binding info is available, xi should reference line 2 (0-indexed: 1).
        assert 1 in xi_bindings[0].reference_lines


class TestScope:
    """Tests for Scope dataclass."""

    def test_scope_creation(self):
        """Test creating a Scope."""
        scope = Scope(
            scope_id="test",
            range=Range(start=Position(0, 0), end=Position(10, 0)),
            scope_type="function",
        )

        assert scope.scope_id == "test"
        assert scope.scope_type == "function"
        assert len(scope.bindings) == 0


class TestBindingIdentifier:
    """Tests for BindingIdentifier dataclass."""

    def test_binding_creation(self):
        """Test creating a BindingIdentifier."""
        binding = BindingIdentifier(
            name="myVar",
            range=Range(start=Position(1, 4), end=Position(1, 9)),
            scope_id="scope_1",
            binding_type="variable",
            is_declaration=True,
        )

        assert binding.name == "myVar"
        assert binding.binding_type == "variable"
        assert binding.is_declaration is True
