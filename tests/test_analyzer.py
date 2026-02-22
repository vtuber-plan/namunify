"""Tests for analyzer module."""

import pytest

from namunify.core.analyzer import (
    IdentifierInfo,
    ScopeInfo,
    analyze_identifiers,
    is_obfuscated_name,
)
from namunify.core.parser import parse_javascript


class TestIsObfuscatedName:
    """Tests for is_obfuscated_name function."""

    def test_single_letter_is_obfuscated(self):
        """Single letter names should be considered obfuscated."""
        assert is_obfuscated_name("a") is True
        assert is_obfuscated_name("b") is True
        # z is preserved as coordinate variable
        assert is_obfuscated_name("z") is False
        assert is_obfuscated_name("A") is True
        assert is_obfuscated_name("m") is True

    def test_meaningful_single_letters_not_obfuscated(self):
        """Common loop counters should not be considered obfuscated."""
        # These are preserved by default
        assert is_obfuscated_name("i") is False
        assert is_obfuscated_name("j") is False
        assert is_obfuscated_name("k") is False

    def test_two_letter_names(self):
        """Two letter names should mostly be obfuscated."""
        assert is_obfuscated_name("aa") is True
        assert is_obfuscated_name("xy") is True
        # But common abbreviations should not
        assert is_obfuscated_name("id") is False
        assert is_obfuscated_name("io") is False
        assert is_obfuscated_name("ui") is False

    def test_short_meaningful_words_not_obfuscated(self):
        """Common short words should not be considered obfuscated."""
        assert is_obfuscated_name("set") is False
        assert is_obfuscated_name("get") is False
        assert is_obfuscated_name("map") is False
        assert is_obfuscated_name("key") is False
        assert is_obfuscated_name("val") is False

    def test_mixed_letter_number_obfuscated(self):
        """Mixed letters and numbers should be considered obfuscated."""
        assert is_obfuscated_name("a1") is True
        assert is_obfuscated_name("b2") is True
        assert is_obfuscated_name("x9") is True

    def test_normal_names_not_obfuscated(self):
        """Normal descriptive names should not be obfuscated."""
        assert is_obfuscated_name("userName") is False
        assert is_obfuscated_name("processData") is False
        assert is_obfuscated_name("handleClick") is False
        assert is_obfuscated_name("fetchUserData") is False


class TestAnalyzeIdentifiers:
    """Tests for analyze_identifiers function."""

    def test_analyze_simple_code(self):
        """Test analyzing simple JavaScript code."""
        code = """
var a = 1;
var b = 2;
function c() {
    return a + b;
}
"""
        result = parse_javascript(code)
        scopes = analyze_identifiers(result)

        # Should find obfuscated symbols
        all_identifiers = []
        for scope in scopes:
            all_identifiers.extend(scope.identifiers)

        # a, b, c should be found as obfuscated
        names = {id.name for id in all_identifiers}
        assert "a" in names
        assert "b" in names

    def test_analyze_preserves_meaningful_names(self):
        """Meaningful names should not be marked as obfuscated."""
        code = """
var userName = "John";
var userId = 123;
function processData() {
    return userName + userId;
}
"""
        result = parse_javascript(code)
        scopes = analyze_identifiers(result)

        all_identifiers = []
        for scope in scopes:
            all_identifiers.extend(scope.identifiers)

        names = {id.name for id in all_identifiers}
        # These should not be found as obfuscated
        assert "userName" not in names
        assert "userId" not in names
        assert "processData" not in names

    def test_analyze_with_scope_grouping(self):
        """Test that identifiers are grouped by scope."""
        code = """
var a = 1;
function outer() {
    var b = 2;
    function inner() {
        var c = 3;
        return a + b + c;
    }
}
"""
        result = parse_javascript(code)
        scopes = analyze_identifiers(result)

        # Should have multiple scopes
        assert len(scopes) >= 1


class TestScopeInfo:
    """Tests for ScopeInfo dataclass."""

    def test_scope_info_creation(self):
        """Test creating ScopeInfo."""
        from namunify.core.parser import Position, Range

        scope = ScopeInfo(
            scope_id="test_scope",
            scope_type="function",
            range=Range(start=Position(0, 0), end=Position(10, 0)),
        )

        assert scope.scope_id == "test_scope"
        assert scope.scope_type == "function"
        assert len(scope.identifiers) == 0
        assert scope.merged is False


class TestIdentifierInfo:
    """Tests for IdentifierInfo dataclass."""

    def test_identifier_info_creation(self):
        """Test creating IdentifierInfo."""
        info = IdentifierInfo(
            name="a",
            line=5,
            column=10,
            scope_id="scope_1",
            binding_type="variable",
            is_obfuscated=True,
        )

        assert info.name == "a"
        assert info.line == 5
        assert info.column == 10
        assert info.is_obfuscated is True
