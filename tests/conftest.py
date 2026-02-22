"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def obfuscated_sample(fixtures_dir: Path) -> str:
    """Return contents of obfuscated sample file."""
    return (fixtures_dir / "obfuscated_sample.js").read_text()


@pytest.fixture
def webpack_bundle(fixtures_dir: Path) -> str:
    """Return contents of webpack bundle sample file."""
    return (fixtures_dir / "webpack_bundle.min.js").read_text()


@pytest.fixture
def simple_code() -> str:
    """Return simple JavaScript code for testing."""
    return """
var a = 1;
var b = 2;
function add(x, y) {
    return x + y;
}
var result = add(a, b);
"""


@pytest.fixture
def nested_scope_code() -> str:
    """Return code with nested scopes for testing."""
    return """
var outer = "value";

function process(data) {
    var temp = data.split("");
    return function transform(item) {
        var result = item.toUpperCase();
        return result;
    };
}

class Calculator {
    constructor(a, b) {
        this.x = a;
        this.y = b;
    }

    add() {
        return this.x + this.y;
    }
}
"""
