"""JavaScript AST parsing using Babel via Node.js."""

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Path to the JS parser (in scripts directory)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_JS_PARSER_PATH = _PROJECT_ROOT / "scripts" / "js_parser.mjs"


@dataclass
class Position:
    """Position in source code."""
    row: int
    column: int

    def __lt__(self, other: "Position") -> bool:
        if self.row != other.row:
            return self.row < other.row
        return self.column < other.column

    def __le__(self, other: "Position") -> bool:
        return self == other or self < other


@dataclass
class Range:
    """Range in source code."""
    start: Position
    end: Position

    def contains(self, other: "Range") -> bool:
        """Check if this range completely contains another range."""
        return self.start <= other.start and self.end >= other.end

    def overlaps(self, other: "Range") -> bool:
        """Check if this range overlaps with another range."""
        return not (self.end < other.start or other.end < self.start)


@dataclass
class BindingIdentifier:
    """A binding identifier in JavaScript code."""
    name: str
    range: Range
    scope_id: Optional[str] = None
    binding_type: str = "variable"  # variable, parameter, function, class
    is_declaration: bool = False
    is_obfuscated: bool = False


@dataclass
class Scope:
    """A scope in JavaScript code."""
    scope_id: str
    range: Range
    scope_type: str  # program, function, arrow, class, block, catch
    bindings: list[BindingIdentifier] = field(default_factory=list)
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing JavaScript code."""
    source_code: str
    lines: list[str]
    scopes: dict[str, Scope]
    all_bindings: list[BindingIdentifier]


def check_node_available() -> bool:
    """Check if Node.js is available on the system."""
    return shutil.which("node") is not None


def ensure_babel_installed() -> bool:
    """Ensure Babel dependencies are installed in project root."""
    node_modules = _PROJECT_ROOT / "node_modules"
    if not node_modules.exists():
        console.print("[yellow]Installing Node.js dependencies...[/yellow]")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(f"[red]npm install failed: {result.stderr}[/red]")
                return False
            console.print("[green]Node.js dependencies installed[/green]")
        except Exception as e:
            console.print(f"[red]Failed to install dependencies: {e}[/red]")
            return False
    return True


def parse_javascript(source_code: str) -> ParseResult:
    """Parse JavaScript code using Babel and extract binding identifiers.

    Args:
        source_code: The JavaScript source code to parse

    Returns:
        ParseResult containing scopes and bindings
    """
    if not check_node_available():
        console.print("[yellow]Node.js not available, falling back to regex parsing[/yellow]")
        return _parse_with_regex(source_code)

    if not ensure_babel_installed():
        console.print("[yellow]Babel not available, falling back to regex parsing[/yellow]")
        return _parse_with_regex(source_code)

    # Write source to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source_code)
        temp_path = Path(f.name)

    try:
        # Run the JS parser
        result = subprocess.run(
            ["node", str(_JS_PARSER_PATH), str(temp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            console.print(f"[yellow]Babel parse error: {result.stderr}[/yellow]")
            return _parse_with_regex(source_code)

        # Parse JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            console.print(f"[yellow]Failed to parse Babel output: {e}[/yellow]")
            return _parse_with_regex(source_code)

        return _convert_babel_result(source_code, data)

    except subprocess.TimeoutExpired:
        console.print("[yellow]Babel parsing timed out[/yellow]")
        return _parse_with_regex(source_code)
    except Exception as e:
        console.print(f"[yellow]Babel parsing error: {e}[/yellow]")
        return _parse_with_regex(source_code)
    finally:
        temp_path.unlink(missing_ok=True)


def _convert_babel_result(source_code: str, data: dict) -> ParseResult:
    """Convert Babel parser output to ParseResult."""
    lines = source_code.split("\n")
    scopes: dict[str, Scope] = {}
    all_bindings: list[BindingIdentifier] = []

    # Convert scopes
    for scope_id, scope_data in data.get("scopes", {}).items():
        range_data = scope_data.get("range", {})
        scope = Scope(
            scope_id=scope_id,
            range=Range(
                start=Position(
                    row=range_data.get("start", {}).get("line", 0),
                    column=range_data.get("start", {}).get("column", 0),
                ),
                end=Position(
                    row=range_data.get("end", {}).get("line", 0),
                    column=range_data.get("end", {}).get("column", 0),
                ),
            ),
            scope_type=scope_data.get("type", "block"),
            parent_id=scope_data.get("parentId"),
            children=scope_data.get("children", []),
        )

        # Convert bindings
        for binding_data in scope_data.get("bindings", []):
            binding = BindingIdentifier(
                name=binding_data.get("name", ""),
                range=Range(
                    start=Position(
                        row=binding_data.get("line", 0),
                        column=binding_data.get("column", 0),
                    ),
                    end=Position(
                        row=binding_data.get("line", 0),
                        column=binding_data.get("column", 0) + len(binding_data.get("name", "")),
                    ),
                ),
                scope_id=scope_id,
                binding_type=binding_data.get("type", "variable"),
                is_declaration=True,
                is_obfuscated=binding_data.get("isObfuscated", False),
            )
            scope.bindings.append(binding)
            all_bindings.append(binding)

        scopes[scope_id] = scope

    return ParseResult(
        source_code=source_code,
        lines=lines,
        scopes=scopes,
        all_bindings=all_bindings,
    )


def _parse_with_regex(source_code: str) -> ParseResult:
    """Fallback regex-based parsing when Babel is not available."""
    import re

    lines = source_code.split("\n")
    scopes: dict[str, Scope] = {}
    all_bindings: list[BindingIdentifier] = []

    # Create a single program scope
    root_scope = Scope(
        scope_id="scope_0",
        range=Range(
            start=Position(0, 0),
            end=Position(len(lines) - 1, len(lines[-1]) if lines else 0),
        ),
        scope_type="program",
    )
    scopes["scope_0"] = root_scope

    # Pattern for variable declarations
    var_pattern = re.compile(r'\b(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)')
    func_pattern = re.compile(r'\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)')

    for i, line in enumerate(lines):
        for match in var_pattern.finditer(line):
            name = match.group(2)
            binding = BindingIdentifier(
                name=name,
                range=Range(
                    start=Position(i, match.start(2)),
                    end=Position(i, match.end(2)),
                ),
                scope_id="scope_0",
                binding_type="variable",
                is_declaration=True,
            )
            root_scope.bindings.append(binding)
            all_bindings.append(binding)

        for match in func_pattern.finditer(line):
            name = match.group(1)
            binding = BindingIdentifier(
                name=name,
                range=Range(
                    start=Position(i, match.start(1)),
                    end=Position(i, match.end(1)),
                ),
                scope_id="scope_0",
                binding_type="function",
                is_declaration=True,
            )
            root_scope.bindings.append(binding)
            all_bindings.append(binding)

    return ParseResult(
        source_code=source_code,
        lines=lines,
        scopes=scopes,
        all_bindings=all_bindings,
    )


def parse_file(file_path: Path) -> ParseResult:
    """Parse a JavaScript file.

    Args:
        file_path: Path to the JavaScript file

    Returns:
        ParseResult containing scopes and bindings
    """
    source_code = file_path.read_text(encoding="utf-8")
    return parse_javascript(source_code)
