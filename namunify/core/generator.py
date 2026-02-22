"""Code generation and output formatting."""

import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def generate_code(
    source_code: str,
    renames: dict[str, str],
    line_specific_renames: Optional[dict[str, str]] = None,
) -> str:
    """Generate renamed code by applying symbol renames.

    Args:
        source_code: Original source code
        renames: Dict mapping old names to new names
        line_specific_renames: Dict mapping "name:line" to new names
            for disambiguating same names on different lines

    Returns:
        Renamed source code
    """
    lines = source_code.split("\n")
    line_specific_renames = line_specific_renames or {}

    # First, apply line-specific renames
    for key, new_name in line_specific_renames.items():
        if ":" in key:
            old_name, line_str = key.rsplit(":", 1)
            try:
                line_num = int(line_str) - 1  # Convert to 0-indexed
                if 0 <= line_num < len(lines):
                    lines[line_num] = _rename_in_line(lines[line_num], old_name, new_name)
            except ValueError:
                continue

    # Join lines back
    result = "\n".join(lines)

    # Then apply general renames (only for symbols not already renamed)
    for old_name, new_name in renames.items():
        # Use word boundary matching to avoid partial replacements
        pattern = r'\b' + re.escape(old_name) + r'\b'
        result = re.sub(pattern, new_name, result)

    return result


def _rename_in_line(line: str, old_name: str, new_name: str) -> str:
    """Rename a symbol in a single line with word boundary matching."""
    pattern = r'\b' + re.escape(old_name) + r'\b'
    return re.sub(pattern, new_name, line)


async def format_with_prettier(code: str, file_path: Optional[Path] = None) -> str:
    """Format code using prettier.

    Args:
        code: Source code to format
        file_path: Optional file path for context (determines parser)

    Returns:
        Formatted code
    """
    if not shutil.which("node"):
        console.print("[yellow]Node.js not available, skipping prettier formatting[/yellow]")
        return code

    try:
        # Use stdin for formatting
        result = subprocess.run(
            ["npx", "prettier", "--parser", "babel"],
            input=code,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return result.stdout
        else:
            console.print(f"[yellow]Prettier formatting failed: {result.stderr}[/yellow]")
            return code

    except subprocess.TimeoutExpired:
        console.print("[yellow]Prettier formatting timed out[/yellow]")
        return code
    except Exception as e:
        console.print(f"[yellow]Prettier formatting error: {e}[/yellow]")
        return code


def save_output(
    code: str,
    output_path: Path,
    create_dirs: bool = True,
) -> None:
    """Save code to file.

    Args:
        code: Source code to save
        output_path: Path to save to
        create_dirs: Whether to create parent directories
    """
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(code, encoding="utf-8")
    console.print(f"[green]Saved output to: {output_path}[/green]")


class CodeGenerator:
    """Code generator for handling incremental renames."""

    def __init__(self, source_code: str):
        self.original_source = source_code
        self.current_source = source_code
        self.applied_renames: dict[str, str] = {}
        self.line_specific_renames: dict[str, str] = {}

    def apply_renames(
        self,
        renames: dict[str, str],
        line_specific: Optional[dict[str, str]] = None,
    ) -> str:
        """Apply renames and return updated code.

        Args:
            renames: Dict mapping old names to new names
            line_specific: Dict mapping "name:line" to new names

        Returns:
            Updated source code
        """
        self.applied_renames.update(renames)
        if line_specific:
            self.line_specific_renames.update(line_specific)

        self.current_source = generate_code(
            self.original_source,
            self.applied_renames,
            self.line_specific_renames,
        )
        return self.current_source

    def get_current_source(self) -> str:
        """Get current state of the code."""
        return self.current_source

    def reset(self) -> None:
        """Reset to original source."""
        self.current_source = self.original_source
        self.applied_renames = {}
        self.line_specific_renames = {}

    async def format_and_save(
        self,
        output_path: Path,
        use_prettier: bool = True,
    ) -> None:
        """Format and save the current code.

        Args:
            output_path: Path to save to
            use_prettier: Whether to format with prettier
        """
        code = self.current_source
        if use_prettier:
            code = await format_with_prettier(code)

        save_output(code, output_path)
