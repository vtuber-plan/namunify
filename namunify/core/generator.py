"""Code generation using Babel AST."""

import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Path to scripts directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


def check_node_available() -> bool:
    """Check if Node.js is available on the system."""
    return shutil.which("node") is not None


def generate_code(
    source_code: str,
    renames: dict[str, str],
    line_specific_renames: Optional[dict[str, str]] = None,
) -> str:
    """Generate renamed code using Babel AST.

    Args:
        source_code: Original source code
        renames: Dict mapping old names to new names
        line_specific_renames: Dict mapping "name:line" to new names

    Returns:
        Renamed source code
    """
    if not renames and not line_specific_renames:
        return source_code

    if not check_node_available():
        console.print("[yellow]Node.js not available, returning original code[/yellow]")
        return source_code

    # Merge all renames
    all_renames = dict(renames) if renames else {}
    if line_specific_renames:
        all_renames.update(line_specific_renames)

    if not all_renames:
        return source_code

    # Write source to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source_code)
        input_file = Path(f.name)

    # Write renames to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_renames, f)
        renames_file = Path(f.name)

    # Write output to temp file
    output_file = input_file.with_suffix('.generated.js')

    try:
        generate_script = _SCRIPTS_DIR / "generate.mjs"
        result = subprocess.run(
            ["node", str(generate_script), str(input_file), str(renames_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and output_file.exists():
            generated_code = output_file.read_text(encoding="utf-8")
            return generated_code
        else:
            console.print(f"[yellow]Babel generation warning: {result.stderr}[/yellow]")
            return source_code

    except subprocess.TimeoutExpired:
        console.print("[yellow]Babel generation timed out[/yellow]")
        return source_code
    except Exception as e:
        console.print(f"[yellow]Babel generation error: {e}[/yellow]")
        return source_code
    finally:
        # Cleanup temp files
        input_file.unlink(missing_ok=True)
        renames_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def uniquify_binding_names(
    source_code: str,
    output_path: Optional[Path] = None,
) -> str:
    """Rename duplicated binding names so each binding name is globally unique.

    Args:
        source_code: Source code to process
        output_path: Optional path to save uniquified code

    Returns:
        Source code with uniquely named bindings
    """
    if not source_code:
        return source_code

    if not check_node_available():
        console.print("[yellow]Node.js not available, skipping binding uniquification[/yellow]")
        return source_code

    # Write source to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source_code)
        input_file = Path(f.name)

    # Write output to temp file unless caller provides output path
    temp_output_file = input_file.with_suffix('.uniquified.js')
    output_file = output_path if output_path is not None else temp_output_file
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        uniquify_script = _SCRIPTS_DIR / "uniquify_bindings.mjs"
        result = subprocess.run(
            ["node", str(uniquify_script), str(input_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and output_file.exists():
            uniquified_code = output_file.read_text(encoding="utf-8")
            if output_path is not None:
                console.print(f"[dim]Uniquified: {output_path}[/dim]")
            return uniquified_code

        console.print(f"[yellow]Binding uniquification warning: {result.stderr}[/yellow]")
        return source_code

    except subprocess.TimeoutExpired:
        console.print("[yellow]Binding uniquification timed out[/yellow]")
        return source_code
    except Exception as e:
        console.print(f"[yellow]Binding uniquification error: {e}[/yellow]")
        return source_code
    finally:
        input_file.unlink(missing_ok=True)
        if output_path is None:
            temp_output_file.unlink(missing_ok=True)


async def format_with_prettier(code: str, file_path: Optional[Path] = None) -> str:
    """Format code using prettier.

    Args:
        code: Source code to format
        file_path: Optional file path for context

    Returns:
        Formatted code
    """
    if not shutil.which("node"):
        console.print("[yellow]Node.js not available, skipping prettier formatting[/yellow]")
        return code

    try:
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
    """Code generator for handling incremental renames using Babel AST."""

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
        """Apply renames using Babel AST and return updated code.

        Args:
            renames: Dict mapping old names to new names
            line_specific: Dict mapping "name:line" to new names

        Returns:
            Updated source code
        """
        self.applied_renames.update(renames)
        if line_specific:
            self.line_specific_renames.update(line_specific)

        # Generate code from original source with all renames applied
        # This ensures AST positions are correct
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
