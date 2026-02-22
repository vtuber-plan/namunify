"""Webcrack integration for unpacking webpack bundles."""

import asyncio
import shutil
import subprocess
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


async def check_webcrack_installed() -> bool:
    """Check if webcrack is installed via npm."""
    try:
        result = subprocess.run(
            ["npm", "list", "-g", "webcrack"],
            capture_output=True,
            text=True,
        )
        return "webcrack" in result.stdout
    except Exception:
        return False


async def install_webcrack() -> bool:
    """Install webcrack globally via npm."""
    console.print("[yellow]Installing webcrack globally...[/yellow]")
    try:
        result = subprocess.run(
            ["npm", "install", "-g", "webcrack"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]webcrack installed successfully![/green]")
            return True
        else:
            console.print(f"[red]Failed to install webcrack: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error installing webcrack: {e}[/red]")
        return False


def beautify_js_file(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """Beautify a JavaScript file using js-beautify.

    Args:
        input_file: Path to the JavaScript file
        output_file: Output path (default: input_file with .beautified.js suffix)

    Returns:
        Path to beautified file
    """
    if not check_node_available():
        console.print("[yellow]Node.js not available, skipping beautification[/yellow]")
        return input_file

    if output_file is None:
        output_file = input_file.with_suffix(".beautified.js")

    beautify_script = _SCRIPTS_DIR / "beautify.mjs"

    try:
        result = subprocess.run(
            ["node", str(beautify_script), str(input_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and output_file.exists():
            console.print(f"[dim]Beautified: {output_file}[/dim]")
            return output_file
        else:
            console.print(f"[yellow]Beautification warning: {result.stderr}[/yellow]")
            return input_file

    except subprocess.TimeoutExpired:
        console.print("[yellow]Beautification timed out[/yellow]")
        return input_file
    except Exception as e:
        console.print(f"[yellow]Beautification error: {e}[/yellow]")
        return input_file


async def unpack_webpack(
    input_file: Path,
    output_dir: Optional[Path] = None,
    force_install: bool = False,
) -> Optional[Path]:
    """Unpack a webpack bundle using webcrack.

    Args:
        input_file: Path to the webpack bundle JavaScript file
        output_dir: Output directory for unpacked files (default: input_file.parent / "unpacked")
        force_install: Force install webcrack if not found

    Returns:
        Path to the output directory, or None if failed
    """
    if not check_node_available():
        console.print("[red]Node.js is required but not found. Please install Node.js first.[/red]")
        return None

    if not await check_webcrack_installed():
        if force_install:
            if not await install_webcrack():
                return None
        else:
            console.print(
                "[red]webcrack is not installed. Run with --install-webcrack or install manually: npm install -g webcrack[/red]"
            )
            return None

    if output_dir is None:
        output_dir = input_file.parent / "unpacked"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Unpacking webpack bundle: {input_file}[/blue]")

    try:
        # Run webcrack via npx to ensure it works
        result = subprocess.run(
            ["npx", "webcrack", str(input_file), "-o", str(output_dir)],
            capture_output=True,
            text=True,
            cwd=input_file.parent,
        )

        if result.returncode != 0:
            console.print(f"[yellow]webcrack warning: {result.stderr}[/yellow]")

        # Check if output files were created
        js_files = list(output_dir.rglob("*.js"))
        if js_files:
            console.print(f"[green]Successfully unpacked {len(js_files)} JavaScript files[/green]")
            return output_dir
        else:
            console.print("[yellow]No JavaScript files found in output[/yellow]")
            # If webcrack didn't produce output, return the original file's directory
            return input_file.parent

    except Exception as e:
        console.print(f"[red]Error running webcrack: {e}[/red]")
        return None


async def beautify_js(input_file: Path, output_file: Optional[Path] = None) -> Optional[Path]:
    """Beautify JavaScript file using prettier.

    Args:
        input_file: Path to the JavaScript file
        output_file: Output path (default: input_file with .beautified.js suffix)

    Returns:
        Path to beautified file, or None if failed
    """
    if not check_node_available():
        console.print("[red]Node.js is required but not found.[/red]")
        return None

    if output_file is None:
        output_file = input_file.with_suffix(".beautified.js")

    try:
        # Try using prettier if available
        result = subprocess.run(
            ["npx", "prettier", "--write", str(input_file)],
            capture_output=True,
            text=True,
            cwd=input_file.parent,
        )

        if result.returncode == 0:
            return input_file

        # Fallback to js-beautify
        result = subprocess.run(
            ["npx", "js-beautify", str(input_file), "-o", str(output_file)],
            capture_output=True,
            text=True,
            cwd=input_file.parent,
        )

        if result.returncode == 0 and output_file.exists():
            return output_file

        console.print("[yellow]Could not beautify file, using original[/yellow]")
        return input_file

    except Exception as e:
        console.print(f"[yellow]Beautification failed: {e}[/yellow]")
        return input_file
