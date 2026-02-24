"""Code generation using Babel AST."""

import json
import queue
import subprocess
import shutil
import tempfile
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any, Callable, Optional

from rich.console import Console

console = Console()

# Path to scripts directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_PROGRESS_PREFIX = "__NAMUNIFY_PROGRESS__"


def check_node_available() -> bool:
    """Check if Node.js is available on the system."""
    return shutil.which("node") is not None


def generate_code(
    source_code: str,
    renames: dict[str, str],
    line_specific_renames: Optional[dict[str, str]] = None,
    beautify: bool = False,
    retain_lines: bool = True,
) -> str:
    """Generate renamed code using Babel AST.

    Args:
        source_code: Original source code
        renames: Dict mapping old names to new names
        line_specific_renames: Dict mapping "name:line" to new names
        beautify: Whether to beautify output after generation
        retain_lines: Whether to preserve line structure for stable positions

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

    try:
        return _generate_code_via_stdin(
            source_code=source_code,
            all_renames=all_renames,
            beautify=beautify,
            retain_lines=retain_lines,
        )
    except Exception as stdin_error:
        # Backward-compatible fallback path.
        console.print(f"[yellow]Babel stdin generation warning: {stdin_error}[/yellow]")
        try:
            return _generate_code_via_temp_files(source_code, all_renames)
        except Exception as file_mode_error:
            console.print(f"[yellow]Babel generation error: {file_mode_error}[/yellow]")
            return source_code


def _generate_code_via_stdin(
    source_code: str,
    all_renames: dict[str, str],
    beautify: bool,
    retain_lines: bool,
) -> str:
    """Generate code by streaming payload to Node script via stdin/stdout."""
    generate_script = _SCRIPTS_DIR / "generate.mjs"
    payload = {
        "code": source_code,
        "renames": all_renames,
        "options": {
            "beautify": beautify,
            "retainLines": retain_lines,
        },
    }

    result = subprocess.run(
        ["node", str(generate_script), "--stdin"],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "unknown error")

    generated = result.stdout
    if not generated and source_code:
        raise RuntimeError("empty stdout from generate.mjs")
    return generated


def _generate_code_via_temp_files(
    source_code: str,
    all_renames: dict[str, str],
) -> str:
    """Legacy file-based fallback for code generation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source_code)
        input_file = Path(f.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_renames, f)
        renames_file = Path(f.name)

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
            return output_file.read_text(encoding="utf-8")

        raise RuntimeError(result.stderr.strip() or "file output missing")
    finally:
        input_file.unlink(missing_ok=True)
        renames_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def uniquify_binding_names(
    source_code: str,
    output_path: Optional[Path] = None,
    timeout_seconds: int = 300,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    heartbeat_interval_seconds: float = 2.0,
) -> str:
    """Rename duplicated binding names so each binding name is globally unique.

    Args:
        source_code: Source code to process
        output_path: Optional path to save uniquified code
        timeout_seconds: Stall timeout in seconds without JS progress updates
        progress_callback: Optional callback to receive progress events
        heartbeat_interval_seconds: Interval between heartbeat events

    Returns:
        Source code with uniquely named bindings
    """
    if not source_code:
        return source_code

    if not check_node_available():
        console.print("[yellow]Node.js not available, skipping binding uniquification[/yellow]")
        return source_code

    # Use adaptive timeout for large files to reduce false timeout while Babel
    # is still making progress. This is an inactivity timeout.
    # Rough heuristic: 1s per 50k chars, bounded to [120, 1200].
    adaptive_timeout = min(1200, max(120, len(source_code) // 50000))
    effective_timeout = max(timeout_seconds, adaptive_timeout)
    heartbeat_interval = max(0.1, heartbeat_interval_seconds)

    # Write source to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source_code)
        input_file = Path(f.name)

    # Write output to temp file unless caller provides output path
    temp_output_file = input_file.with_suffix('.uniquified.js')
    output_file = output_path if output_path is not None else temp_output_file
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    callback_failed = False

    def emit_progress(event: str, **data: Any) -> None:
        nonlocal callback_failed
        if progress_callback is None or callback_failed:
            return

        payload = {"event": event, **data}
        try:
            progress_callback(payload)
        except Exception as callback_error:
            callback_failed = True
            console.print(
                f"[yellow]Uniquify progress callback error: {callback_error}[/yellow]"
            )

    last_js_progress_at = monotonic()

    def parse_progress_line(line: str) -> bool:
        """Parse structured progress line emitted by Node script."""
        nonlocal last_js_progress_at
        line = line.strip()
        if not line.startswith(_PROGRESS_PREFIX):
            return False

        raw_payload = line[len(_PROGRESS_PREFIX):].strip()
        if not raw_payload:
            return True

        try:
            progress = json.loads(raw_payload)
        except json.JSONDecodeError:
            return False

        if not isinstance(progress, dict):
            return True

        event_name = progress.get("event")
        if not isinstance(event_name, str) or not event_name:
            return True

        event_data = {k: v for k, v in progress.items() if k != "event"}
        last_js_progress_at = monotonic()
        emit_progress(event_name, source="js", **event_data)
        return True

    stream_queue: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
    reader_threads: list[Thread] = []

    def start_stream_reader(stream_name: str, stream: Any) -> None:
        if stream is None or not hasattr(stream, "readline"):
            return

        def reader() -> None:
            try:
                for line in iter(stream.readline, ""):
                    if line == "":
                        break
                    stream_queue.put((stream_name, line))
            finally:
                stream_queue.put((stream_name, None))

        thread = Thread(target=reader, daemon=True)
        thread.start()
        reader_threads.append(thread)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain_stream_queue() -> None:
        while True:
            try:
                stream_name, line = stream_queue.get_nowait()
            except queue.Empty:
                break

            if line is None:
                continue

            clean_line = line.rstrip("\r\n")
            if stream_name == "stderr" and parse_progress_line(clean_line):
                continue

            if stream_name == "stdout":
                stdout_lines.append(clean_line)
            else:
                stderr_lines.append(clean_line)

    def finalize_streams(process: Any) -> tuple[str, str]:
        if reader_threads:
            if hasattr(process, "wait"):
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass

            for _ in range(40):
                drain_stream_queue()
                if all(not thread.is_alive() for thread in reader_threads):
                    break
                sleep(0.05)

            drain_stream_queue()
            stdout_text = "\n".join(line for line in stdout_lines if line).strip()
            stderr_text = "\n".join(line for line in stderr_lines if line).strip()
            return stdout_text, stderr_text

        if hasattr(process, "communicate"):
            stdout_text, stderr_text = process.communicate()
            return (stdout_text or "").strip(), (stderr_text or "").strip()
        return "", ""

    try:
        uniquify_script = _SCRIPTS_DIR / "uniquify_bindings.mjs"
        command = ["node", str(uniquify_script), str(input_file), str(output_file)]
        start_time = monotonic()
        last_heartbeat_at = start_time

        emit_progress(
            "started",
            source="python",
            elapsed_seconds=0.0,
            timeout_seconds=effective_timeout,
            source_length=len(source_code),
            output_path=str(output_file),
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        start_stream_reader("stdout", getattr(process, "stdout", None))
        start_stream_reader("stderr", getattr(process, "stderr", None))

        while process.poll() is None:
            drain_stream_queue()
            now = monotonic()
            elapsed = now - start_time
            stalled_for = now - last_js_progress_at
            if stalled_for >= effective_timeout:
                process.kill()
                stdout, stderr = finalize_streams(process)
                emit_progress(
                    "timeout",
                    source="python",
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=effective_timeout,
                    timeout_type="progress_stalled",
                    stalled_seconds=round(stalled_for, 3),
                    stderr=(stderr or "").strip(),
                    stdout=(stdout or "").strip(),
                )
                console.print(
                    f"[yellow]Binding uniquification timed out after {effective_timeout}s without JS progress[/yellow]"
                )
                return source_code

            if now - last_heartbeat_at >= heartbeat_interval:
                last_heartbeat_at = now
                emit_progress(
                    "heartbeat",
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=effective_timeout,
                )

            sleep(0.1)

        stdout, stderr = finalize_streams(process)
        elapsed = monotonic() - start_time

        if process.returncode == 0 and output_file.exists():
            uniquified_code = output_file.read_text(encoding="utf-8")
            emit_progress(
                "completed",
                source="python",
                elapsed_seconds=round(elapsed, 3),
                timeout_seconds=effective_timeout,
                output_path=str(output_file),
            )
            if output_path is not None:
                console.print(f"[dim]Uniquified: {output_path}[/dim]")
            return uniquified_code

        emit_progress(
            "failed",
            source="python",
            elapsed_seconds=round(elapsed, 3),
            timeout_seconds=effective_timeout,
            returncode=process.returncode,
            stderr=(stderr or "").strip(),
            stdout=(stdout or "").strip(),
        )
        console.print(f"[yellow]Binding uniquification warning: {stderr}[/yellow]")
        return source_code

    except Exception as e:
        emit_progress("error", source="python", error=str(e))
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

    def __init__(
        self,
        source_code: str,
        beautify_after_generate: bool = False,
        retain_lines: bool = True,
    ):
        self.original_source = source_code
        self.current_source = source_code
        self.applied_renames: dict[str, str] = {}
        self.line_specific_renames: dict[str, str] = {}
        self.beautify_after_generate = beautify_after_generate
        self.retain_lines = retain_lines

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
            beautify=self.beautify_after_generate,
            retain_lines=self.retain_lines,
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
