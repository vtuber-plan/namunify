"""State management for checkpoint/resume functionality."""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import threading

from rich.console import Console

console = Console()

# Default state directory
_STATE_DIR = Path(".namunify_state")


@dataclass
class ProcessingState:
    """State for checkpoint/resume functionality."""
    input_file: str
    output_file: str
    total_scopes: int
    processed_scopes: int
    all_renames: dict[str, str]
    scope_renames: list[dict]  # List of {scope_id, renames} for each processed scope
    created_at: str
    updated_at: str
    config: dict
    status: str  # "in_progress", "completed", "error"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        return cls(**data)


class StateManager:
    """Manages checkpoint state for resumable processing."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or _STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_state: Optional[ProcessingState] = None
        self._state_file: Optional[Path] = None

    def _get_state_file(self, input_file: Path) -> Path:
        """Get state file path for an input file."""
        # Use hash of absolute path as filename
        file_hash = hashlib.md5(str(input_file.resolve()).encode()).hexdigest()[:12]
        return self.state_dir / f"{input_file.stem}_{file_hash}_state.json"

    def has_state(self, input_file: Path) -> bool:
        """Check if there's existing state for the input file."""
        state_file = self._get_state_file(input_file)
        if not state_file.exists():
            return False

        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            state = ProcessingState.from_dict(data)
            return state.status == "in_progress" and state.processed_scopes < state.total_scopes
        except Exception:
            return False

    def load_state(self, input_file: Path) -> Optional[ProcessingState]:
        """Load existing state for the input file."""
        state_file = self._get_state_file(input_file)
        if not state_file.exists():
            return None

        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            state = ProcessingState.from_dict(data)
            self._current_state = state
            self._state_file = state_file
            return state
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load state file: {e}[/yellow]")
            return None

    def save_state(self, state: ProcessingState) -> None:
        """Save current state to file."""
        with self._lock:
            state.updated_at = datetime.now().isoformat()
            state_file = self._state_file or self._get_state_file(Path(state.input_file))
            state_file.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self._current_state = state
            self._state_file = state_file

    def create_state(
        self,
        input_file: Path,
        output_file: Path,
        total_scopes: int,
        config: dict,
    ) -> ProcessingState:
        """Create a new processing state."""
        now = datetime.now().isoformat()
        state = ProcessingState(
            input_file=str(input_file),
            output_file=str(output_file),
            total_scopes=total_scopes,
            processed_scopes=0,
            all_renames={},
            scope_renames=[],
            created_at=now,
            updated_at=now,
            config=config,
            status="in_progress",
        )
        self._state_file = self._get_state_file(input_file)
        self.save_state(state)
        return state

    def update_progress(
        self,
        scope_id: str,
        scope_index: int,
        renames: dict[str, str],
    ) -> None:
        """Update state with processed scope."""
        if self._current_state is None:
            return

        with self._lock:
            self._current_state.processed_scopes = scope_index + 1
            self._current_state.all_renames.update(renames)
            self._current_state.scope_renames.append({
                "scope_id": scope_id,
                "scope_index": scope_index,
                "renames": renames,
            })

        # Save to disk
        self.save_state(self._current_state)

    def mark_completed(self) -> None:
        """Mark processing as completed."""
        if self._current_state is None:
            return

        self._current_state.status = "completed"
        self.save_state(self._current_state)

    def mark_error(self, error_message: str) -> None:
        """Mark processing as errored."""
        if self._current_state is None:
            return

        self._current_state.status = f"error: {error_message}"
        self.save_state(self._current_state)

    def clear_state(self, input_file: Optional[Path] = None) -> None:
        """Clear state file."""
        if input_file:
            state_file = self._get_state_file(input_file)
            if state_file.exists():
                state_file.unlink()
        elif self._state_file and self._state_file.exists():
            self._state_file.unlink()

        self._current_state = None
        self._state_file = None

    def get_progress_summary(self, state: ProcessingState) -> str:
        """Get a human-readable progress summary."""
        percent = (state.processed_scopes / state.total_scopes * 100) if state.total_scopes > 0 else 0
        return (
            f"Progress: {state.processed_scopes}/{state.total_scopes} scopes ({percent:.1f}%), "
            f"{len(state.all_renames)} symbols renamed"
        )

    def list_pending_states(self) -> list[dict]:
        """List all pending/in-progress states."""
        pending = []
        for state_file in self.state_dir.glob("*_state.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                state = ProcessingState.from_dict(data)
                if state.status == "in_progress":
                    pending.append({
                        "file": state_file,
                        "input_file": state.input_file,
                        "progress": self.get_progress_summary(state),
                        "updated_at": state.updated_at,
                    })
            except Exception:
                continue
        return pending


def ask_resume(state: ProcessingState, state_manager: StateManager) -> bool:
    """Ask user whether to resume from checkpoint."""
    from rich.prompt import Confirm

    console.print(f"\n[yellow]Found incomplete processing state:[/yellow]")
    console.print(f"  Input: {state.input_file}")
    console.print(f"  Output: {state.output_file}")
    console.print(f"  {state_manager.get_progress_summary(state)}")
    console.print(f"  Last updated: {state.updated_at}")

    return Confirm.ask("\nResume from checkpoint?", default=True)
