"""Tests for CLI processing pipeline order."""

from types import SimpleNamespace

import pytest

from namunify.config import Config
from namunify import cli
from namunify.state import ProcessingState


@pytest.mark.asyncio
async def test_process_file_uniquify_before_beautify(monkeypatch, tmp_path):
    """process_file should run uniquify before beautify."""
    input_file = tmp_path / "sample.js"
    original_code = "const userName = 'x';\nconsole.log(userName);\n"
    input_file.write_text(original_code, encoding="utf-8")

    call_order: list[str] = []
    observed: dict[str, object] = {}

    def fake_uniquify(
        source_code: str,
        output_path=None,
        timeout_seconds: int = 300,
        progress_callback=None,
        heartbeat_interval_seconds: float = 2.0,
    ) -> str:
        call_order.append("uniquify")
        observed["source_before_uniquify"] = source_code

        updated = source_code.replace("userName", "uniqueUserName")
        if output_path is not None:
            output_path.write_text(updated, encoding="utf-8")
        return updated

    def fake_beautify(input_path, output_file=None):
        call_order.append("beautify")
        observed["beautify_input"] = input_path
        return input_path

    monkeypatch.setattr(cli, "uniquify_binding_names", fake_uniquify)
    monkeypatch.setattr(cli, "beautify_js_file", fake_beautify)
    monkeypatch.setattr(cli, "parse_javascript", lambda source: SimpleNamespace(scopes={}, all_bindings=[]))
    monkeypatch.setattr(cli, "analyze_identifiers", lambda *args, **kwargs: [])

    config = Config()
    stats = await cli.process_file(input_file, config, resume=False)

    assert call_order == ["uniquify", "beautify"]
    assert observed["source_before_uniquify"] == original_code
    assert observed["beautify_input"] == input_file.with_suffix(".uniquified.js")
    assert stats["symbols_found"] == 0
    assert stats["symbols_renamed"] == 0


@pytest.mark.asyncio
async def test_process_file_resume_uses_cached_preprocess(monkeypatch, tmp_path):
    """Resuming with valid preprocess cache should skip uniquify/beautify reruns."""
    input_file = tmp_path / "sample.js"
    original_code = "const userName = 'x';\nconsole.log(userName);\n"
    input_file.write_text(original_code, encoding="utf-8")

    uniquified_file = tmp_path / "sample.uniquified.js"
    beautified_file = tmp_path / "sample.uniquified.beautified.js"
    uniquified_file.write_text("const uniqueUserName = 'x';\nconsole.log(uniqueUserName);\n", encoding="utf-8")
    beautified_file.write_text("const uniqueUserName = \"x\";\nconsole.log(uniqueUserName);\n", encoding="utf-8")

    source_sha256 = cli._compute_sha256(original_code)
    state = ProcessingState(
        input_file=str(input_file),
        output_file=str(tmp_path / "out.js"),
        total_scopes=10,
        processed_scopes=3,
        all_renames={},
        scope_renames=[],
        created_at="2026-02-24T00:00:00",
        updated_at="2026-02-24T00:00:00",
        config={
            cli._PREPROCESS_CACHE_KEY: {
                "source_sha256": source_sha256,
                "enable_uniquify": True,
                "uniquified_file": str(uniquified_file),
                "beautified_file": str(beautified_file),
            }
        },
        status="in_progress",
    )

    class DummyStateManager:
        def __init__(self):
            self._current_state = state

        def has_state(self, _input_file):
            return True

        def load_state(self, _input_file):
            return state

        def save_state(self, _state):
            self._current_state = _state

        def clear_state(self, _input_file=None):
            self._current_state = None

    dummy_state_manager = DummyStateManager()
    monkeypatch.setattr(cli, "state_manager", dummy_state_manager)
    monkeypatch.setattr(cli, "ask_resume", lambda *_args, **_kwargs: True)

    def fail_uniquify(*_args, **_kwargs):
        raise AssertionError("uniquify should be skipped")

    def fail_beautify(*_args, **_kwargs):
        raise AssertionError("beautify should be skipped")

    monkeypatch.setattr(cli, "uniquify_binding_names", fail_uniquify)
    monkeypatch.setattr(cli, "beautify_js_file", fail_beautify)
    monkeypatch.setattr(cli, "parse_javascript", lambda source: SimpleNamespace(scopes={}, all_bindings=[]))
    monkeypatch.setattr(cli, "analyze_identifiers", lambda *args, **kwargs: [])

    config = Config()
    stats = await cli.process_file(input_file, config, resume=True)

    assert stats["symbols_found"] == 0
    assert stats["symbols_renamed"] == 0
