"""Tests for CLI processing pipeline order."""

from types import SimpleNamespace

import pytest

from namunify.config import Config
from namunify import cli


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
