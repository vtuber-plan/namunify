"""Tests for generator module."""

import json
import shutil
from pathlib import Path

import pytest

import namunify.core.generator as generator_module
from namunify.core.generator import (
    CodeGenerator,
    generate_code,
    uniquify_binding_names,
)
from namunify.core.parser import parse_javascript

NODE_AVAILABLE = shutil.which("node") is not None


class TestGenerateCode:
    """Tests for generate_code function."""

    def test_simple_rename(self):
        """Test simple variable renaming."""
        code = "var a = 1; console.log(a);"
        renames = {"a": "value"}

        result = generate_code(code, renames)

        assert "value" in result
        assert "var value = 1" in result

    def test_multiple_renames(self):
        """Test multiple variable renames."""
        code = "var a = 1; var b = 2; return a + b;"
        renames = {"a": "first", "b": "second"}

        result = generate_code(code, renames)

        assert "first" in result
        assert "second" in result

    def test_line_specific_rename(self):
        """Test line-specific renaming."""
        code = "var a = 1;\nvar a = 2;\nconsole.log(a);"
        line_specific = {"a:2": "second_a"}

        result = generate_code(code, {}, line_specific)

        lines = result.split("\n")
        assert "second_a" in lines[1]

    def test_word_boundary_matching(self):
        """Test that renaming respects word boundaries."""
        code = "var ab = 1; var abc = 2;"
        renames = {"ab": "first"}

        result = generate_code(code, renames)

        assert "first" in result
        assert "abc" in result  # Should not be renamed
        assert "firstc" not in result

    def test_rename_with_context(self):
        """Test renaming with surrounding context."""
        code = "var a = 1; var b = 2; console.log(a + b);"
        renames = {"a": "first", "b": "second"}

        result = generate_code(code, renames)

        assert "var first = 1" in result
        assert "var second = 2" in result
        assert "console.log(first + second)" in result

    def test_empty_renames(self):
        """Test with empty rename dict."""
        code = "var a = 1;"
        result = generate_code(code, {})

        assert result == code


class TestCodeGenerator:
    """Tests for CodeGenerator class."""

    def test_initialization(self):
        """Test CodeGenerator initialization."""
        code = "var a = 1;"
        gen = CodeGenerator(code)

        assert gen.original_source == code
        assert gen.current_source == code

    def test_apply_renames(self):
        """Test applying renames."""
        code = "var a = 1; var b = 2;"
        gen = CodeGenerator(code)

        gen.apply_renames({"a": "first"})
        assert "first" in gen.get_current_source()

        gen.apply_renames({"b": "second"})
        assert "second" in gen.get_current_source()

    def test_incremental_renames(self):
        """Test incremental rename application."""
        code = "var a = 1; var b = 2; var c = 3;"
        gen = CodeGenerator(code)

        gen.apply_renames({"a": "first"})
        first_result = gen.get_current_source()
        assert "first" in first_result

        gen.apply_renames({"b": "second"})
        second_result = gen.get_current_source()
        assert "first" in second_result
        assert "second" in second_result

    def test_reset(self):
        """Test resetting to original source."""
        code = "var a = 1;"
        gen = CodeGenerator(code)

        gen.apply_renames({"a": "value"})
        assert "value" in gen.get_current_source()

        gen.reset()
        assert gen.get_current_source() == code
        assert len(gen.applied_renames) == 0

    def test_line_specific_tracking(self):
        """Test line-specific rename tracking."""
        code = "var a = 1;\nvar a = 2;"
        gen = CodeGenerator(code)

        gen.apply_renames({}, {"a:2": "second_a"})
        assert "second_a" in gen.get_current_source()
        assert "a:2" in gen.line_specific_renames

    def test_combined_renames(self):
        """Test combining general and line-specific renames."""
        code = "var a = 1;\nvar a = 2;\nvar b = 3;"
        gen = CodeGenerator(code)

        gen.apply_renames(
            {"b": "value"},
            {"a:2": "second_a"},
        )

        result = gen.get_current_source()
        assert "value" in result
        assert "second_a" in result


class TestUniquifyProgress:
    """Tests for uniquify heartbeat/progress reporting."""

    def test_uniquify_emits_progress_events(self, monkeypatch):
        """Progress callback should receive started/heartbeat/completed events."""
        code = "function keep(alpha) { return alpha; }"
        progress_events = []

        class DummyProcess:
            def __init__(self):
                self.returncode = None
                self._poll_count = 0

            def poll(self):
                self._poll_count += 1
                if self._poll_count >= 4:
                    self.returncode = 0
                    return 0
                return None

            def communicate(self):
                return ("ok", "")

            def kill(self):
                self.returncode = -9

        def fake_popen(cmd, stdout=None, stderr=None, text=None, bufsize=None):
            output_file = Path(cmd[3])
            output_file.write_text(code, encoding="utf-8")
            return DummyProcess()

        monkeypatch.setattr(generator_module, "check_node_available", lambda: True)
        monkeypatch.setattr(generator_module.subprocess, "Popen", fake_popen)

        result = uniquify_binding_names(
            code,
            progress_callback=progress_events.append,
            heartbeat_interval_seconds=0.05,
            timeout_seconds=1,
        )

        assert result == code
        event_types = {event["event"] for event in progress_events}
        assert "started" in event_types
        assert "heartbeat" in event_types
        assert "completed" in event_types

    def test_uniquify_parses_js_progress_events(self, monkeypatch):
        """JS progress payloads should be parsed and surfaced in callback."""
        code = "function keep(alpha) { return alpha; }"
        progress_events = []

        class DummyStream:
            def __init__(self, lines):
                self._lines = list(lines)

            def readline(self):
                if self._lines:
                    return self._lines.pop(0)
                return ""

        class DummyProcess:
            def __init__(self, output_file: Path):
                self.returncode = None
                self._poll_count = 0
                self.stdout = DummyStream(["Generated: ok\n"])
                self.stderr = DummyStream([
                    (
                        f"{generator_module._PROGRESS_PREFIX}"
                        f"{json.dumps({'event': 'started', 'totalBindingsToRename': 2})}\n"
                    ),
                    (
                        f"{generator_module._PROGRESS_PREFIX}"
                        f"{json.dumps({'event': 'rename_progress', 'renamedBindings': 1, 'totalBindingsToRename': 2})}\n"
                    ),
                    (
                        f"{generator_module._PROGRESS_PREFIX}"
                        f"{json.dumps({'event': 'completed', 'renamedBindings': 2, 'totalBindingsToRename': 2})}\n"
                    ),
                ])
                output_file.write_text(code, encoding="utf-8")

            def poll(self):
                self._poll_count += 1
                if self._poll_count >= 3:
                    self.returncode = 0
                    return 0
                return None

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

        def fake_popen(cmd, stdout=None, stderr=None, text=None, bufsize=None):
            return DummyProcess(Path(cmd[3]))

        monkeypatch.setattr(generator_module, "check_node_available", lambda: True)
        monkeypatch.setattr(generator_module.subprocess, "Popen", fake_popen)

        result = uniquify_binding_names(
            code,
            progress_callback=progress_events.append,
            heartbeat_interval_seconds=5.0,
            timeout_seconds=1,
        )

        assert result == code
        js_progress = [
            event
            for event in progress_events
            if event.get("source") == "js" and event.get("event") == "rename_progress"
        ]
        assert js_progress
        assert js_progress[-1]["renamedBindings"] == 1
        assert js_progress[-1]["totalBindingsToRename"] == 2


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is required for Babel-based transforms")
class TestUniquifyBindingNames:
    """Tests for automatic binding-name uniquification."""

    def test_uniquify_same_names_across_scopes(self):
        """Bindings with same names in different scopes should become unique."""
        code = """
function first(a) {
  const b = a + 1;
  return b;
}

function second(a) {
  const b = a + 2;
  return b;
}
"""
        result = uniquify_binding_names(code)
        parse_result = parse_javascript(result)
        binding_names = [binding.name for binding in parse_result.all_bindings]

        assert len(binding_names) == len(set(binding_names))
        assert any("__u" in name for name in binding_names)

    def test_keep_unique_bindings_stable(self):
        """Already-unique bindings should not get rewritten."""
        code = "function run(alpha) { const beta = alpha + 1; return beta; }"
        result = uniquify_binding_names(code)
        parse_result = parse_javascript(result)
        binding_names = {binding.name for binding in parse_result.all_bindings}

        assert "run" in binding_names
        assert "alpha" in binding_names
        assert "beta" in binding_names

    def test_write_uniquified_output_file(self, tmp_path):
        """Uniquified intermediate result should be writable to a target file."""
        code = "function first(a){return a;} function second(a){return a;}"
        output_file = tmp_path / "sample.uniquified.js"

        result = uniquify_binding_names(code, output_path=output_file)

        assert output_file.exists()
        saved = output_file.read_text(encoding="utf-8")
        assert saved == result
        assert "__u" in saved
