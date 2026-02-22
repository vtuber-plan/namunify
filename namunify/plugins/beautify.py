"""Beautify plugin for code formatting."""

import subprocess
import shutil
from namunify.plugins.base import Plugin, PluginContext
from rich.console import Console

console = Console()


class BeautifyPlugin(Plugin):
    """Plugin to beautify/format JavaScript code."""

    name = "beautify"
    description = "Beautify JavaScript code using prettier or js-beautify"
    priority = 10  # Run early in the chain

    def __init__(
        self,
        use_prettier: bool = True,
        fallback_to_js_beautify: bool = True,
    ):
        self.use_prettier = use_prettier
        self.fallback_to_js_beautify = fallback_to_js_beautify

    def should_run(self, context: PluginContext) -> bool:
        """Check if formatting tools are available."""
        return shutil.which("node") is not None

    async def process(self, context: PluginContext) -> PluginContext:
        """Beautify the code."""
        code = context.source_code

        if self.use_prettier:
            code = await self._format_with_prettier(code)

        if self.fallback_to_js_beautify and code == context.source_code:
            code = await self._format_with_js_beautify(code)

        context.source_code = code
        return context

    async def _format_with_prettier(self, code: str) -> str:
        """Format code using prettier."""
        try:
            result = subprocess.run(
                ["npx", "prettier", "--parser", "babel", "--print-width", "100"],
                input=code,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                console.print("[dim]Formatted with prettier[/dim]")
                return result.stdout
            else:
                console.print(f"[dim]Prettier failed: {result.stderr[:100]}[/dim]")

        except subprocess.TimeoutExpired:
            console.print("[dim]Prettier timed out[/dim]")
        except Exception as e:
            console.print(f"[dim]Prettier error: {e}[/dim]")

        return code

    async def _format_with_js_beautify(self, code: str) -> str:
        """Format code using js-beautify as fallback."""
        try:
            result = subprocess.run(
                ["npx", "js-beautify", "--indent-size", "2"],
                input=code,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                console.print("[dim]Formatted with js-beautify[/dim]")
                return result.stdout

        except subprocess.TimeoutExpired:
            console.print("[dim]js-beautify timed out[/dim]")
        except Exception as e:
            console.print(f"[dim]js-beautify error: {e}[/dim]")

        return code


class RemoveDeadCodePlugin(Plugin):
    """Plugin to remove dead code."""

    name = "remove_dead_code"
    description = "Remove unreachable/dead code"
    priority = 50

    async def process(self, context: PluginContext) -> PluginContext:
        """Remove dead code patterns."""
        # TODO: Implement dead code removal
        return context


class SimplifyExpressionsPlugin(Plugin):
    """Plugin to simplify complex expressions."""

    name = "simplify_expressions"
    description = "Simplify complex expressions"
    priority = 60

    async def process(self, context: PluginContext) -> PluginContext:
        """Simplify expressions."""
        # TODO: Implement expression simplification
        return context
