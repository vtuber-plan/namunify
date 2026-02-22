"""Base plugin interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PluginContext:
    """Context passed to plugins during processing."""
    source_code: str
    file_path: Optional[Path] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Plugin(ABC):
    """Abstract base class for plugins."""

    name: str = "base_plugin"
    description: str = "Base plugin class"
    priority: int = 100  # Lower priority runs first

    @abstractmethod
    async def process(self, context: PluginContext) -> PluginContext:
        """Process the code and return updated context.

        Args:
            context: Current processing context

        Returns:
            Updated context with modifications
        """
        pass

    def should_run(self, context: PluginContext) -> bool:
        """Determine if this plugin should run.

        Args:
            context: Current processing context

        Returns:
            True if plugin should run
        """
        return True


class PluginChain:
    """Manages a chain of plugins to apply sequentially."""

    def __init__(self):
        self.plugins: list[Plugin] = []

    def add_plugin(self, plugin: Plugin) -> "PluginChain":
        """Add a plugin to the chain.

        Args:
            plugin: Plugin to add

        Returns:
            Self for chaining
        """
        self.plugins.append(plugin)
        # Sort by priority
        self.plugins.sort(key=lambda p: p.priority)
        return self

    async def run(self, context: PluginContext) -> PluginContext:
        """Run all plugins in sequence using reduce pattern.

        Args:
            context: Initial context

        Returns:
            Final context after all plugins
        """
        for plugin in self.plugins:
            if plugin.should_run(context):
                context = await plugin.process(context)
        return context

    def __or__(self, other: "PluginChain") -> "PluginChain":
        """Combine two plugin chains."""
        combined = PluginChain()
        combined.plugins = sorted(
            self.plugins + other.plugins,
            key=lambda p: p.priority,
        )
        return combined
