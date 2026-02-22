"""Plugin system for namunify."""

from namunify.plugins.base import Plugin, PluginChain
from namunify.plugins.beautify import BeautifyPlugin

__all__ = ["Plugin", "PluginChain", "BeautifyPlugin"]
