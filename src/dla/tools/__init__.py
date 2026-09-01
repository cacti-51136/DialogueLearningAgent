"""工具插件化系统（doc/08）。

对外暴露契约与注册表，并提供内置插件构建工厂。
核心零依赖：仅用标准库；插件读取冷记忆库（同为 core 包），不引入任何重型依赖。
"""

from __future__ import annotations

from typing import List, Optional

from .executor import run_tool
from .loader import discover_plugins
from .protocol import Tool, ToolContext, ToolResult, validate_args
from .registry import ToolRegistry
from .router import route

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "validate_args",
    "ToolRegistry",
    "run_tool",
    "route",
    "discover_plugins",
    "build_builtin_registry",
]


def build_builtin_registry() -> ToolRegistry:
    """构建内置工具注册表（doc/08 §5）。

    优先用 ``discover_plugins`` 扫描 ``dla.tools.plugins`` 下所有 ``TOOL`` 实例；
    任一插件加载失败不影响其他。返回已注册的 :class:`ToolRegistry`。
    """
    reg = ToolRegistry()
    for tool in discover_plugins():
        reg.register(tool)
    return reg
