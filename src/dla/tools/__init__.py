"""工具插件化系统（doc/08）。

对外暴露契约与注册表，并提供内置插件构建工厂。
核心零依赖：仅用标准库；插件读取冷记忆库（同为 core 包），不引入任何重型依赖。
"""

from __future__ import annotations

from .executor import run_tool
from .loader import discover_all, discover_directory, discover_entry_points
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
    "discover_all",
    "discover_directory",
    "discover_entry_points",
    "build_builtin_registry",
]


def build_builtin_registry(include_entry_points: bool = True) -> ToolRegistry:
    """构建内置工具注册表（doc/08 §5）。

    扫描 ``dla.tools.plugins`` 目录 + entry_points 发行包（doc/08 §2），加载失败单插件隔离；
    返回已 ``load_from`` 的 :class:`ToolRegistry`（原子快照就绪，可进行热更新）。
    """
    reg = ToolRegistry()
    reg.load_from(discover_all(include_entry_points=include_entry_points))
    return reg
