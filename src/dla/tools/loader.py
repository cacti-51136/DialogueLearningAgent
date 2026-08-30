"""插件扫描加载器（doc/08：插件扫描 / entry_points 加载 / 静态校验）。

扫描 ``src/dla/tools/plugins`` 下各子包，读取其导出的 ``TOOL`` 实例（doc/08 首个内置插件 recall_memory）。
加载失败的单插件不影响其他插件（隔离容错）。
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

from .protocol import Tool

_PLUGINS_PKG = "src.dla.tools.plugins"


def discover_plugins(plugins_pkg: str = _PLUGINS_PKG) -> List[Tool]:
    tools: List[Tool] = []
    try:
        pkg = importlib.import_module(plugins_pkg)
    except ImportError:
        return tools
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            m = importlib.import_module(f"{plugins_pkg}.{mod.name}")
        except Exception:  # noqa: BLE001 - 单插件失败隔离
            continue
        t = getattr(m, "TOOL", None)
        if isinstance(t, Tool):
            tools.append(t)
    return tools
