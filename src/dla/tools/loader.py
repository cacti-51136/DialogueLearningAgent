"""插件扫描加载器（doc/08：插件目录扫描 / entry_points 发行包加载 / 静态校验）。

- 扫描 ``dla.tools.plugins`` 下各子包，读取其导出的 ``TOOL``（单个）或 ``TOOLS``（列表）实例。
- 同时支持通过 ``importlib.metadata`` 的 ``entry_points``（group=``dla.tools``）加载已 pip 安装的发行包插件。
- 加载失败的单插件不影响其他插件（隔离容错）；每个插件额外返回其 ``module`` 引用，供注册表热更新时原位 reload。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
from typing import List, Optional, Tuple

from .protocol import Tool

_PLUGINS_PKG = "dla.tools.plugins"
_ENTRY_GROUP = "dla.tools"
# 兼容两种导入上下文：源码直接运行时为 ``dla...``（src 在 path），
# 经 CLI（apps/cli/main.py）运行时为 ``src.dla...``（项目根在 path）。
_PLUGINS_PKG_CANDIDATES = (_PLUGINS_PKG, "src.dla.tools.plugins")


def _extract_tools(module) -> List[Tool]:
    """从插件模块抽取 ``Tool`` 实例（兼容 ``TOOL`` 单例与 ``TOOLS`` 列表）。"""
    tools: List[Tool] = []
    t = getattr(module, "TOOL", None)
    if isinstance(t, Tool):
        tools.append(t)
    for t2 in getattr(module, "TOOLS", []) or []:
        if isinstance(t2, Tool):
            tools.append(t2)
    return tools


def discover_directory(plugins_pkg: Optional[str] = None) -> List[Tuple[Tool, str, object]]:
    """扫描插件目录，返回 ``(tool, plugin_name, module)`` 三元组。

    未指定 ``plugins_pkg`` 时自动探测 ``dla.tools.plugins`` 与 ``src.dla.tools.plugins`` 两种导入形态。
    """
    candidates = [plugins_pkg] if plugins_pkg else list(_PLUGINS_PKG_CANDIDATES)
    for pkg_name in candidates:
        out: List[Tuple[Tool, str, object]] = []
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            name = mod.name
            try:
                m = importlib.import_module(f"{pkg_name}.{name}")
            except Exception:  # noqa: BLE001 - 单插件失败隔离
                continue
            for t in _extract_tools(m):
                out.append((t, name, m))
        # 仅在该候选确实抽出工具时才返回。
        # 反例（双导入树）：src 与项目根同时在 sys.path 时，``dla.*`` 与 ``src.dla.*``
        # 是两棵**互不相同**的模块树，各自的 ``Tool`` 类也不同。若本 loader 属于
        # ``src.dla`` 树，却先命中 ``dla.tools.plugins``，抽出的实例来自另一棵树，
        # isinstance(t, Tool) 恒为 False → out 为空。此时必须继续尝试下一个候选
        # （``src.dla.tools.plugins``）才能拿到同树的工具实例。
        if out:
            return out
    return []


def discover_entry_points(group: str = _ENTRY_GROUP) -> List[Tuple[Tool, str, object]]:
    """从已安装的 entry_points 发行包加载插件（doc/08 §2 插件即发行包）。"""
    out: List[Tuple[Tool, str, object]] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.10+ : entry_points().select(group=...); 旧版: dict 取值
        chosen = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    except Exception:  # noqa: BLE001
        return out
    for ep in chosen:
        try:
            obj = ep.load()
        except Exception:  # noqa: BLE001 - 单插件失败隔离
            continue
        plugin_name = ep.name
        if isinstance(obj, Tool):
            out.append((obj, plugin_name, obj))
        elif isinstance(obj, (list, tuple)):
            for t in obj:  # type: ignore[union-attr]
                if isinstance(t, Tool):
                    out.append((t, plugin_name, t))
        else:
            # 视为插件模块/包：抽取其 TOOL/TOOLS
            module = obj if hasattr(obj, "__path__") or hasattr(obj, "__file__") else None
            if module is not None:
                for t in _extract_tools(module):
                    out.append((t, plugin_name, module))
    return out


def discover_all(
    plugins_pkg: Optional[str] = None,
    include_entry_points: bool = True,
) -> List[Tuple[Tool, str, object]]:
    """合并目录插件与 entry_points 发行包插件。

    未指定 ``plugins_pkg`` 时交给 ``discover_directory`` 自动探测两种导入形态
    （``dla.tools.plugins`` / ``src.dla.tools.plugins``），兼容 CLI 与源码两种运行上下文。
    """
    out = discover_directory(plugins_pkg)
    if include_entry_points:
        out += discover_entry_points()
    return out


def reload_module(module) -> Optional[object]:
    """对已知插件模块做原位 reload（doc/08 热更新：从磁盘重新加载，不重启进程）。"""
    try:
        return importlib.reload(module)
    except Exception:  # noqa: BLE001
        return None
