"""工具注册表（doc/08 ToolRegistry：原子快照 + 热更新 + 回滚）。

- 注册/注销工具时版本号自增。
- ``snapshot`` 冻结当前工具集，进行中对话持旧快照、新对话用新快照（doc/08 热更新不中断进行中对话）。
- 加载失败时 ``rollback`` 到 last-good 版本（doc/01 §10）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .protocol import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._version = 0
        self._snapshots: Dict[int, Dict[str, Tool]] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._version += 1

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._version += 1

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def snapshot(self) -> int:
        self._snapshots[self._version] = dict(self._tools)
        return self._version

    def get_snapshot(self, version: int) -> Dict[str, Tool]:
        return self._snapshots.get(version, dict(self._tools))

    def rollback(self, version: int) -> None:
        if version in self._snapshots:
            self._tools = dict(self._snapshots[version])
            self._version += 1
