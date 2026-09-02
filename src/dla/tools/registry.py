"""工具注册表（doc/08 ToolRegistry：原子快照 + 热更新 + 回滚 + 影子灰度）。

设计要点（doc/08 §3）：
- ``snapshot()`` 返回当前工具集的**只读副本引用**；进行中对话在轮次开始处捕获该引用，
  之后即便发生热更新（替换指针），正在跑的对话仍用旧快照，互不干扰。
- ``reload(plugin_name)`` 构建新快照 → ``_validate`` 校验契约/冲突/签名 → 成功则原子替换指针，
  失败回滚到 ``_last_good``；绝不让非法插件替换当前生效快照。
- ``shadow``：先以影子身份观测（记录会被选中/执行，但不真正生效），``promote_shadow`` 确认后切换。
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

from .loader import discover_all, reload_module
from .protocol import Tool


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._snapshot: Dict[str, Tool] = {}
        self._sources: Dict[str, str] = {}          # tool_name -> plugin_name
        self._modules: Dict[str, object] = {}        # plugin_name -> module（用于原位 reload）
        self._disabled: set = set()                  # 当前禁用集合
        self._enabled_overrides: set = set()         # 经 set_enabled(True) 显式启用的非只读工具
        self._last_good: Dict[str, Tool] = {}
        self._shadow: Optional[Dict[str, Tool]] = None
        if tools:
            for t in tools:
                self._snapshot[t.name] = t
            # 危险/有副作用工具默认禁用，只读工具默认启用（doc/08 §5）
            self._disabled = {
                t.name for t in tools if not t.is_readonly and t.name not in self._enabled_overrides
            }
            self._last_good = dict(self._snapshot)

    # ---- 初始加载 ----
    def load_from(self, discovered: List[Tuple[Tool, str, object]]) -> None:
        """从 loader 的发现结果装载（合并目录插件与 entry_points 发行包）。"""
        for t, src, module in discovered:
            self._snapshot[t.name] = t
            self._sources[t.name] = src
            if module is not None:
                self._modules[src] = module
        # 重新计算禁用集：非只读工具默认禁用，已显式启用的不受此限（doc/08 §5）
        self._disabled = {
            t.name for t in self._snapshot.values()
            if not t.is_readonly and t.name not in self._enabled_overrides
        }
        self._last_good = dict(self._snapshot)

    # ---- 快照（原子性核心）----
    def snapshot(self) -> Dict[str, Tool]:
        """返回当前生效快照的只读副本；进行中对话持有此引用，热更新不影响之。"""
        return dict(self._snapshot)

    def all(self) -> List[Tool]:
        return [t for n, t in self._snapshot.items() if n not in self._disabled]

    def get(self, name: str) -> Optional[Tool]:
        if name in self._disabled:
            return None
        return self._snapshot.get(name)

    def is_enabled(self, name: str) -> bool:
        return name in self._snapshot and name not in self._disabled

    def is_registered(self, name: str) -> bool:
        return name in self._snapshot

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """启/停某工具（危险工具需显式 enable，doc/08 §5）。返回是否操作成功。"""
        if name not in self._snapshot:
            return False
        if enabled:
            self._enabled_overrides.add(name)
            self._disabled.discard(name)
        else:
            self._enabled_overrides.discard(name)
            self._disabled.add(name)
        return True

    def enabled_names(self) -> List[str]:
        return [n for n in self._snapshot if n not in self._disabled]

    def disabled_names(self) -> List[str]:
        return list(self._disabled)

    # ---- 热更新 ----
    def _build_candidate(self, plugin_name: Optional[str]) -> Dict[str, Tool]:
        """构建候选快照：从磁盘重新加载插件目录 / entry_points。

        - ``plugin_name=None``：全量重建（重新扫描并 reload 所有已知模块）。
        - 指定 ``plugin_name``：仅原位 reload 该插件模块，刷新其工具；其余保持。
        """
        candidate = dict(self._snapshot)  # 以当前为基线，保证未变插件稳定
        if plugin_name:
            module = self._modules.get(plugin_name)
            if module is None:
                return candidate
            reloaded = reload_module(module)
            if reloaded is not None:
                from .loader import _extract_tools

                for t in _extract_tools(reloaded):
                    candidate[t.name] = t
                    self._sources[t.name] = plugin_name
            return candidate

        # 全量：重新扫描目录 + entry_points，并对已知模块做原位 reload 以拾取磁盘改动
        discovered = discover_all()
        for t, src, module in discovered:
            candidate[t.name] = t
            self._sources[t.name] = src
            if module is not None:
                self._modules[src] = module
        return candidate

    def _recompute_disabled(self, snap: Dict[str, Tool]) -> None:
        """依据快照重新计算禁用集：非只读工具默认禁用，已显式启用的例外（doc/08 §5）。"""
        self._disabled = {
            n for n, t in snap.items() if not t.is_readonly and n not in self._enabled_overrides
        }

    @staticmethod
    def _validate(snap: Dict[str, Tool]) -> Optional[str]:
        """校验快照：契约完整性 + 名称唯一 + 可执行。返回错误串（None 表示通过）。"""
        seen = set()
        for name, t in snap.items():
            if not getattr(t, "name", None):
                return f"工具缺少 name 字段: {t!r}"
            if name != t.name:
                return f"工具 name 与键不一致: 键={name} name={t.name}"
            if not callable(getattr(t, "run", None)):
                return f"工具 {name} 缺少可执行 run"
            if not getattr(t, "description", ""):
                return f"工具 {name} 缺少 description（路由与 function-calling 共用）"
            if name in seen:
                return f"工具名冲突: {name}"
            seen.add(name)
        return None

    def reload(self, plugin_name: Optional[str] = None, shadow: bool = False) -> Dict[str, object]:
        """热更新：构建候选快照 → 校验 → 原子替换或回滚（doc/08 §3.2/§3.3）。

        返回 ``{"ok": bool, "shadow": bool, "error": Optional[str], "count": int}``。
        """
        candidate = self._build_candidate(plugin_name)
        err = self._validate(candidate)
        if err:
            # 回滚：保持当前生效快照不变（_last_good 本就是旧值，无需动作；此处防御性复位）
            self._snapshot = dict(self._last_good) if self._last_good else self._snapshot
            self._recompute_disabled(self._snapshot)
            return {"ok": False, "shadow": False, "error": err, "count": len(self._snapshot)}
        if shadow:
            self._shadow = candidate
            return {"ok": True, "shadow": True, "error": None, "count": len(candidate)}
        # 成功：记录旧快照为 last-good，再原子替换指针
        self._last_good = dict(self._snapshot)
        self._snapshot = candidate
        self._recompute_disabled(self._snapshot)
        return {"ok": True, "shadow": False, "error": None, "count": len(candidate)}

    def has_shadow(self) -> bool:
        return self._shadow is not None

    def promote_shadow(self) -> Dict[str, object]:
        """将影子快照提升为生效快照（确认观测无误后切换，doc/08 §3.3）。"""
        if self._shadow is None:
            return {"ok": False, "error": "当前无影子快照", "count": len(self._snapshot)}
        err = self._validate(self._shadow)
        if err:
            return {"ok": False, "error": err, "count": len(self._snapshot)}
        self._last_good = dict(self._snapshot)
        self._snapshot = self._shadow
        self._recompute_disabled(self._snapshot)
        self._shadow = None
        return {"ok": True, "error": None, "count": len(self._snapshot)}

    def discard_shadow(self) -> None:
        self._shadow = None
