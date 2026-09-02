"""热更新监听器（doc/08 §3.2/§3.3：文件监听 watch / 影子灰度 shadow）。

零依赖实现：后台线程周期性计算插件目录的「内容指纹」（各文件 mtime+size 聚合哈希），
指纹变化即触发 ``registry.reload``。模式：
- ``watch``：变更即原子替换生效快照；
- ``shadow``：变更先载入影子快照观测，不真正生效，需 ``promote_shadow`` 确认。

监听目标目录默认取自 ``settings.tools_plugin_dir``（相对项目根解析）。也支持显式指定目录。
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Callable, List, Optional

from .registry import ToolRegistry


def _dir_fingerprint(path: str) -> str:
    """聚合目录下所有 .py 文件的 mtime+size 为稳定指纹（无外部依赖）。"""
    h = hashlib.sha1()
    if not os.path.isdir(path):
        return h.hexdigest()
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            if fn.endswith(".py"):
                fp = os.path.join(root, fn)
                try:
                    st = os.stat(fp)
                    h.update(f"{fn}:{st.st_mtime_ns}:{st.st_size};".encode("utf-8"))
                except OSError:
                    continue
    return h.hexdigest()


class HotReloadWatcher:
    """后台监听插件目录，变更时按模式触发注册表热更新。"""

    def __init__(
        self,
        registry: ToolRegistry,
        plugin_dir: str,
        mode: str = "watch",  # watch | shadow
        interval: float = 1.0,
        on_reload: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._registry = registry
        self._dir = plugin_dir
        self._mode = mode
        self._interval = max(0.2, interval)
        self._on_reload = on_reload
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fp: Optional[str] = _dir_fingerprint(self._dir)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _tick(self) -> None:
        fp = _dir_fingerprint(self._dir)
        if self._last_fp is None:
            self._last_fp = fp
            return
        if fp == self._last_fp:
            return
        self._last_fp = fp
        shadow = self._mode == "shadow"
        result = self._registry.reload(shadow=shadow)
        if self._on_reload is not None:
            self._on_reload(result)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - 监听线程内部异常不应致死
                pass
            self._stop.wait(self._interval)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._last_fp = _dir_fingerprint(self._dir)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None

    def scan_once(self) -> dict:
        """手动触发一次检查（测试/调试用）。"""
        before = self._last_fp
        self._tick()
        changed = before != self._last_fp
        return {"checked": True, "changed": changed, "mode": self._mode}
