"""对话历史冷热记忆子系统（doc/07）。

对外暴露：
- :class:`.embeddings.Embedder`：零依赖确定性 embedding（可注入真实 backbone）。
- :class:`.store.ColdMemoryStore`：SQLite 冷记忆库（检索 / 重排 / 软删除）。
- :func:`build_memory`：从（连接, settings）构造冷库，失败安全返回 ``None``。
"""

from __future__ import annotations

from typing import Optional

from .embeddings import Embedder
from .store import ColdMemoryStore

__all__ = ["Embedder", "ColdMemoryStore", "build_memory"]


def build_memory(conn, settings) -> Optional[ColdMemoryStore]:
    """用引擎已有的 SQLite 连接构造冷记忆库；任何异常都安全降级为 ``None``。

    降级后引擎行为与无记忆模式一致（``cold_memory`` 为空），不影响主流程。
    """
    try:
        return ColdMemoryStore(conn, settings)
    except Exception:  # noqa: BLE001 - 记忆是增强项，绝不应阻断对话主链路
        return None
