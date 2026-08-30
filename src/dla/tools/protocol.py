"""工具插件协议（doc/08 Tool 契约）。

统一 ``Tool`` 数据结构：``name`` / ``description`` / ``parameters``（JSON Schema 片段）/
``run``（处理函数）/ ``dangerous`` 标志（危险工具默认禁用，doc/08）。``ToolResult`` 为执行结果，
``ToolContext`` 为执行期注入的上下文（仓储 / 配置 / 会话等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolContext:
    session_id: str
    repo: Any = None
    settings: Any = None
    recent_summaries: List[str] = field(default_factory=list)
    recent_messages: List[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema 片段
    run: Callable[[dict, ToolContext], ToolResult]
    dangerous: bool = False
