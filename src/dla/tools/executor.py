"""工具执行器（doc/08：派发执行 + 超时熔断 + 降级）。"""

from __future__ import annotations

import threading
from typing import Any

from .protocol import Tool, ToolContext, ToolResult


def run_tool(tool: Tool, args: dict, ctx: ToolContext, timeout: float = 10.0) -> ToolResult:
    """执行工具，超时则降级为失败结果（doc/08 超时熔断）。"""
    holder: dict[str, ToolResult] = {}

    def _run() -> None:
        try:
            holder["r"] = tool.run(args, ctx)
        except Exception as e:  # noqa: BLE001
            holder["r"] = ToolResult(ok=False, error=f"工具执行异常: {e}")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return ToolResult(ok=False, error=f"工具 {tool.name} 执行超时")
    return holder.get("r", ToolResult(ok=False, error="无返回"))
