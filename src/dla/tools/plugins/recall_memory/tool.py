"""recall_memory：首个内置工具插件（doc/08 / doc/07 §4.7）。

从冷热记忆仓储取回当前会话最近的对话摘要片段，供 Agent 回顾上下文。
本期实现为「取回压缩摘要链最近 N 条」（doc/07 D22：默认历史即摘要链）；原文片段召回依赖冷库，纳入后续 doc/07 实现。
"""

from __future__ import annotations

from ...protocol import Tool, ToolContext, ToolResult


def _run(args: dict, ctx: ToolContext) -> ToolResult:
    top_k = int(args.get("top_k", 3))
    if ctx.repo is None:
        return ToolResult(ok=False, error="无存储后端")
    items = ctx.repo.list_recent_summaries(ctx.session_id, top_k)
    if not items:
        return ToolResult(ok=True, content="（暂无可召回的对话摘要）", metadata={"count": 0})
    text = "\n".join(f"- {x}" for x in items)
    return ToolResult(ok=True, content=text, metadata={"count": len(items)})


TOOL = Tool(
    name="recall_memory",
    description="取回当前会话最近的对话摘要片段，用于回顾上下文",
    parameters={
        "type": "object",
        "properties": {"top_k": {"type": "integer", "description": "取回条数，默认 3"}},
    },
    run=_run,
)
