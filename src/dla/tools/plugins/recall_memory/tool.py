"""recall_memory：首个内置工具插件（doc/08 / doc/07 §4.7）。

从对话记忆取回历史内容供 Agent 回顾上下文。优先级：
1. 若传入 ``query`` 且冷记忆库可用（``ctx.memory``），走**语义召回**取回与 query 最相关的
   历史原文/摘要片段（doc/07 冷库已落地，此处即「按需取回原文」）；
2. 否则回退为取回当前会话最近的摘要链（doc/07 D22：默认历史即摘要链）。
只读、不修改记忆（doc/08 §5 边界）。
"""

from __future__ import annotations

from ...protocol import Tool, ToolContext, ToolResult

# 记忆召回触发线索（doc-08 §4 关键词命中路由）：命中任一即视为"意图回忆历史"
_RECALL_CUES = (
    "记得", "记得吗", "记得不", "之前", "提过", "说过", "提到过", "聊过",
    "我们说过", "你记得", "你之前", "上次", "recall", "remember", "memory",
)


def _can_handle(text: str, ctx: ToolContext) -> float:
    low = text.lower()
    if any(cue in low for cue in _RECALL_CUES):
        return 0.8  # 高于 tools_auto_threshold（默认 0.5），触发自动召回
    return 0.0


def _run(args: dict, ctx: ToolContext) -> ToolResult:
    top_k = int(args.get("top_k", 3))
    top_k = max(1, min(top_k, 10))
    query = str(args.get("query") or "").strip()

    # ① 语义召回：冷记忆库（doc/07）按需取回原文/摘要
    if ctx.memory is not None and query:
        try:
            items = ctx.memory.search(query, top_k=top_k, sim_threshold=0.0, scope="all")
        except Exception:  # noqa: BLE001 - 检索失败回退摘要
            items = []
        if items:
            text = "\n".join(f"{i + 1}. {it.display}" for i, it in enumerate(items))
            return ToolResult(
                ok=True,
                content=text,
                metadata={
                    "count": len(items),
                    "mode": "semantic",
                    "items": [
                        {"session_id": it.session_id, "turn": it.turn, "similarity": round(it.similarity, 3)}
                        for it in items
                    ],
                },
            )

    # ② 回退：近期摘要链
    if ctx.repo is not None:
        items = ctx.repo.list_recent_summaries(ctx.session_id, top_k)
        if items:
            text = "\n".join(f"- {x}" for x in items)
            return ToolResult(ok=True, content=text, metadata={"count": len(items), "mode": "summary"})

    return ToolResult(ok=True, content="（暂无可召回的对话记忆）", metadata={"count": 0})


TOOL = Tool(
    name="recall_memory",
    description=(
        "取回对话历史记忆供 Agent 回顾上下文：传入 query 时按语义召回最相关的历史原文/摘要；"
        "否则取回当前会话最近的摘要链。当用户问“你还记得…”、“之前我们说过…”、“我之前提到过…”时启用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "语义检索查询（可选）；为空则取回近期摘要链"},
            "top_k": {"type": "integer", "description": "取回条数，默认 3"},
        },
    },
    run=_run,
    can_handle=_can_handle,
)

