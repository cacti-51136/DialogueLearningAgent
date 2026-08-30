"""两级工具路由（doc/08：低成本粗筛 top-N → LLM 选定）。

本期实现第一阶段（规则/关键词粗筛）：依据 query 与工具名/描述的相关性打分，返回 top-N 候选。
第二阶段（LLM 在候选中选定）由编排层在调用 LLM 时附上候选 schema 实现；此处仅产出候选集。
"""

from __future__ import annotations

from typing import List

from .protocol import Tool


def route(query: str, tools: List[Tool], top_n: int = 6) -> List[Tool]:
    """返回与 query 最相关的 top-N 工具候选（doc/08 粗筛）。"""
    q = (query or "").lower()
    scored: List[tuple] = []
    for t in tools:
        score = 0.0
        tokens = set(t.name.lower().split("_")) | set(t.description.lower().split())
        for tok in tokens:
            if tok and tok in q:
                score += 1.0
        if score == 0.0:
            score = 0.05  # 兜底：保证至少能返回候选供 LLM 决策
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_n]]
