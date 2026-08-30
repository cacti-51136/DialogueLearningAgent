"""上下文自动压缩与预算管控（doc/11）。

doc/11 核心：整窗 token 占比监控 + 阶梯阈值触发紧凑协议。本模块提供纯函数版
``ContextWindow`` / ``fill_ratio`` / ``compact``，不依赖 LLM（epoch 合并用拼接截断，
原文已落冷库可 recall，无损优先）。

紧凑协议（按损失从小到大，doc/11 §4）：
1. 驱逐瞬态 detail_blocks（零损失）
2. 合并 epoch 摘要（最旧若干条 summary 合并为 1 条，截断 EPOCH_MAX_CHARS）
3. 裁剪冷记忆（保留最近 COLD_TRIM_TOP_M）
4. 工具 schema 极简（仅保留占位提示）
5. HARD 仍超 → 丢弃最旧 epoch
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

from .budget import estimate_tokens


@dataclass
class ContextWindow:
    system_prompt: str = ""
    history: List[str] = field(default_factory=list)  # turn_summary 链（每条一元素）
    cold_memory: List[str] = field(default_factory=list)
    detail_blocks: List[str] = field(default_factory=list)
    tool_schema: str = ""
    current_user_msg: str = ""


def fill_ratio(window: ContextWindow, cfg) -> float:
    """估算整窗 token 占比（doc/11 §3）。"""
    total_tokens = (
        estimate_tokens(window.system_prompt)
        + estimate_tokens("\n".join(window.history))
        + estimate_tokens("\n".join(window.cold_memory))
        + estimate_tokens("\n".join(window.detail_blocks))
        + estimate_tokens(window.tool_schema)
        + estimate_tokens(window.current_user_msg)
    )
    budget = max(1, int(cfg.ctx_max_tokens * (1.0 - cfg.ctx_reserve)))
    return total_tokens / budget


def compact(window: ContextWindow, cfg) -> Tuple[ContextWindow, List[str]]:  # noqa: C901
    """执行一次紧凑协议，返回新窗口与已执行动作日志（doc/11 §4）。

    无损优先：被移除的 detail/历史摘要原文已落冷库，可经 recall_memory 取回；绝不丢未落盘信息。
    """
    w = copy.deepcopy(window)
    actions: List[str] = []

    # 1. 驱逐瞬态 detail_blocks（零损失）
    if w.detail_blocks:
        w.detail_blocks = []
        actions.append("evict_details")

    # 2. 合并 epoch 摘要（无损：最旧若干条合并为 1 条）
    if len(w.history) > cfg.ctx_summary_compact_after and cfg.ctx_epoch_merge_n >= 1:
        keep = max(cfg.ctx_epoch_merge_n, 1)
        oldest = w.history[: len(w.history) - keep]
        recent = w.history[len(w.history) - keep :]
        if oldest:
            merged = "；".join(oldest)
            if len(merged) > cfg.ctx_epoch_max_chars:
                merged = merged[: cfg.ctx_epoch_max_chars] + "…"
            w.history = [f"[epoch] {merged}"] + recent
            actions.append("merge_epoch")

    # 3. 裁剪冷记忆（保留最近 COLD_TRIM_TOP_M）
    if len(w.cold_memory) > cfg.ctx_cold_trim_top_m:
        w.cold_memory = w.cold_memory[-cfg.ctx_cold_trim_top_m :]
        actions.append("trim_cold")

    # 4. 工具 schema 极简
    if w.tool_schema:
        w.tool_schema = "<tools: 已启用，按需调用本地工具>"
        actions.append("simplify_tools")

    # 5. HARD：仍超 → 丢弃最旧 epoch
    if fill_ratio(w, cfg) >= cfg.ctx_hard_ratio and w.history:
        w.history = w.history[1:]
        actions.append("drop_oldest_epoch")

    return w, actions
