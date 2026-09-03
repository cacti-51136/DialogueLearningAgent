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


def window_tokens(window: ContextWindow) -> int:
    """整窗 token 估算绝对值（doc/11 §8.1 可观测字段 tokens_before/tokens_after）。

    与 :func:`fill_ratio` 使用完全相同的口径，保证
    ``window_tokens(w) / budget == fill_ratio(w, cfg)`` 恒成立。
    """
    return (
        estimate_tokens(window.system_prompt)
        + estimate_tokens("\n".join(window.history))
        + estimate_tokens("\n".join(window.cold_memory))
        + estimate_tokens("\n".join(window.detail_blocks))
        + estimate_tokens(window.tool_schema)
        + estimate_tokens(window.current_user_msg)
    )


def fill_ratio(window: ContextWindow, cfg) -> float:
    """估算整窗 token 占比（doc/11 §3）。

    ``system_prompt`` 此处只含**核心段**（角色/用户肖像/交流风格），历史/冷记忆/细节/工具
    schema 为独立字段，各计一次，避免与已嵌进 system_prompt 的内容重复计数。
    """
    budget = max(1, int(cfg.ctx_max_tokens * (1.0 - cfg.ctx_reserve)))
    return window_tokens(window) / budget


def window_parts(window: ContextWindow, cfg) -> dict:
    """分项 token 估算（doc-11 §9 调试帧：system / summary链 / cold / detail / tool / 当前）。"""
    return {
        "system_core": estimate_tokens(window.system_prompt),
        "summary_chain": estimate_tokens("\n".join(window.history)),
        "cold": estimate_tokens("\n".join(window.cold_memory)),
        "detail": estimate_tokens("\n".join(window.detail_blocks)),
        "tool": estimate_tokens(window.tool_schema),
        "current": estimate_tokens(window.current_user_msg),
    }


def trigger_level_of(ratio: float, cfg) -> str:
    """按整窗占比判定触发档位（doc/11 §8.1 trigger_level 可观测字段）。

    阶梯：``>= hard`` → HARD；``>= compact`` → COMPACT；``>= warn`` → WARN；否则 NONE。
    手动触发（CLI ``--force``）由调用方直接传 ``MANUAL``。
    """
    if ratio >= getattr(cfg, "ctx_hard_ratio", 0.95):
        return "HARD"
    if ratio >= getattr(cfg, "ctx_compact_ratio", 0.85):
        return "COMPACT"
    if ratio >= getattr(cfg, "ctx_warn_ratio", 0.70):
        return "WARN"
    return "NONE"


def compact(window: ContextWindow, cfg) -> Tuple[ContextWindow, List[str]]:  # noqa: C901
    """执行一次紧凑协议，返回新窗口与已执行动作日志（doc/11 §4）。

    无损优先：被移除的 detail/历史摘要原文已落冷库，可经 recall_memory 取回；绝不丢未落盘信息。
    """
    w = copy.deepcopy(window)
    actions: List[str] = []
    compact_ratio = cfg.ctx_compact_ratio

    def _still_over() -> bool:
        return fill_ratio(w, cfg) >= compact_ratio

    # 1. 驱逐瞬态 detail_blocks（零损失）
    if w.detail_blocks and _still_over():
        w.detail_blocks = []
        actions.append("evict_details")
        if not _still_over():
            return w, actions

    # 2. 合并 epoch 摘要（无损：最旧若干条合并为 1 条；仍超则继续扩大合并窗口）
    while (
        len(w.history) > max(cfg.ctx_summary_compact_after, 1)
        and _still_over()
        and cfg.ctx_epoch_merge_n >= 1
    ):
        keep = max(cfg.ctx_epoch_merge_n, 1)
        if len(w.history) <= keep:
            break
        oldest = w.history[: len(w.history) - keep]
        recent = w.history[len(w.history) - keep :]
        merged = "；".join(oldest)
        if len(merged) > cfg.ctx_epoch_max_chars:
            merged = merged[: cfg.ctx_epoch_max_chars] + "…"
        w.history = [f"[epoch] {merged}"] + recent
        actions.append("merge_epoch" if not actions or actions[-1] != "merge_epoch" else "merge_epoch_again")
        if not _still_over():
            return w, actions

    # 3. 裁剪冷记忆（保留最近 COLD_TRIM_TOP_M）
    if len(w.cold_memory) > cfg.ctx_cold_trim_top_m and _still_over():
        w.cold_memory = w.cold_memory[-cfg.ctx_cold_trim_top_m :]
        actions.append("trim_cold")
        if not _still_over():
            return w, actions

    # 4. 工具 schema 极简（仅保留占位提示）
    if w.tool_schema and _still_over():
        w.tool_schema = "<tools: 已启用，按需调用本地工具>"
        actions.append("simplify_tools")
        if not _still_over():
            return w, actions

    # 5. HARD：仍超 → 反复丢弃最旧 epoch，直至低于 COMPACT 或历史耗尽
    while fill_ratio(w, cfg) >= cfg.ctx_hard_ratio and len(w.history) > 1:
        w.history = w.history[1:]
        if not actions or actions[-1] != "drop_oldest_epoch":
            actions.append("drop_oldest_epoch")
        if fill_ratio(w, cfg) < compact_ratio:
            break

    return w, actions
