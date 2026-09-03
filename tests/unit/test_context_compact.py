"""上下文自动压缩与预算管控单测（doc/11 §3/§4/§9）。

覆盖：确定性 token 估算、fill_ratio 公式、window_parts 分项、紧凑协议各动作（驱逐 detail /
合并 epoch / 裁剪冷记忆 / 工具 schema 极简 / HARD 丢弃）、「达标即停」早停语义、以及
不修改入参（无损）。
"""

import types

from dla.prompt.context_compact import (
    ContextWindow,
    compact,
    fill_ratio,
    trigger_level_of,
    window_parts,
    window_tokens,
)


def _cfg(**overrides):
    base = dict(
        ctx_max_tokens=128000,
        ctx_reserve=0.20,
        ctx_warn_ratio=0.70,
        ctx_compact_ratio=0.85,
        ctx_hard_ratio=0.95,
        ctx_auto_compact=True,
        ctx_summary_compact_after=40,
        ctx_epoch_merge_n=10,
        ctx_epoch_max_chars=200,
        ctx_cold_trim_top_m=2,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _win(system="", history=None, cold=None, detail=None, tool="", current=""):
    return ContextWindow(
        system_prompt=system,
        history=history or [],
        cold_memory=cold or [],
        detail_blocks=detail or [],
        tool_schema=tool,
        current_user_msg=current,
    )


# ---- 确定性 token 估算 ----
def test_estimate_is_deterministic_per_char():
    from dla.prompt.budget import estimate_tokens

    # 纯中文：每字 1 token
    assert estimate_tokens("你好世界") == 4
    # 纯英文：约 4 字符 1 token
    assert estimate_tokens("hello") == 2  # 5 字符 → ceil(5/4)=2
    # 空串
    assert estimate_tokens("") == 0


def test_fill_ratio_formula():
    cfg = _cfg(ctx_max_tokens=1000, ctx_reserve=0.20)
    # budget_in = 1000 * 0.8 = 800
    win = _win(system="a" * 800)  # 800 非中文 char → ceil(800/4)=200 tokens
    # 估算 200 / 800 = 0.25
    assert abs(fill_ratio(win, cfg) - 0.25) < 0.02


def test_window_tokens_matches_fill_ratio_numerator():
    """window_tokens 与 fill_ratio 必须同口径（doc/11 §8.1 可观测字段自洽）。"""
    cfg = _cfg(ctx_max_tokens=1000, ctx_reserve=0.20)  # budget_in = 800
    win = _win(system="核心", history=["h1", "h2"], cold=["c1"], detail=["d1"], tool="t", current="问")
    total = window_tokens(win)
    assert total == sum(window_parts(win, cfg).values())
    # 恒等式：window_tokens / budget == fill_ratio
    assert abs(total / 800 - fill_ratio(win, cfg)) < 1e-9


def test_trigger_level_of_staged_thresholds():
    cfg = _cfg(ctx_warn_ratio=0.70, ctx_compact_ratio=0.85, ctx_hard_ratio=0.95)
    assert trigger_level_of(0.50, cfg) == "NONE"
    assert trigger_level_of(0.70, cfg) == "WARN"      # 下边界含入
    assert trigger_level_of(0.84, cfg) == "WARN"
    assert trigger_level_of(0.85, cfg) == "COMPACT"
    assert trigger_level_of(0.94, cfg) == "COMPACT"
    assert trigger_level_of(0.95, cfg) == "HARD"
    assert trigger_level_of(1.50, cfg) == "HARD"


def test_window_parts_sum_equals_fill_numerator():
    cfg = _cfg()
    win = _win(system="核心", history=["h1", "h2"], cold=["c1"], detail=["d1"], tool="t", current="问")
    parts = window_parts(win, cfg)
    assert parts["system_core"] == 2  # 核心 = 2 中文
    total = sum(parts.values())
    # fill_ratio 的分子 = 各分项之和（不含预算分母）
    assert total == (
        parts["system_core"] + parts["summary_chain"] + parts["cold"]
        + parts["detail"] + parts["tool"] + parts["current"]
    )


# ---- 紧凑协议：各动作 ----
def test_compact_evicts_details():
    cfg = _cfg(ctx_max_tokens=200, ctx_reserve=0.2)  # budget_in=160，COMPACT=136，易触发
    win = _win(system="x" * 40, detail=["长片段内容" * 50], current="y")  # 约 250 tokens
    out, actions = compact(win, cfg)
    assert "evict_details" in actions
    assert out.detail_blocks == []
    # 驱逐后 token 下降
    assert fill_ratio(out, cfg) < fill_ratio(win, cfg)


def test_compact_epoch_merge():
    # 关键：epoch_max_chars 调小 → 合并后截断压缩，单轮合并即可压到 COMPACT 以下（达标即停）。
    cfg = _cfg(ctx_max_tokens=200, ctx_summary_compact_after=5, ctx_epoch_merge_n=3, ctx_epoch_max_chars=60)
    hist = [f"第{i}轮摘要：练习了过去式与发音要点" for i in range(12)]  # 12×~15 token > COMPACT
    win = _win(system="x" * 40, history=hist, current="y")
    out, actions = compact(win, cfg)
    assert "merge_epoch" in actions
    # 12 条 → 合并最旧 9 条成 1 条 epoch + 保留最近 3 条 = 4 条
    assert len(out.history) == 4
    assert any(h.startswith("[epoch]") for h in out.history)
    # 合并保留最旧内容（截断在 EPOCH_MAX_CHARS 内；原文已落冷库无损）
    epoch = next(h for h in out.history if h.startswith("[epoch]"))
    assert "第0轮摘要" in epoch
    # 合并后已压到 COMPACT 以下 → 不应触发 HARD 丢弃
    assert "drop_oldest_epoch" not in actions


def test_compact_trim_cold():
    cfg = _cfg(ctx_max_tokens=50, ctx_cold_trim_top_m=2)
    win = _win(system="x" * 40, cold=[f"冷记忆{i}：" + "历史对话要点较长内容" * 8 for i in range(5)], current="y")
    out, actions = compact(win, cfg)
    assert "trim_cold" in actions
    assert len(out.cold_memory) == 2


def test_compact_simplify_tool_schema():
    cfg = _cfg(ctx_max_tokens=28)
    win = _win(system="x" * 40, tool="可用工具：\n- recall_memory\n- calc", current="y")
    out, actions = compact(win, cfg)
    assert "simplify_tools" in actions
    assert "已启用" in out.tool_schema
    assert "recall_memory" not in out.tool_schema


def test_compact_hard_drops_oldest_when_still_over():
    # 一个极小的窗口：即便合并后单条仍超 HARD → 反复丢弃最旧 epoch
    cfg = _cfg(
        ctx_max_tokens=100, ctx_reserve=0.0,  # budget_in = 100，HARD=0.95 → 95 tokens
        ctx_compact_ratio=0.5,
        ctx_hard_ratio=0.95,
        ctx_summary_compact_after=2,
        ctx_epoch_merge_n=1,  # 每次只合并最近 1 条，制造"合并也压不下去"的极端情形
        ctx_epoch_max_chars=10000,  # 不截断，确保仍超
    )
    # 每条都极长，合并成一条 epoch 仍超 HARD
    long_item = "长摘要内容" * 200  # 约 800 tokens
    hist = [long_item for _ in range(6)]
    win = _win(system="", history=hist, current="")
    out, actions = compact(win, cfg)
    assert "drop_oldest_epoch" in actions
    # 历史被削减
    assert len(out.history) < len(hist)


# ---- 「达标即停」早停语义 ----
def test_compact_stops_once_under_compact():
    cfg = _cfg(ctx_max_tokens=250, ctx_reserve=0.2, ctx_summary_compact_after=5, ctx_epoch_merge_n=3)
    # budget_in=200，COMPACT=170；15 条摘要 ≈ 226 token > 170 触发
    hist = [f"摘要{i}：练习过去式与发音要点" for i in range(15)]
    win = _win(system="x" * 40, history=hist, current="y")
    out, actions = compact(win, cfg)
    assert fill_ratio(out, cfg) < cfg.ctx_compact_ratio
    # 至少发生了一次有效压缩
    assert actions
    # 「达标即停」：最终 fill 已低于 COMPACT，不出现多余的驱逐动作
    assert "evict_details" not in actions


# ---- 无损：不修改入参 ----
def test_compact_does_not_mutate_input():
    cfg = _cfg()
    hist = [f"摘要{i}内容" for i in range(12)]
    win = _win(system="x" * 40, history=hist, detail=["临时细节"], cold=["c1", "c2", "c3"])
    before_hist = list(win.history)
    before_detail = list(win.detail_blocks)
    compact(win, cfg)
    assert win.history == before_hist
    assert win.detail_blocks == before_detail
