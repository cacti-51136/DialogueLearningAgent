"""分析触发策略（doc/01 D7：分析器不是每轮都调 LLM）。

策略（冷启动强制 + 周期 + 启发式事件触发）以控制成本与延迟：
- 冷启动前 ``cold_start_turns`` 轮：每轮都分析（建立初始画像）。
- 之后每 ``period`` 轮分析一次（周期性校准）。
- 启发式事件（显式反馈 / 强情绪信号）可临时触发（由调用方结合 heuristics 判定）。

本类只负责「是否调用 LLM 分析」的判定，不含分析逻辑本身。
"""

from __future__ import annotations


class AnalysisTrigger:
    def __init__(
        self,
        cold_start_turns: int = 2,
        period: int = 3,
        enable_heuristics: bool = True,
    ) -> None:
        self.cold_start_turns = cold_start_turns
        self.period = max(1, period)
        self.enable_heuristics = enable_heuristics

    def should_analyze(self, turn: int, *, force_event: bool = False) -> bool:
        """判定本轮是否应调用 LLM 分析。

        turn 从 1 开始计数。force_event 由启发式事件（如用户显式反馈）触发。
        """
        if turn <= self.cold_start_turns:
            return True
        if force_event and self.enable_heuristics:
            return True
        if turn % self.period == 0:
            return True
        return False
