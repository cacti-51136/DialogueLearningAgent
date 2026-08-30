"""时间衰减与冷却（doc/02 §4 衰减 / §6 稳定性冷却）。

纯函数。
"""

from __future__ import annotations

import math


def decay_factor(elapsed_seconds: float, half_life_hours: float) -> float:
    """返回 [0,1] 的衰减系数。"""
    hours = elapsed_seconds / 3600.0
    return math.exp(-math.log(2.0) / max(half_life_hours, 1e-9) * hours)


def passed_cooldown(last_turn: int, current_turn: int, cooldown_turns: int) -> bool:
    """是否满足冷却轮次（doc/02 §6 D6 冷却）。"""
    return (current_turn - last_turn) >= max(cooldown_turns, 0)


def hysteresis_passed(delta: float, threshold: float) -> bool:
    """滞回判定：变化量是否超过阈值（doc/02 §6 滞回 Δ）。"""
    return abs(delta) >= threshold
