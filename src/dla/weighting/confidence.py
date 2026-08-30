"""证据 → 置信度（doc/02 §4）。

纯函数模块。置信度（confidence）与权重（weight）是两个独立概念（doc/01 §2）：
- ``evidence_total``：证据强度和，带时间衰减。
- ``confidence``：conf = E / (E + K)，K 为平滑系数（``DLA_WEIGHT__PRIOR_STRENGTH``）。
"""

from __future__ import annotations

import math
from typing import Sequence

from ..core.models import Evidence


def half_life_lambda(half_life_hours: float) -> float:
    """半衰期 → 衰减常数 λ。"""
    return math.log(2.0) / max(half_life_hours, 1e-9)


def decayed_total(
    evidence: Sequence[Evidence], now: float, half_life_hours: float
) -> float:
    """对证据序列按时间衰减后求和（doc/02 §4 指数衰减）。"""
    lam = half_life_lambda(half_life_hours)
    total = 0.0
    for e in evidence:
        dt = max(0.0, now - e.timestamp)
        hours = dt / 3600.0
        total += e.intensity * math.exp(-lam * hours)
    return total


def confidence(e_total: float, prior_strength: float) -> float:
    """平滑置信度，输出 0~1。E 越大越接近 1。"""
    return e_total / (e_total + max(prior_strength, 1e-9))


def weight_from_conf(conf: float, salience: float = 1.0) -> float:
    """权重 = 置信度 × 显著度（doc/02 §3 的 w = conf × salience）。"""
    return max(0.0, min(1.0, conf * salience))
