"""层间耦合规则推导（doc/02 §5.2 阶段一：确定性规则优先）。

纯函数。输入当前 L1 活跃集与 L2 权重，遍历耦合规则（doc/01 D4 确定性优先），产出：
- ``sets``：类别型维度应被置顶的值（如 role=tutor）。
- ``boosts``：程度型 L3 维度的累加权重。
- ``maps``：由 L2 情绪/脾性源词触发的「源 → 目标」映射（供 kw_agent_map 涌现沉淀，doc/03 §2.15）。

规则命中条件：``when.l1`` 任一命中活跃 L1，或 ``when.l2`` 任一命中权重>0 的 L2。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from ..config.loader import CouplingRule
from ..keywords.lexicon import Lexicon

# 仅这些 L2 源词（情绪/脾性）允许沉淀为可学习映射
_MAPPABLE_PREFIXES = ("user_mood.", "user_temper.")


@dataclass
class RuleEffect:
    sets: Dict[str, str] = field(default_factory=dict)  # dimension -> value_key
    boosts: Dict[str, float] = field(default_factory=dict)  # l3_key -> accumulated weight
    maps: List[Tuple[str, str, float]] = field(default_factory=list)  # (src_l2, dst_l3, weight)


def apply_rules(
    rules: Iterable[CouplingRule],
    active_l1: Set[str],
    l2_weights: Dict[str, float],
    lexicon: Lexicon,
) -> RuleEffect:
    """依据活跃 L1 / L2 权重匹配规则，叠加 set/boost 效果并沉淀可学习映射。"""
    eff = RuleEffect()
    for r in rules:
        hit = False
        if r.when_l1 and any(k in active_l1 for k in r.when_l1):
            hit = True
        l2_triggers: List[str] = []
        if r.when_l2:
            l2_triggers = [k for k in r.when_l2 if l2_weights.get(k, 0.0) > 0.0]
            if l2_triggers:
                hit = True
        if not hit:
            continue
        for dim, val in r.set_cmds:
            eff.sets[dim] = val
        for k, w in r.boost_cmds:
            eff.boosts[k] = eff.boosts.get(k, 0.0) + w
        # 仅当规则由情绪/脾性 L2 源词触发时，沉淀「源→目标」映射
        for src in l2_triggers:
            if not any(src.startswith(p) for p in _MAPPABLE_PREFIXES):
                continue
            for dst, w in r.boost_cmds:
                eff.maps.append((src, dst, w))
    return eff
