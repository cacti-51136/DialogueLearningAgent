"""维度内归一化与冲突消解（doc/02 §3 / §7）。

纯函数。给定 key→weight 原始权重与词库，按维度分组：
- 类别型维度（categorical）：取权重最高者为激活（输出 value_key=1.0），其余丢弃。
- 程度型维度（scalar）：**直接透传** 引擎算出的绝对权重（已落在 [0,1]），仅做安全 clamp。

设计说明（与原 doc/02 §7「权重占比归一化」的偏差）：
原实现对程度型维度做 sum 归一化（w / Σw）。这在「维度内只有一个成员」时会把
任何权重都变成 1.0，彻底抹掉引擎精心计算的强度，使得：
  (a) 单成员标量维度（empathy / patience / verbosity 等）永远输出 1.0；
  (b) 人格无法随对话「上升/下降」地演化（bench 验证 empathy 随受挫上升即失败）。

程度型权重在引擎里已是「强度」语义（基线 + 耦合 boost，clamp 到 [0,1]），下游
（prompt 组装、稳定性检测、人格演进）都依赖这个绝对值。因此改为**透传 + clamp**，
保留绝对强度；维度内多成员（如 tone.* / correction.* 互斥项）的相对强弱已由各自
权重天然体现，由下游按需取 max 或并列展示。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

from ..core.models import Layer
from ..keywords.lexicon import Lexicon


def resolve_layer(
    weights: Dict[str, float], lexicon: Lexicon, layer: Layer
) -> Dict[str, float]:
    """对一层原始权重做维度内冲突消解，返回规范权重。"""
    by_dim: Dict[str, Dict[str, float]] = defaultdict(dict)
    for key, w in weights.items():
        kw = lexicon.get(key)
        if kw is None:
            continue
        by_dim[kw.dimension][key] = w

    out: Dict[str, float] = {}
    for dim, members in by_dim.items():
        if not members:
            continue
        first = lexicon.get(next(iter(members)))
        if first is None:
            continue
        if first.ktype.value == "categorical":
            # 类别型：argmax 胜出，置 1.0，其余丢弃
            best = max(members, key=lambda k: members[k])
            out[best] = 1.0
        else:
            # 程度型：透传绝对权重，仅 clamp 到 [0,1]
            for k, w in members.items():
                out[k] = max(0.0, min(1.0, w))
    return out
