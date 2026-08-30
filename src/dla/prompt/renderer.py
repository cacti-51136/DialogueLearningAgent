"""关键词 → 自然语言片段（doc/01 §5 prompt/renderer.py）。

把三层权重快照渲染为可读片段，供 assembler 组装进 System Prompt。渲染本身不判定裁剪，
裁剪由 assembler 依据预算执行。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..core.models import Layer
from ..keywords.lexicon import Lexicon


def render_segments(
    weights: Dict[str, float], lexicon: Lexicon, layer: Layer
) -> List[Tuple[str, float, str]]:
    """返回 ``(key, weight, text)`` 三元组列表（doc/02 §7 层内排序候选）。"""
    out: List[Tuple[str, float, str]] = []
    for key, w in weights.items():
        kw = lexicon.get(key)
        if kw is None:
            continue
        text = f"{kw.name}={w:.2f}"
        out.append((key, w, text))
    return out


def render_layer_block(
    weights: Dict[str, float], lexicon: Lexicon, layer: Layer, title: str
) -> str:
    segs = render_segments(weights, lexicon, layer)
    if not segs:
        return ""
    body = "；".join(text for _k, _w, text in segs)
    return f"【{title}】{body}"
