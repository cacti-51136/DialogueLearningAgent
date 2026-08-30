"""Token 预算（doc/01 §5 Prompt 预算 / D5 分层预算硬裁剪 / doc/11 整窗监控）。

提供与分词器无关的确定性 token 估算（``deterministic`` 中英文混合策略），以及
「按权重降序裁剪」工具——保证最重要的词一定进 Prompt，长尾被自然裁掉。
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

_CJK = re.compile(r"[一-鿿]")  # 中日韩统一表意文字


def estimate_tokens(text: str, tokenizer: str = "deterministic") -> int:
    """确定性 token 估算（doc/11 §2）。

    - 中文按字计（1 字 ≈ 1 token）。
    - 非中文字符按 ~4 字符 ≈ 1 token。
    - ``api`` 模式仅作占位（真实实现需接模型 tokenizer，本期未启用）。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    non_cjk = len(re.sub(r"\s", "", _CJK.sub("", text)))
    if tokenizer == "api":
        return max(1, round(len(text) / 3))
    return cjk + max(0, math.ceil(non_cjk / 4))


def budget_for_layer(total: int, ratio: float) -> int:
    return max(1, int(total * ratio))


def truncate_by_budget(
    segments: List[Tuple[str, float, str]],
    token_budget: int,
    estimate=estimate_tokens,
) -> List[str]:
    """按权重降序把片段塞进预算，超预算即停止（doc/01 D5）。

    ``segments`` 为 ``(key, weight, text)`` 三元组；返回被选中的文本列表（保持权重降序）。
    """
    segs = sorted(segments, key=lambda x: x[1], reverse=True)
    out: List[str] = []
    used = 0
    for _key, _w, text in segs:
        t = estimate(text)
        if out and used + t > token_budget:
            break
        out.append(text)
        used += t
    return out
