"""本地启发式分析（零成本、即时，doc/01 D7 / doc/04 思维链②）。

职责：
1. ``extract_heuristics``：基于词典/长度等即时产出候选证据（与 LLM 抽取互补）。
2. ``detect_repetition``：回复重复/循环护栏（doc/01 §8 / doc/04 §2.3 ⑩ 重复护栏帧）：
   检测「与近 N 轮复读」与「单条自重复退化报文」，供编排层降级处理。

全部纯函数 / 标准库，无 LLM 依赖。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.models import Evidence, Layer
from ..keywords.lexicon import Lexicon

# 本地轻量情绪词典（与 FakeLLM 内置词典区分：这里是「零成本启发式」）
_HEURISTIC_MOOD = {
    "难": ("user_mood.frustrated", 0.6),
    "烦": ("user_mood.frustrated", 0.6),
    "挫败": ("user_mood.frustrated", 0.7),
    "急": ("user_temper.impatient", 0.55),
    "开心": ("user_mood.excited", 0.6),
    "喜欢": ("user_mood.excited", 0.55),
    "紧张": ("user_mood.anxious", 0.6),
}


def extract_heuristics(
    text: str, turn: int, now: float, lexicon: Lexicon
) -> List[Evidence]:
    """从文本即时产出候选证据（经白名单归一，doc/01 D11）。"""
    ev: List[Evidence] = []
    if not text:
        return ev
    for token, (key, intensity) in _HEURISTIC_MOOD.items():
        if token in text:
            canon = lexicon.normalize(key)
            if canon is None:
                continue  # 不在白名单，丢弃（反哺词库用日志，由调用方记）
            ev.append(
                Evidence(
                    key=canon,
                    intensity=intensity,
                    timestamp=now,
                    source="heuristic",
                    turn=turn,
                    raw=text[:50],
                )
            )
    return ev


def _char_ngrams(s: str, n: int = 2) -> set:
    s = s.strip()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _max_ngram_ratio(text: str, n: int = 2) -> float:
    """单条文本内最高频 n-gram 占比（检测退化报文，如刷屏/半句卡死）。"""
    if not text:
        return 0.0
    if len(text) >= n:
        cnt = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    else:
        cnt = Counter([text])
    total = sum(cnt.values())
    return max(cnt.values()) / total if total else 0.0


def detect_repetition(
    reply: str,
    recent_replies: Sequence[str],
    sim_threshold: float = 0.75,
    self_ngram_ratio: float = 0.5,
    n: int = 2,
) -> Tuple[bool, Dict[str, float]]:
    """检测回复是否重复/退化（doc/01 §8 护栏）。

    返回 (是否命中, 指标字典)。命中条件：与近 N 轮任一回复相似度 ≥ 阈值，
    或单条自重复 n-gram 占比 ≥ 阈值。
    """
    reply_clean = re.sub(r"<turn_summary>.*?</turn_summary>", "", reply, flags=re.S)
    recent = [
        re.sub(r"<turn_summary>.*?</turn_summary>", "", r, flags=re.S) for r in recent_replies
    ]
    max_sim = max((_jaccard(_char_ngrams(reply_clean, n), _char_ngrams(r, n)) for r in recent), default=0.0)
    ratio = _max_ngram_ratio(reply_clean, n)
    hit = (max_sim >= sim_threshold) or (ratio >= self_ngram_ratio)
    return hit, {"max_sim": round(max_sim, 3), "self_ngram_ratio": round(ratio, 3)}
