"""词库自动演化（doc/02 §11 / doc/01 D13）：候选发现 + 升级审批。

设计要点：
- LLM 抽取的关键词分两类：命中白名单的 → 直接 feed 引擎；词表外的 → 进候选队列。
- 候选需达到「置信度 ≥ min_conf 且 观察次数 ≥ min_observed」才可被收编（doc/02 §11.4）。
- 首期**不开放自动采纳**（doc/02 §11.6）：候选仅供人工审核（``dla keyword review``）。

本模块只管理候选的生命周期，不修改词表/引擎状态（由调用方决定何时写入）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core.models import Layer
from .keywords.lexicon import Lexicon


@dataclass
class Candidate:
    key: str
    suggested_action: str  # add | update | delete
    intensity: float
    confidence: float
    source: str
    observed_count: int = 1
    rationale: str = ""


class CandidateRegistry:
    """追踪词表外候选的出现次数与置信度，决定是否可收编。"""

    def __init__(self, min_conf: float = 0.7, min_observed: int = 3) -> None:
        self.min_conf = min_conf
        self.min_observed = min_observed
        self._cands: Dict[str, Candidate] = {}

    def observe(self, key: str, intensity: float, confidence: float, source: str) -> None:
        if key in self._cands:
            c = self._cands[key]
            c.observed_count += 1
            c.intensity = max(c.intensity, intensity)
            c.confidence = max(c.confidence, confidence)
        else:
            self._cands[key] = Candidate(
                key=key,
                suggested_action="add",
                intensity=intensity,
                confidence=confidence,
                source=source,
                observed_count=1,
            )

    def ready_to_adopt(self, key: str) -> bool:
        c = self._cands.get(key)
        if not c:
            return False
        return c.confidence >= self.min_conf and c.observed_count >= self.min_observed

    def pending(self) -> List[Candidate]:
        return [c for c in self._cands.values() if not self.ready_to_adopt(c.key)]

    def adoptable(self) -> List[Candidate]:
        return [c for c in self._cands.values() if self.ready_to_adopt(c.key)]


def discover_from_extractions(
    extractions: List[dict], lexicon: Lexicon, registry: CandidateRegistry
) -> Tuple[List[dict], List[str]]:
    """分离已知词（直接可用）与未知词（进候选队列）。

    返回 (known_extractions, newly_seen_unknown_keys)。
    """
    known: List[dict] = []
    new_unknown: List[str] = []
    for e in extractions:
        key = e["key"]
        if lexicon.is_known(key):
            known.append(e)
        else:
            if key not in registry._cands:
                new_unknown.append(key)
            registry.observe(
                key,
                intensity=e.get("intensity", 0.5),
                confidence=e.get("confidence", 0.5),
                source=e.get("source", "llm"),
            )
    return known, new_unknown
