"""核心领域模型（doc/01 §2 术语表 / doc/02 / doc/03）。

本模块**零外部依赖**，仅用标准库。它定义了三层关键词权重系统的原子数据结构：
``Keyword``（受控词）、``Evidence``（观测证据）、``KeywordState``（运行时状态）、
``WeightSnapshot``（每轮快照）、``TurnSummary``（压缩摘要链节点）等。

核心约束（doc/01 D11）：关键词必须命中白名单词表，模型自由造词一律丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Layer(str, Enum):
    """三层权重层。"""

    L1 = "l1"  # 功能场景层（场景 + 双方目标）
    L2 = "l2"  # 用户肖像层（预设 + 对话中分析出的用户描述）
    L3 = "l3"  # agent 肖像层（适合当前场景/目标的交流对象提示词）


class KeywordType(str, Enum):
    """关键词类型（doc/01 D1）。"""

    CATEGORICAL = "categorical"  # 类别型：维度内互斥，取 argmax
    SCALAR = "scalar"  # 程度型：连续值，维度内加权平均


@dataclass(frozen=True)
class Keyword:
    """受控词表中的一条关键词（不可变）。"""

    key: str  # 唯一标识，如 "mood.frustrated" / "role.tutor"
    layer: Layer
    dimension: str  # 维度名，同维度内做归一化与冲突消解
    name: str  # 人类可读名（中文）
    ktype: KeywordType
    description: str = ""
    is_base: bool = False  # 基础词不可删（doc/01 D17 护栏）

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        return f"Keyword({self.key},{self.layer.value},{self.ktype.value})"


@dataclass
class Evidence:
    """一条支撑某用户/agent 判断的原始观测（置信度来源，doc/02 §4）。"""

    key: str
    intensity: float  # 观测强度 0~1
    timestamp: float
    source: str  # heuristic | llm | preset | bootstrap | recall
    turn: int
    raw: Optional[str] = None  # 触发该证据的原文片段（调试用，不进 Prompt）

    def __post_init__(self) -> None:
        self.intensity = min(1.0, max(0.0, float(self.intensity)))


@dataclass
class KeywordState:
    """某关键词在运行时的累积状态。"""

    key: str
    layer: Layer
    dimension: str
    weight: float = 0.0  # 归一化后权重 0~1
    confidence: float = 0.0  # 0~1
    evidence_total: float = 0.0  # 证据强度和（含衰减）
    observed_count: int = 0
    last_turn: int = 0


@dataclass
class LayerWeights:
    """某一层的权重结果。"""

    layer: Layer
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class WeightSnapshot:
    """某一轮三层权重的完整存档（doc/01 D10 / doc/03 §2.4）。"""

    turn: int
    l1: dict[str, float] = field(default_factory=dict)
    l2: dict[str, float] = field(default_factory=dict)
    l3: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class TurnSummary:
    """压缩摘要链节点（doc/07 D22 / doc/03 §2.11）。

    kind="turn" 为每轮 LLM 回传摘要；kind="epoch" 为 doc/11 的合并摘要。
    """

    turn: int
    text: str  # <=100 字（turn）/ <=EPOCH_MAX_CHARS（epoch）
    timestamp: float = 0.0
    kind: str = "turn"
    keywords_snapshot: Optional[dict] = None  # 该轮关键权重摘要（可选，供调试）
