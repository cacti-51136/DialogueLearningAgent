"""一轮对话的事件流（doc/04 §1 交互层共享事件）。

引擎通过 ``DialogueEngine.stream_reply_sync`` 产出这些事件，供 PyQt / API(SSE) 等
外壳消费。采用「引擎产出事件、外壳只发信号更新 UI」的解耦模式（doc/04 §4.2）。

核心包 **禁止 import** qt / fastapi（doc/01 §4 关键约束）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import WeightSnapshot


@dataclass
class TurnEvent:
    """一轮对话中所有事件的基类。"""


@dataclass
class TokenEvent(TurnEvent):
    """流式文本片段（逐 token 推给 UI）。

    ``replace=True`` 表示以本段**替换**此前累积文本（如重复护栏触发后重试）。
    UI 正常累积 token 显示，收到 ``DoneEvent`` 时以 ``final_text`` 为准兜底。
    """

    text: str
    replace: bool = False


@dataclass
class WeightUpdateEvent(TurnEvent):
    """本轮三层权重快照，刷新右侧权重面板。"""

    snapshot: WeightSnapshot


@dataclass
class PersonaChangeEvent(TurnEvent):
    """人格显著切换提示（Δ ≥ τ_notify）。"""

    delta: float
    action: str = "rebuilt"
    diff: List[dict] = field(default_factory=list)


@dataclass
class ChainStepEvent(TurnEvent):
    """调试思维链的一步（doc/04 §4.4），仅调试模式填充。"""

    step: str
    detail: dict = field(default_factory=dict)


@dataclass
class DoneEvent(TurnEvent):
    """一轮结束。``final_text`` 为权威最终文本（UI 以它为准）。"""

    turn: int
    final_text: str
    rep_hit: bool = False
    notify: bool = False
    summary: str = ""
    candidate_count: int = 0
    compact_actions: List[str] = field(default_factory=list)
    usage: Optional[dict] = None


@dataclass
class ErrorEvent(TurnEvent):
    """错误（已对用户友好化，不含堆栈/内部路径）。"""

    message: str
    code: str = "INTERNAL_ERROR"
