"""LLM 结构化分析器（doc/01 §7 步骤3 / doc/02 §4 抽取）。

封装「调用 LLM 做一轮对话分析」：输入用户当前消息 + 上一轮 agent 回复，要求模型以
JSON 回传：关键词抽取（extractions）、场景实时判定（scene_ops，auto/free 模式用）、
词库操作建议（agent_ops）、本轮压缩摘要（turn_summary ≤100 字）、置信度。

关键约束（doc/01 D11 / doc/02 §11.9）：抽取到词表外关键词一律丢弃（不进引擎，但可反哺词库）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.errors import AnalysisError
from ..core.ports import ChatMessage
from ..keywords.lexicon import Lexicon

_SYSTEM_PROMPT = """你是 DialogueLearningAgent 的分析器。基于用户发言与上一轮 agent 回复，输出 JSON：
{
  "extractions": [{"key": "受控关键词(必须来自给定词表)", "intensity": 0~1}],
  "scene_ops": [],            // auto/free 模式下对 L1 场景关键词的实时增减（可选）
  "agent_ops": [],            // 对场景工作集关键词的增删改建议（可选，带护栏）
  "turn_summary": "≤100字本轮压缩摘要",
  "confidence": 0~1
}
仅输出 JSON，不要解释。情绪/脾性类关键词（user_mood.*/user_temper.*）只能从用户真实言行提炼，不得采信用户自述标签。"""


@dataclass
class AnalysisResult:
    extractions: List[dict] = field(default_factory=list)  # 白名单过滤后的 {key,intensity}
    scene_ops: List[dict] = field(default_factory=list)
    agent_ops: List[dict] = field(default_factory=list)
    turn_summary: str = ""
    confidence: float = 0.5
    raw_unknown: List[str] = field(default_factory=list)  # 词表外关键词（反哺词库）


def analyze(
    client,
    user_text: str,
    prev_agent_text: str,
    lexicon: Lexicon,
    *,
    model: Optional[str] = None,
) -> AnalysisResult:
    """调用 LLM 完成一轮结构化分析。失败抛 AnalysisError。"""
    user_msg = (
        f"上一轮 agent 说：{prev_agent_text or '（无）'}\n"
        f"用户现在说：{user_text}"
    )
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_msg),
    ]
    try:
        resp = client.complete(
            messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            **({"model": model} if model else {}),
        )
        data = json.loads(resp.content)
    except json.JSONDecodeError as e:
        raise AnalysisError(f"分析器 JSON 解析失败: {e}") from e
    except Exception as e:  # noqa: BLE001 - 统一转 AnalysisError
        raise AnalysisError(f"分析器调用失败: {e}") from e

    if not isinstance(data, dict):
        raise AnalysisError("分析器返回非 dict")

    known: List[dict] = []
    unknown: List[str] = []
    for e in data.get("extractions", []):
        key = e.get("key")
        if not key:
            continue
        canon = lexicon.normalize(str(key))
        if canon is None:
            unknown.append(str(key))
            continue
        intensity = max(0.0, min(1.0, float(e.get("intensity", 0.5))))
        known.append({"key": canon, "intensity": intensity})

    summary = str(data.get("turn_summary", ""))[:100]
    return AnalysisResult(
        extractions=known,
        scene_ops=list(data.get("scene_ops", [])),
        agent_ops=list(data.get("agent_ops", [])),
        turn_summary=summary,
        confidence=float(data.get("confidence", 0.5)),
        raw_unknown=unknown,
    )
