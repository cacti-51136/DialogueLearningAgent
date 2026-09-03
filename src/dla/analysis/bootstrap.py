"""自由模式 Bootstrap：由「场景描述」一次性生成三层基础关键词（doc/02 §3.6 / 决策 D26）。

与逐轮分析器（``llm_analyzer``）的区别：

- 本模块是**一次性冷启动**（会话首轮之前），不参与每轮循环；
- 输出直接作为 L1/L2/L3 的**起点种子**，之后仍由 ``scene_ops`` / 用户肖像分析 /
  ``agent_ops`` 逐步演化（即用户要的「再逐渐更新」）；
- 所有种子 ``source = bootstrap``、``src_confidence = 0.60``（doc/02 §3.2，
  介于 inferred 0.50 与 preset 0.80 之间：比实时推断可靠，比策展模板保守）。

硬护栏（违反即丢弃并记入 ``rejected``，供审计与反哺词库）：

- 三层所有词都必须过**白名单校验**（不在受控词表内 → 丢弃）；
- 层归属必须匹配（L1 的词不能混进 L2/L3）；
- **``user_temper.*`` / ``user_mood.*`` 禁止注入**——脾性/情绪只能从对话中涌现
  （doc/02 §11.9），Bootstrap 不得预设。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.errors import AnalysisError
from ..core.models import Layer
from ..core.ports import ChatMessage
from ..keywords.lexicon import Lexicon

# 禁止由 Bootstrap 预设的维度（doc/02 §11.9：情绪/脾性只从对话涌现）
_FORBIDDEN_PREFIXES = ("user_temper.", "user_mood.")

_SYSTEM_PROMPT = """你是关键词引导器。根据用户给出的"对话场景描述"，为三层各产出一组基础关键词。

要求：
- 每层输出的 key 必须**严格来自给定受控词表**，不得自造。
- L1 = 功能场景层（场景 scene.*、双方目标 goal_agent.*、领域 domain.*、约束 constraint.*）
- L2 = 用户肖像层（身份 user_profile.*、水平 user_level.*、目标 user_goal.*）
      **禁止输出 user_temper.* 与 user_mood.***——脾性/情绪必须由真实对话涌现，不得预设。
- L3 = agent 肖像层（角色 agent_role.*、语气 agent_tone.*、风格 agent_style.*、共情 empathy 等）
- intensity 取值 0~1，表示该词在该场景下的贴合强度。

仅输出 JSON，不要解释：
{
  "l1": [{"key": "scene.xxx", "intensity": 0.9}],
  "l2": [{"key": "user_level.xxx", "intensity": 0.7}],
  "l3": [{"key": "agent_role.xxx", "intensity": 0.8}, {"key": "empathy", "value": 0.8}]
}"""


@dataclass
class BootstrapSeeds:
    """Bootstrap 产出的三层种子（均已过白名单与护栏）。"""

    l1: Dict[str, float] = field(default_factory=dict)
    l2: Dict[str, float] = field(default_factory=dict)
    l3: Dict[str, float] = field(default_factory=dict)
    rejected: List[dict] = field(default_factory=list)  # {key, layer, reason}
    raw_unknown: List[str] = field(default_factory=list)  # 词表外（反哺词库）
    describe: str = ""

    @property
    def total(self) -> int:
        return len(self.l1) + len(self.l2) + len(self.l3)


def _pick_intensity(item: dict) -> float:
    """取强度值：兼容 ``intensity`` 与标量维常用的 ``value`` 两种写法。"""
    raw = item.get("intensity", item.get("value", 0.5))
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


def _validate_layer(
    items: object,
    layer: Layer,
    lexicon: Lexicon,
    *,
    allow_forbidden: bool = False,
) -> tuple[Dict[str, float], List[dict], List[str]]:
    """对单层条目做白名单 + 层归属 + 禁词校验。

    返回 ``(accepted, rejected, unknown)``。
    """
    accepted: Dict[str, float] = {}
    rejected: List[dict] = []
    unknown: List[str] = []
    if not isinstance(items, list):
        return accepted, rejected, unknown

    for item in items:
        if not isinstance(item, dict):
            continue
        raw_key = item.get("key")
        if not raw_key:
            continue
        key_str = str(raw_key)

        # 1) 禁词护栏（脾性/情绪不得预设）
        if not allow_forbidden and key_str.lower().startswith(_FORBIDDEN_PREFIXES):
            rejected.append({"key": key_str, "layer": layer.value, "reason": "forbidden_dimension"})
            continue

        # 2) 白名单归一
        canon = lexicon.normalize(key_str)
        if canon is None:
            unknown.append(key_str)
            rejected.append({"key": key_str, "layer": layer.value, "reason": "not_in_lexicon"})
            continue

        # 3) 层归属必须匹配
        kw = lexicon.get(canon)
        if kw is None or kw.layer != layer:
            rejected.append(
                {"key": canon, "layer": layer.value, "reason": "layer_mismatch"}
            )
            continue

        accepted[canon] = _pick_intensity(item)

    return accepted, rejected, unknown


def bootstrap(
    client,
    describe: str,
    lexicon: Lexicon,
    *,
    model: Optional[str] = None,
) -> BootstrapSeeds:
    """调用 LLM 由场景描述生成三层基础关键词。失败抛 :class:`AnalysisError`。

    调用方应先确认 ``mode == "free"`` 且确有描述文本（缺描述时的降级策略见
    :func:`should_degrade_to_auto`）。
    """
    if not (describe or "").strip():
        raise AnalysisError("Bootstrap 需要非空的场景描述")

    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=describe.strip()),
    ]
    try:
        resp = client.complete(
            messages,
            temperature=0.3,
            response_format={"type": "json_object"},
            **({"model": model} if model else {}),
        )
        data = _loads(resp.content)
    except AnalysisError:
        raise
    except Exception as e:  # noqa: BLE001 - 统一转 AnalysisError
        raise AnalysisError(f"Bootstrap 调用失败: {e}") from e

    if not isinstance(data, dict):
        raise AnalysisError("Bootstrap 返回非 dict")

    l1, rej1, unk1 = _validate_layer(data.get("l1", []), Layer.L1, lexicon)
    l2, rej2, unk2 = _validate_layer(data.get("l2", []), Layer.L2, lexicon)
    l3, rej3, unk3 = _validate_layer(data.get("l3", []), Layer.L3, lexicon)

    return BootstrapSeeds(
        l1=l1,
        l2=l2,
        l3=l3,
        rejected=rej1 + rej2 + rej3,
        raw_unknown=unk1 + unk2 + unk3,
        describe=describe.strip(),
    )


def _loads(text: str) -> dict:
    """容错解析：优先严格 JSON，失败则截取首个 {...} 块再试。"""
    import json

    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            pass
    raise AnalysisError(f"Bootstrap JSON 解析失败: {text[:200]!r}")


def should_degrade_to_auto(mode: str, describe: Optional[str], require_desc: bool) -> bool:
    """判定是否应因缺少描述而降级为 auto 模式（doc/02 §3.6）。

    仅 ``free`` 模式且 ``DLA_MODE__FREE_REQUIRE_DESC=true`` 且无描述时降级。
    """
    return mode == "free" and require_desc and not (describe or "").strip()
