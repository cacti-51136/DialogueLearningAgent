"""三层权重计算引擎总入口（doc/02 核心算法 / doc/01 D1-D3）。

职责：
- 维护每关键词的证据累积（观测 + 预设），带时间衰减（doc/02 §4）。
- ``compute_l1 / compute_l2 / compute_l3`` 三层权重计算。
- L3 由 L1×L2 推导：场景基线 + 确定性耦合规则（doc/02 §5.2 阶段一）+ 可选 LLM 精炼(β)/LVM 先验(γ) 融合。
- 与 ``resolver`` 协作做维度内归一化/冲突消解。

有状态但**不依赖任何 UI/LLM**；纯 Python + 标准库。被 orchestration 层调用。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..config.loader import KeywordLib
from ..core.models import Evidence, Layer, WeightSnapshot
from .confidence import confidence, decayed_total
from .coupling import apply_rules
from .resolver import resolve_layer


@dataclass
class WeightEngineConfig:
    prior_strength: float = 2.0  # K
    default_half_life_hours: float = 6.0
    llm_fusion_beta: float = 0.3  # β
    lvm_gamma: float = 0.2  # γ


class WeightEngine:
    """三层权重引擎（有状态）。"""

    def __init__(
        self,
        lib: KeywordLib,
        cfg: Optional[WeightEngineConfig] = None,
        clock=None,
    ) -> None:
        self.lib = lib
        self.cfg = cfg or WeightEngineConfig()
        # clock: 任意提供 .now() 的对象；缺省用 time.time
        self._clock = clock

        self._obs: Dict[str, List[Evidence]] = defaultdict(list)
        self._preset: Dict[str, float] = {}  # key -> preset_intensity (稳定不衰减)
        self._active_l1: Set[str] = set()
        self._baseline_l3: Dict[str, float] = {}
        self._w3_llm: Dict[str, float] = {}  # 可选 LLM 精炼结果
        self._p_agent: Dict[str, float] = {}  # 可选 LVM 学习先验
        # 来源置信度（doc/02 §3.2：w = w_base × src_confidence × salience）
        # 缺省 1.0 = 不打折，保证既有路径（场景模板 preset 等）数值完全不变。
        self._src_conf: Dict[str, float] = {}

    # ---- 时钟 ----
    def _now(self) -> float:
        if self._clock is not None:
            return self._clock.now()
        import time

        return time.time()

    # ---- 注入接口 ----
    def set_l1_active(self, keys: List[str], intensity: float = 1.0, turn: int = 0) -> None:
        """设置当前活跃场景 L1（fixed 模式加载模板 / free 模式 Bootstrap）。"""
        self._active_l1 = set(keys)
        for k in keys:
            self._preset[k] = intensity

    def set_l1_scene(self, scene: Dict[str, float]) -> None:
        """按 key→intensity 字典注入活跃场景 L1（场景模板含各自强度）。"""
        self._active_l1 = set(scene.keys())
        for k, v in scene.items():
            self._preset[k] = max(0.0, min(1.0, v))

    def set_l3_baseline(self, baseline: Dict[str, float]) -> None:
        """注入场景 L3 基线（来自场景模板 l3_baseline），作为 w3_base 起点。"""
        self._baseline_l3 = dict(baseline)

    def set_preset(self, key: str, intensity: float, turn: int = 0) -> None:
        """注入一个预设强度（如 free 模式 L2 弱种子）。"""
        self._preset[key] = max(0.0, min(1.0, intensity))

    def set_src_conf(self, key: str, src_confidence: float) -> None:
        """设置来源置信度（doc/02 §3.2）。

        ``w(k) = w_base(k) × src_confidence(source) × salience(k)``。
        取值：user_explicit 1.00 / preset 0.80 / bootstrap 0.60 / inferred 0.50。

        注意：缺省为 1.0（不打折），故**仅对显式设置的关键词生效**——
        这样引入本机制不会改变既有场景模板路径的任何数值。
        """
        self._src_conf[key] = max(0.0, min(1.0, float(src_confidence)))

    def src_conf(self, key: str) -> float:
        """读取来源置信度（缺省 1.0）。"""
        return self._src_conf.get(key, 1.0)

    # ---- 运行期工作集增删改（doc/06 §4.2/§4.4：scene_ops L1 / agent_ops L3）----
    # 这些方法是纯机械的「写工作集」，护栏（delta 上限 / 模板词保护 / 词表校验）
    # 在 orchestration 层 _consume_ops 中统一裁决，这里只负责把已裁决的结果落进引擎。
    def add_l1_key(self, key: str, intensity: float = 0.6) -> None:
        """新增一个 L1 场景词到活跃工作集（scene_ops add）。"""
        self._active_l1.add(key)
        self._preset[key] = max(0.0, min(1.0, float(intensity)))

    def update_l1_key(self, key: str, delta: float) -> bool:
        """调整 L1 场景词强度（scene_ops update）。不在工作集返回 False。"""
        if key not in self._active_l1:
            return False
        self._preset[key] = max(0.0, min(1.0, self._preset.get(key, 0.0) + float(delta)))
        return True

    def remove_l1_key(self, key: str) -> None:
        """从 L1 活跃工作集移除（scene_ops delete，仅 auto_added）。"""
        self._active_l1.discard(key)
        self._preset.pop(key, None)

    def add_l3_key(self, key: str, intensity: float = 0.6) -> None:
        """新增一个 L3 工作集词（agent_ops add，仅非新维度已收编词）。"""
        self._baseline_l3[key] = max(0.0, min(1.0, float(intensity)))

    def update_l3_key(self, key: str, delta: float) -> bool:
        """调整 L3 工作集词强度（agent_ops update）。不在工作集返回 False。"""
        if key not in self._baseline_l3:
            return False
        self._baseline_l3[key] = max(0.0, min(1.0, self._baseline_l3[key] + float(delta)))
        return True

    def remove_l3_key(self, key: str) -> None:
        """从 L3 工作集移除（agent_ops delete，仅 auto_added）。"""
        self._baseline_l3.pop(key, None)

    def l3_working_keys(self) -> Set[str]:
        """当前 L3 工作集（模板基线 + 运行期新增）。"""
        return set(self._baseline_l3.keys())

    def add_evidence(self, ev: Evidence) -> None:
        """追加一条观测证据（来自 heuristics / llm_analyzer）。"""
        self._obs[ev.key].append(ev)

    def set_l3_llm(self, weights: Dict[str, float]) -> None:
        """注入 LLM 精炼后的 L3 权重（doc/02 §5.2 阶段二，可选）。"""
        self._w3_llm = dict(weights)

    def set_p_agent(self, weights: Dict[str, float]) -> None:
        """注入 LVM 在线学习先验（doc/06，可选）。"""
        self._p_agent = dict(weights)

    def active_l1(self) -> Set[str]:
        return set(self._active_l1)

    # ---- 证据累积计算 ----
    def _e_total(self, key: str) -> float:
        now = self._now()
        obs = decayed_total(self._obs.get(key, []), now, self.cfg.default_half_life_hours)
        pre = self._preset.get(key, 0.0) * self.cfg.prior_strength
        return obs + pre

    def _weight_of(self, key: str) -> float:
        # doc/02 §3.2：w = w_base × src_confidence × salience（salience 缺省 1.0）
        return confidence(self._e_total(key), self.cfg.prior_strength) * self._src_conf.get(key, 1.0)

    # ---- 三层计算 ----
    def compute_l1(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for kw in self.lib.lexicon.for_layer(Layer.L1):
            out[kw.key] = self._weight_of(kw.key)
        return out

    def compute_l2(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for kw in self.lib.lexicon.for_layer(Layer.L2):
            out[kw.key] = self._weight_of(kw.key)
        return out

    def compute_l3(self, l2_raw: Dict[str, float]) -> Dict[str, float]:
        eff = apply_rules(self.lib.rules, self._active_l1, l2_raw, self.lib.lexicon)
        beta = self.cfg.llm_fusion_beta
        gamma = self.cfg.lvm_gamma
        w3: Dict[str, float] = {}
        for kw in self.lib.lexicon.for_layer(Layer.L3):
            base = self._baseline_l3.get(kw.key, 0.0)
            boost = eff.boosts.get(kw.key, 0.0)
            w3_rule = max(0.0, min(1.0, base + boost))
            # 未启用 LLM/LVM 时，对应项以规则值兜底 → 退化为纯规则推导
            w_llm = self._w3_llm.get(kw.key, w3_rule)
            p = self._p_agent.get(kw.key, w3_rule)
            w = (1.0 - beta - gamma) * w3_rule + beta * w_llm + gamma * p
            w3[kw.key] = max(0.0, min(1.0, w))
        # 类别型置顶（doc/02 §5.2 set 命令覆盖 argmax）
        for _dim, val in eff.sets.items():
            w3[val] = 1.0
        return w3

    def compute_all(self, turn: int) -> WeightSnapshot:
        """计算三层权重并做维度内归一化，返回快照。"""
        l1_raw = self.compute_l1()
        l2_raw = self.compute_l2()
        l3_raw = self.compute_l3(l2_raw)
        snapshot = WeightSnapshot(
            turn=turn,
            l1=resolve_layer(l1_raw, self.lib.lexicon, Layer.L1),
            l2=resolve_layer(l2_raw, self.lib.lexicon, Layer.L2),
            l3=resolve_layer(l3_raw, self.lib.lexicon, Layer.L3),
            timestamp=self._now(),
        )
        return snapshot
