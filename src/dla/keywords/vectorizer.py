"""本地向量化模型 LVM（doc/06 子系统核心）。

关键词编码为 d 维冻结向量 ``e_k``；跨层关联由可训练的**双线性矩阵** ``M_r``（多关系头）
刻画——这是"本地模型权重"。前向：由场景/用户上下文 ``q_t`` 检索出 Agent 关键词先验
``p_agent``（§3/§5）；反向：``agent_ops`` + ``satisfaction`` 作为监督信号，对 ``M_r`` 做
**解析梯度 + 动量 SGD** 更新（§6.3，纯 numpy，无 GPU、无 autograd）。

设计取舍：
- ``e_k`` **冻结不训练**（doc/06 §2.1）：语义锚点一旦被训练污染，关联就会失真。学习只发生在 M_r。
- 随机回退下 ``e_k`` 按 keyword_id **确定性初始化**（同一词每次启动一致，可复现）；可选 backbone 热启动。
- 为了数值稳定与可复现，所有 ``e_k`` 统一 L2 归一化（backbone 向量本身已归一化，随机向量额外归一化到单位球）。
- 关系头 bias 仅对 L3 关键词端维护（打分两端中的"被检索端"永远是 L3 关键词），规模有界。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.models import Layer

# 四个关系头（doc/06 §2.2）
RELATION_HEADS = ("scene_agent", "user_agent", "user_user", "scene_constraint")
# 参与主动前向检索（identity / agent_prior）的两个头
ACTIVE_HEADS = ("scene_agent", "user_agent")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / max(len(x), 1))
    return e / s


@dataclass
class LvmState:
    """可序列化的 LVM 状态（关系头矩阵 + 偏置 + 步数）。"""

    dim: int = 64
    step: int = 0
    heads: Dict[str, "HeadWeights"] = field(default_factory=dict)


@dataclass
class HeadWeights:
    matrix: np.ndarray  # d×d
    bias: np.ndarray    # len(l3_keys)

    def to_dict(self) -> dict:
        return {"matrix": self.matrix.tolist(), "bias": self.bias.tolist()}

    @staticmethod
    def from_dict(d: dict, dim: int) -> "HeadWeights":
        return HeadWeights(
            matrix=np.array(d["matrix"], dtype=float).reshape(dim, dim),
            bias=np.array(d["bias"], dtype=float),
        )


class KeywordVectorizer:
    """关键词向量化与双线性关联（doc/06 §2/§3/§5/§6）。"""

    def __init__(
        self,
        lexicon,
        dim: int = 64,
        init_scale: float = 0.1,
        backbone=None,
    ) -> None:
        self.dim = dim
        self.lex = lexicon
        self._init_scale = init_scale
        self._backbone = backbone

        self._all_keys: List[str] = list(lexicon.all_keys())
        # L3 关键词是双线性打分的"被检索端"，偏置在此维度上维护
        self._l3_keys: List[str] = [k.key for k in lexicon.for_layer(Layer.L3)]
        self._l3_index: Dict[str, int] = {k: i for i, k in enumerate(self._l3_keys)}

        # 冻结嵌入 e_k（确定性）
        self._e: Dict[str, np.ndarray] = {k: self._make_embedding(k) for k in self._all_keys}

        # 可训练关系头 M_r（初始化 α·I）与偏置（0）
        self._M: Dict[str, np.ndarray] = {}
        self._bias: Dict[str, np.ndarray] = {}
        self.step = 0
        for r in RELATION_HEADS:
            self._M[r] = np.eye(dim) * init_scale
            self._bias[r] = np.zeros(len(self._l3_keys))
        # 动量
        self._mom: Dict[str, np.ndarray] = {r: np.zeros((dim, dim)) for r in RELATION_HEADS}
        self._bias_mom: Dict[str, np.ndarray] = {r: np.zeros(len(self._l3_keys)) for r in RELATION_HEADS}

    # ---- 冻结嵌入 ----
    def _make_embedding(self, key: str) -> np.ndarray:
        """确定性基础向量：backbone 热启动，否则按 keyword_id 哈希随机初始化。"""
        if self._backbone is not None:
            try:
                kw = self.lex.get(key)
                text = ""
                if kw is not None:
                    text = f"{kw.name} / {kw.description}"
                vec = self._backbone.encode(text or key)
                vec = np.asarray(vec, dtype=float)
                if vec.shape[0] != self.dim:
                    # backbone 维度不符则截断/补零
                    if vec.shape[0] > self.dim:
                        vec = vec[: self.dim]
                    else:
                        vec = np.pad(vec, (0, self.dim - vec.shape[0]))
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
                return vec
            except Exception:  # noqa: BLE001 - backbone 不可用（缺依赖/离线）则回退随机
                pass
        # 确定性随机回退
        seed = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.normal(0.0, 0.1, self.dim)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    # ---- 上下文向量 ----
    def context_vector(
        self, w1: Dict[str, float], w2: Dict[str, float]
    ) -> np.ndarray:
        """构造场景 + 用户肖像上下文向量 q_t = Σ L1 w·e + Σ L2 w·e（doc/06 §3/§5）。"""
        q = np.zeros(self.dim)
        for k, w in w1.items():
            e = self._e.get(k)
            if e is not None:
                q += float(w) * e
        for k, w in w2.items():
            e = self._e.get(k)
            if e is not None:
                q += float(w) * e
        return q

    # ---- 双线性打分 ----
    def _score(self, head: str, q: np.ndarray, j: int) -> float:
        e = self._e[self._l3_keys[j]]
        return float(q @ (self._M[head] @ e)) / self.dim + float(self._bias[head][j])

    def identity_keywords(self, q: np.ndarray, top_t: int = 6, tau: float = 0.2) -> List[str]:
        """冷启动 Agent 身份关键字（doc-06 §3）。返回 q 在 scene_agent 头下的 Top-T L3 词。"""
        if not self._l3_keys:
            return []
        logits = np.array([self._score("scene_agent", q, i) for i in range(len(self._l3_keys))])
        p = _softmax(logits / tau)
        order = np.argsort(-p)[: min(top_t, len(self._l3_keys))]
        return [self._l3_keys[i] for i in order]

    def agent_prior(self, q: np.ndarray, tau: float = 0.2) -> Dict[str, float]:
        """Agent 关键词先验软分布 p_agent（doc-06 §5）：q 在 user_agent 头下的 softmax。"""
        if not self._l3_keys:
            return {}
        logits = np.array([self._score("user_agent", q, i) for i in range(len(self._l3_keys))])
        p = _softmax(logits / tau)
        return {self._l3_keys[i]: float(p[i]) for i in range(len(self._l3_keys))}

    def has_learned(self) -> bool:
        """是否已发生过训练步（用于引擎侧零回归门控：未训练不注入 p_agent）。"""
        return self.step > 0

    # ---- 训练：解析梯度 + 动量 SGD（doc-06 §6.3）----
    def ce_loss(self, q: np.ndarray, p_star: Dict[str, float], head: str = "user_agent", tau: float = 0.2) -> float:
        """交叉熵损失（doc-06 §6.2 L_ce），供调试与梯度校验使用。"""
        n = len(self._l3_keys)
        if n == 0:
            return 0.0
        target_full = np.zeros(n)
        for k, v in p_star.items():
            if k in self._l3_index:
                target_full[self._l3_index[k]] = max(0.0, float(v))
        target_soft = _softmax(target_full) if target_full.sum() > 0 else target_full
        logits = np.array([self._score(head, q, i) for i in range(n)])
        pred = _softmax(logits / tau)
        return float(-np.sum(target_soft * np.log(pred + 1e-12)))

    def _grad_ce(self, q: np.ndarray, p_star: Dict[str, float], head: str, tau: float, weight_decay: float):
        """解析梯度（仅交叉熵 + 权重衰减，不含 margin）。返回 (gM, gBias, loss)。

        关键：双线性打分 ``ŝ_j = qᵀ M e_j / d + b_j`` 对 ``M`` 的解析梯度为
        ``Σ_j (pred_j - target_j)/τ · q e_jᵀ / d``（见 doc-06 §6.3）。
        """
        n = len(self._l3_keys)
        target_full = np.zeros(n)
        for k, v in p_star.items():
            if k in self._l3_index:
                target_full[self._l3_index[k]] = max(0.0, float(v))
        target_soft = _softmax(target_full) if target_full.sum() > 0 else target_full
        logits = np.array([self._score(head, q, i) for i in range(n)])
        pred = _softmax(logits / tau)
        g_logit = (pred - target_soft) / tau

        d = self.dim
        gM = np.zeros((self.dim, self.dim))
        gBias = np.zeros(n)
        for j in range(n):
            ej = self._e[self._l3_keys[j]]
            gM += g_logit[j] * np.outer(q, ej) / d
            gBias[j] = g_logit[j]
        # 注意：与 doc-06 §6.3 公式一致——对“交叉熵求和”取解析梯度（不再除以 n），
        # 与 ce_loss 的求和实现严格对应，也避免词表变大时梯度被无谓缩小。
        gM += weight_decay * self._M[head]
        return gM, gBias, float(-np.sum(target_soft * np.log(pred + 1e-12)))

    def train_step(
        self,
        q: np.ndarray,
        p_star: Dict[str, float],
        head: str = "user_agent",
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        lr_decay: float = 0.0,
        margin_lambda: float = 0.0,
        margin_m: float = 0.2,
        tau: float = 0.2,
    ) -> float:
        """对单条 (q, p*) 样本做一次小步更新，返回该步交叉熵损失。

        p* 为 L3 关键词 -> 目标权重（≥0）。缺失的 L3 词视为 0（不强制覆盖全空间）。
        """
        n = len(self._l3_keys)
        if n == 0:
            return 0.0

        gM, gBias, loss = self._grad_ce(q, p_star, head, tau, weight_decay)

        # 排序边界损失（可选）：pos = 目标>0 的词，neg = 其余
        if margin_lambda > 0:
            target_full = np.zeros(n)
            for k, v in p_star.items():
                if k in self._l3_index:
                    target_full[self._l3_index[k]] = max(0.0, float(v))
            if target_full.sum() > 0:
                logits = np.array([self._score(head, q, i) for i in range(n)])
                pos = [j for j in range(n) if target_full[j] > 0]
                neg = [j for j in range(n) if target_full[j] <= 0]
                if pos and neg:
                    for jp in pos:
                        for jn in neg:
                            if margin_m + logits[jn] - logits[jp] > 0:
                                ep = self._e[self._l3_keys[jp]]
                                en = self._e[self._l3_keys[jn]]
                                gM += margin_lambda * (np.outer(q, en) - np.outer(q, ep)) / self.dim
                                gBias[jp] += margin_lambda
                                gBias[jn] -= margin_lambda

        # 梯度裁剪
        norm = np.linalg.norm(gM)
        if grad_clip > 0 and norm > grad_clip:
            gM = gM * (grad_clip / norm)

        # 学习率按步衰减
        eff_lr = lr / (1.0 + lr_decay * self.step)

        # 动量更新（Momentum SGD）
        v = momentum * self._mom[head] - eff_lr * gM
        self._mom[head] = v
        self._M[head] = self._M[head] + v

        vb = momentum * self._bias_mom[head] - eff_lr * gBias
        self._bias_mom[head] = vb
        self._bias[head] = self._bias[head] + vb

        self.step += 1
        return loss

    # ---- 序列化 ----
    def export_state(self) -> dict:
        return {
            "dim": self.dim,
            "step": self.step,
            "heads": {r: self._heads_to_dict(r) for r in RELATION_HEADS},
        }

    def _heads_to_dict(self, r: str) -> dict:
        return {"matrix": self._M[r].tolist(), "bias": self._bias[r].tolist()}

    def import_state(self, state: dict) -> None:
        dim = int(state.get("dim", self.dim))
        self.dim = dim
        self.step = int(state.get("step", 0))
        for r in RELATION_HEADS:
            hd = state.get("heads", {}).get(r)
            if not hd:
                continue
            self._M[r] = np.array(hd["matrix"], dtype=float).reshape(dim, dim)
            bias = np.array(hd["bias"], dtype=float)
            if bias.shape[0] != len(self._l3_keys):
                # 词表变化导致维度不符：裁剪/补零后继续（不致命）
                if bias.shape[0] > len(self._l3_keys):
                    bias = bias[: len(self._l3_keys)]
                else:
                    bias = np.pad(bias, (0, len(self._l3_keys) - bias.shape[0]))
            self._bias[r] = bias
        # 动量随新状态重置（避免形状/尺度错配）
        self._mom = {r: np.zeros((dim, dim)) for r in RELATION_HEADS}
        self._bias_mom = {r: np.zeros(len(self._l3_keys)) for r in RELATION_HEADS}

    def embeddings_dict(self) -> Dict[str, List[float]]:
        """导出冻结嵌入（可选持久化 / 反哺，doc-06 §2.1）。"""
        return {k: self._e[k].tolist() for k in self._all_keys}

    def load_embeddings(self, emb: Dict[str, List[float]]) -> None:
        for k, v in emb.items():
            if k in self._e and len(v) == self.dim:
                self._e[k] = np.array(v, dtype=float)
