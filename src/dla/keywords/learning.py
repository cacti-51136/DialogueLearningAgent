"""LVM 在线学习器（doc-06 §6 子系统封装）。

把每轮的 ``agent_ops``（前向回路监督）与 ``satisfaction``（后向回路监督）转成训练样本
``(q, p*)``，攒入回放缓冲，每轮做 1~3 个 epoch 的小步更新，并把关系头矩阵 / 偏置 / 动量 /
步数持久化到 SQLite（``lvm_relation_heads`` / ``lvm_training_log`` / ``feedback_signals``）。

与引擎解耦：本类只依赖 vectorizer + settings + 可选 repo，便于单测。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class Sample:
    q: np.ndarray
    p_star: Dict[str, float]
    head: str = "user_agent"
    weight: float = 1.0


class OnlineLearner:
    """LVM 在线学习协调器。"""

    def __init__(self, vectorizer, settings, repo=None) -> None:
        self.v = vectorizer
        self.s = settings
        self.repo = repo
        self.replay: Deque[Sample] = deque(maxlen=max(1, int(settings.lvm_replay_buffer)))
        self._last_loss: List[float] = []

    # ---- 样本构造 ----
    def add_sample(self, q: np.ndarray, p_star: Dict[str, float], head: str = "user_agent", weight: float = 1.0) -> None:
        """直接追加一条训练样本（已构造好目标分布）。"""
        if not p_star:
            return
        self.replay.append(Sample(q=np.asarray(q, dtype=float), p_star=dict(p_star), head=head, weight=weight))

    def record_agent_ops(self, q: np.ndarray, ops: list, applied_only: bool = True) -> int:
        """把本轮生效的 agent_ops 转成 user_agent 头的目标分布 p*（doc-06 §6.1）。

        update +δ / add → 该词目标分升高；delete → 目标分设 0（不强制覆盖全空间）。
        返回构造出的样本数（0 表示无有效 ops）。
        """
        if not ops:
            return 0
        p_star: Dict[str, float] = {}
        for op in ops:
            if not isinstance(op, dict):
                continue
            if applied_only and not bool(op.get("_applied", True)):
                continue
            op_type = str(op.get("op") or "").lower()
            key = op.get("key")
            if not key:
                continue
            if op_type == "update":
                delta = float(op.get("delta", 0.0))
                p_star[key] = p_star.get(key, 0.0) + delta + 0.5  # 以 0.5 为基线，便于 softmax 区分
            elif op_type == "add":
                intensity = max(0.0, min(1.0, float(op.get("intensity", 0.6))))
                p_star[key] = max(p_star.get(key, 0.0), intensity + 0.3)
            elif op_type == "delete":
                p_star[key] = 0.0
            else:
                continue
        if not p_star:
            return 0
        self.add_sample(q, p_star, head="user_agent")
        return 1

    def record_satisfaction(
        self,
        session_id: Optional[str],
        turn: int,
        q_prev: np.ndarray,
        active_l3: List[str],
        score: float,
        signal: str = "",
        based_on_turn: Optional[int] = None,
        head: str = "user_agent",
    ) -> int:
        """把满意度信号转成训练样本（doc-06 §6.1 后向回路）。

        score ∈ [-1,1]：>0 提升上一轮活跃 L3 关键词权重，<0 压低。
        同时落 ``feedback_signals`` 表供可观测。
        """
        if not active_l3 or abs(score) < 1e-6:
            return 0
        target = 1.0 if score > 0 else 0.0
        p_star = {k: target for k in active_l3}
        self.add_sample(q_prev, p_star, head=head)
        if self.repo is not None:
            try:
                self.repo.log_feedback_signal(
                    session_id, turn, float(score), signal, based_on_turn
                )
            except Exception:  # noqa: BLE001
                pass
        return 1

    # ---- 训练 ----
    def train(self, epochs: int = 1) -> List[float]:
        """在回放缓冲上做 epochs 个小步更新，返回每步损失。"""
        losses: List[float] = []
        if not self.replay:
            return losses
        epochs = max(1, epochs)
        for _ in range(epochs):
            # 复制一份以避免训练中数据变化（理论上不会），并随机化顺序缓解相关偏差
            batch = list(self.replay)
            for sample in batch:
                loss = self.v.train_step(
                    sample.q,
                    sample.p_star,
                    head=sample.head,
                    lr=self.s.lvm_learning_rate,
                    momentum=self.s.lvm_momentum,
                    weight_decay=self.s.lvm_weight_decay,
                    grad_clip=self.s.lvm_grad_clip,
                    lr_decay=self.s.lvm_lr_decay,
                    margin_lambda=self.s.lvm_margin_lambda,
                    tau=self.s.lvm_temp,
                )
                losses.append(loss)
        self._last_loss = losses
        self._persist()
        return losses

    @property
    def last_loss(self) -> List[float]:
        return self._last_loss

    # ---- 持久化 ----
    def _persist(self) -> None:
        if self.repo is None:
            return
        try:
            self.repo.save_lvm_heads(self.v.export_state(), self.v.embeddings_dict())
            if self._last_loss:
                self.repo.log_training_step(
                    self.v.step, float(np.mean(self._last_loss)), self.s.lvm_learning_rate, len(self.replay), "user_agent"
                )
        except Exception:  # noqa: BLE001 - 持久化失败绝不阻断对话
            pass

    def reset(self) -> None:
        """一键复位（doc-06 §9）：清空回放与步数，矩阵回到 α·I。"""
        self.replay.clear()
        self._last_loss = []
        dim = self.v.dim
        for r in ("scene_agent", "user_agent", "user_user", "scene_constraint"):
            self.v._M[r] = np.eye(dim) * self.v._init_scale
            self.v._bias[r] = np.zeros(len(self.v._l3_keys))
            self.v._mom[r] = np.zeros((dim, dim))
            self.v._bias_mom[r] = np.zeros(len(self.v._l3_keys))
        self.v.step = 0
        if self.repo is not None:
            try:
                self.repo.lvm_reset()
            except Exception:  # noqa: BLE001
                pass
