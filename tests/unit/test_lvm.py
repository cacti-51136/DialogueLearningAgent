"""LVM 向量化与在线学习：纯数学/单元单测（doc/06）。

全部离线、纯 numpy，不依赖 LLM 与 SQLite；可独立复跑。
覆盖：冻结嵌入确定性、上下文向量、身份关键字、agent 先验、解析梯度（数值校验）、
训练降损、嵌入冻结、状态序列化、OnlineLearner 样本构造与持久化。
"""
import json
import os

import numpy as np
import pytest

np = pytest.importorskip("numpy")

from dla.config.loader import load_keyword_lib
from dla.config.settings import Settings
from dla.keywords.learning import OnlineLearner
from dla.keywords.vectorizer import KeywordVectorizer

KW_DIR = "config/keywords"
COUPLE = "config/coupling_rules.yaml"
REL = "scene_agent"


def _lexicon():
    return load_keyword_lib(KW_DIR, COUPLE).lexicon


def test_embedding_deterministic():
    """同一 keyword_id 跨实例确定性初始化（doc-06 §2.1）。"""
    lex = _lexicon()
    v1 = KeywordVectorizer(lex, dim=16)
    v2 = KeywordVectorizer(lex, dim=16)
    k = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)[0].key
    assert np.allclose(v1._e[k], v2._e[k])


def test_embeddings_normalized():
    """e_k 统一 L2 归一化到单位球（数值稳定）。"""
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=32)
    for k, e in v._e.items():
        assert abs(np.linalg.norm(e) - 1.0) < 1e-6


def test_context_vector_empty_is_zero():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=8)
    q = v.context_vector({}, {})
    assert np.allclose(q, 0.0)


def test_context_vector_linear():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=8)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    k = l1[0].key
    q1 = v.context_vector({k: 0.5}, {})
    q2 = v.context_vector({k: 1.0}, {})
    assert np.allclose(q2, 2 * q1)


def test_identity_keywords_returns_top_t():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=32)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    # 用一个真实 L1 词构造 q（确定性非空）
    k = l1[0].key
    q = v.context_vector({k: 1.0}, {})
    ids = v.identity_keywords(q, top_t=3, tau=0.2)
    assert len(ids) == 3
    # identity_keywords 返回的是 L3 关键词 key；部分 L3 词是裸词（如 verbosity），不一定含 "."
    assert all(isinstance(x, str) for x in ids)
    assert set(ids).issubset(set(v._l3_keys))


def test_agent_prior_sums_to_one():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=32)
    l2 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L2)
    k = l2[0].key
    q = v.context_vector({}, {k: 0.6})
    prior = v.agent_prior(q, tau=0.2)
    assert set(prior.keys()) == set(v._l3_keys)
    assert abs(sum(prior.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in prior.values())


def test_train_step_reduces_loss():
    """训练步应使（同分布样本上的）交叉熵损失下降（doc-06 §6.2/§6.3）。

    注意 doc-06 §6.1 规定 p* = softmax(目标分)：单一弱目标分（如 1.0）会让 p* 接近均匀分布，
    其熵≈loss0，模型几乎没有下降空间。这里用明确可分的强信号（目标分 4.0，p* 近乎 one-hot）
    来演示优化器确实能把损失压下来。
    """
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=16)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    target = l3[0].key
    q = v._e[l1[0].key].copy() * 2.0  # 非空上下文
    p_star = {target: 4.0}
    loss0 = v.ce_loss(q, p_star, head=REL, tau=0.2)
    losses = [v.train_step(q, p_star, head=REL, lr=0.02, momentum=0.9, margin_lambda=0.0) for _ in range(60)]
    assert losses[-1] < loss0 - 0.5, f"loss 未下降: {loss0} -> {losses[-1]}"


def test_analytic_gradient_matches_numeric():
    """双线性打分对 M_r 的解析梯度应与数值梯度一致（doc-06 §6.3 关键公式）。"""
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=12)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    target = l3[0].key
    q = v._e[l1[0].key].copy()
    p_star = {target: 1.0}
    tau = 0.2
    n = len(v._l3_keys)

    gM, _gB, _l = v._grad_ce(q, p_star, REL, tau, weight_decay=0.0)

    eps = 1e-5
    a, b = 0, 1
    M0 = v._M[REL].copy()
    M0[a, b] += eps
    v._M[REL] = M0
    loss_plus = v.ce_loss(q, p_star, head=REL, tau=tau)
    M0[a, b] -= 2 * eps
    v._M[REL] = M0
    loss_minus = v.ce_loss(q, p_star, head=REL, tau=tau)
    v._M[REL] = M0 + eps  # restore
    numeric = (loss_plus - loss_minus) / (2 * eps)
    # 解析梯度应与数值梯度一致（doc-06 §6.3：对交叉熵求和取梯度，无 1/n 缩放）
    assert abs(gM[a, b] - numeric) < 1e-3


def test_embeddings_frozen_after_training():
    """e_k 冻结不训练：训练后嵌入不变（doc-06 §2.1）。"""
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=16)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    before = {k: v._e[k].copy() for k in v._e}
    for _ in range(20):
        v.train_step(v._e[l1[0].key], {l3[0].key: 1.0}, head=REL, margin_lambda=0.0)
    for k in before:
        assert np.allclose(v._e[k], before[k])


def test_export_import_roundtrip():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=16)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    for _ in range(5):
        v.train_step(v._e[l1[0].key], {l3[0].key: 1.0}, head=REL, margin_lambda=0.0)
    state = v.export_state()
    v2 = KeywordVectorizer(lex, dim=16)
    v2.import_state(state)
    assert v2.step == v.step
    for r in ("scene_agent", "user_agent", "user_user", "scene_constraint"):
        assert np.allclose(v2._M[r], v._M[r])
        assert np.allclose(v2._bias[r], v._bias[r])


# ---------------- OnlineLearner ----------------
def _settings():
    s = Settings.load()
    return s


def test_learner_record_agent_ops_builds_sample():
    lex = _lexicon()
    v = KeywordVectorizer(lex, dim=16)
    learner = OnlineLearner(v, _settings(), repo=None)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    key = l3[0].key
    q = v._e[lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)[0].key]
    n = learner.record_agent_ops(q, [{"op": "update", "key": key, "delta": 0.2}])
    assert n == 1
    assert len(learner.replay) == 1
    assert key in learner.replay[0].p_star


def test_learner_train_persists_with_repo(tmp_path):
    """训练后应把关系头/训练日志落库（doc-06 §6.5）。"""
    import sqlite3

    from dla.storage.migrator import migrate
    from dla.storage.repositories import SQLiteRepo
    from dla.storage.sqlite import get_connection

    lex = _lexicon()
    db = str(tmp_path / "lvm.db")
    conn = get_connection(db)
    migrate(conn, os.path.join(os.path.dirname(__file__), "..", "..", "migrations"))
    repo = SQLiteRepo(conn)
    v = KeywordVectorizer(lex, dim=16)
    learner = OnlineLearner(v, _settings(), repo=repo)
    l1 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    q = v._e[l1[0].key]
    for _ in range(5):
        learner.record_agent_ops(q, [{"op": "add", "key": l3[0].key, "intensity": 0.6}])
        learner.train(epochs=1)
    assert v.has_learned()
    state = repo.load_lvm_heads()
    assert state is not None
    assert state["step"] > 0
    logs = repo.recent_training_log(5)
    assert logs and all(r["loss"] is not None for r in logs)


def test_learner_record_satisfaction_logs_feedback(tmp_path):
    lex = _lexicon()
    import sqlite3

    from dla.storage.migrator import migrate
    from dla.storage.repositories import SQLiteRepo
    from dla.storage.sqlite import get_connection

    db = str(tmp_path / "fb.db")
    conn = get_connection(db)
    migrate(conn, os.path.join(os.path.dirname(__file__), "..", "..", "migrations"))
    repo = SQLiteRepo(conn)
    v = KeywordVectorizer(lex, dim=16)
    learner = OnlineLearner(v, _settings(), repo=repo)
    l3 = lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L3)
    q_prev = v._e[lex.for_layer(__import__("dla.core.models", fromlist=["Layer"]).Layer.L1)[0].key]
    n = learner.record_satisfaction("S1", 3, q_prev, [l3[0].key, l3[1].key], 0.8, "用户满意", based_on_turn=2)
    assert n == 1
    fb = repo.recent_feedback_signals("S1")
    assert fb and fb[0]["score"] == 0.8
