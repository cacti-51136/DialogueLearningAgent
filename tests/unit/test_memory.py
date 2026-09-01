"""记忆子系统单测（doc/07 §4 / §9）。

覆盖：确定性 embedding 的稳定性与余弦相似度、冷库写入/检索 top-K 排序、相似度阈值过滤、
importance 重排、软删除、scope。检索使用可注入的 embed_fn 构造已知相似度，避免依赖真实语义。
"""

import json
import sqlite3
import time
import types

from dla.memory.embeddings import Embedder
from dla.memory.store import ColdMemoryStore


def _settings(dim=64):
    return types.SimpleNamespace(memory_embed_dim=dim, memory_embed_backbone="")


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _embed_fn_factory(dim):
    """把文本确定性映射到单位向量（索引 = hash(text) % dim），便于构造已知相似度。"""

    def fn(text):
        v = [0.0] * dim
        v[hash(text) % dim] = 1.0
        return v

    return fn


def _stored(user_text, agent_text, summary):
    """还原 add_turn 写入冷库时被 embedding 的完整文本（含 summary 后缀）。"""
    return f"用户：{user_text}\n代理：{agent_text}\n{summary}"


def test_cosine_basics():
    a = [1.0, 0.0, 0.0]
    assert Embedder.cosine(a, a) == 1.0
    assert Embedder.cosine(a, [0.0, 1.0, 0.0]) == 0.0
    assert Embedder.cosine(a, [1.0, 1.0, 0.0]) > 0.7
    assert Embedder.cosine(a, []) == 0.0


def test_deterministic_embedder_stable():
    e = Embedder(128)
    v1 = e.embed("我母语是粤语")
    v2 = e.embed("我母语是粤语")
    assert v1 == v2
    # 不同文本通常得到不同（但确定）向量
    v3 = e.embed("完全不同的内容 xyz")
    assert v1 != v3


def test_store_search_top_k_and_order():
    dim = 64
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=_embed_fn_factory(dim)))
    store.add_turn("s1", 1, "alpha", "r1", "sum-a")
    store.add_turn("s1", 2, "beta", "r2", "sum-b")

    q = _stored("alpha", "r1", "sum-a")  # 与第一条完全一致的文本 → 余弦 1.0
    hits = store.search(q, top_k=4, sim_threshold=0.0)
    assert len(hits) == 2
    assert hits[0].mid == 1  # 完全匹配排第一
    assert hits[1].mid == 2


def test_search_sim_threshold_filters():
    dim = 64
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=_embed_fn_factory(dim)))
    store.add_turn("s1", 1, "alpha", "r1", "sum-a")
    store.add_turn("s1", 2, "beta", "r2", "sum-b")

    q = _stored("beta", "r2", "sum-b")
    # 阈值设到 0.99：仅完全匹配的第二条通过，另一条（余弦 0）被过滤
    hits = store.search(q, top_k=4, sim_threshold=0.99)
    assert [h.mid for h in hits] == [2]


def test_search_top_k_limits():
    dim = 64
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=_embed_fn_factory(dim)))
    for i in range(5):
        store.add_turn("s1", i + 1, f"item-{i}", f"r{i}", f"sum-{i}")

    q = _stored("item-0", "r0", "sum-0")
    hits = store.search(q, top_k=2, sim_threshold=0.0)
    assert len(hits) == 2


def test_importance_rerank():
    dim = 64
    embed_fn = _embed_fn_factory(dim)
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=embed_fn))
    # 两条记忆与查询共享同一向量（相似度相同且非零），仅靠 importance 区分排序
    query = "用户：same\n代理：same"
    vec_q = json.dumps(embed_fn(query))
    cur = store.conn.cursor()
    cur.execute(
        "INSERT INTO cold_memory(session_id,turn,kind,text,summary,importance,created_at) VALUES(?,?,?,?,?,?,?)",
        ("s1", 1, "turn", "t", "low", 0.2, time.time()),
    )
    mid1 = cur.lastrowid
    cur.execute(
        "INSERT INTO cold_memory(session_id,turn,kind,text,summary,importance,created_at) VALUES(?,?,?,?,?,?,?)",
        ("s1", 2, "turn", "t", "high", 0.9, time.time()),
    )
    mid2 = cur.lastrowid
    cur.execute("INSERT INTO memory_index(id,vec) VALUES(?,?)", (mid1, vec_q))
    cur.execute("INSERT INTO memory_index(id,vec) VALUES(?,?)", (mid2, vec_q))
    store.conn.commit()

    hits = store.search(query, top_k=4, sim_threshold=0.0)
    assert len(hits) == 2
    assert hits[0].importance == 0.9
    assert hits[1].importance == 0.2


def test_forget_soft_deletes():
    dim = 64
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=_embed_fn_factory(dim)))
    mid = store.add_turn("s1", 1, "alpha", "r1", "sum-a")
    assert store.count() == 1

    store.forget(mid)
    assert store.count() == 0
    q = _stored("alpha", "r1", "sum-a")
    assert store.search(q, top_k=4, sim_threshold=0.0) == []


def test_scope_current_session():
    dim = 64
    store = ColdMemoryStore(_conn(), _settings(dim), embedder=Embedder(dim, embed_fn=_embed_fn_factory(dim)))
    store.add_turn("A", 1, "alpha", "r1", "sum-a")
    store.add_turn("B", 1, "beta", "r2", "sum-b")

    q = _stored("alpha", "r1", "sum-a")
    all_hits = store.search(q, top_k=4, sim_threshold=0.0, scope="all")
    cur_hits = store.search(q, top_k=4, sim_threshold=0.0, scope="current_session", current_session="A")
    assert len(all_hits) == 2
    assert [h.session_id for h in cur_hits] == ["A"]
