"""冷记忆存储（doc/07 Cold Memory）。

SQLite 持久化对话片段（turn）及其 embedding 与元数据；支持语义检索（余弦 top-K）、
按 importance 重排、相似度阈值过滤、跨/单会话 scope，以及软删除（遗忘）。
与 doc/06 的 ``keyword_embeddings`` 物理隔离：本模块只用 ``cold_memory`` / ``memory_index`` 两张表。

零依赖：embedding 默认走 :mod:`.embeddings` 的确定性实现；检索算法为纯 Python 余弦。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional

from .embeddings import Embedder


@dataclass
class ColdMemoryItem:
    """检索命中项（含相似度，供 UI / 调试展示）。"""

    mid: int
    session_id: str
    turn: int
    kind: str
    text: str
    summary: str
    importance: float
    similarity: float = 0.0
    created_at: float = 0.0

    @property
    def display(self) -> str:
        body = self.summary or self.text
        return f"[记忆·会话{self.session_id}第{self.turn}轮] {body}"


class ColdMemoryStore:
    """基于 SQLite 的冷记忆库（doc/07 §2.2 / §4）。"""

    def __init__(self, conn, settings, embedder: Optional[Embedder] = None) -> None:
        self.conn = conn
        self.dim = int(getattr(settings, "memory_embed_dim", 384))
        self.embedder = embedder or Embedder(
            self.dim, getattr(settings, "memory_embed_backbone", "")
        )
        self.ensure_schema()

    # ---- schema ----
    def ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cold_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                turn        INTEGER NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'turn',
                text        TEXT NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                importance  REAL NOT NULL DEFAULT 1.0,
                deleted     INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_index (
                id  INTEGER PRIMARY KEY,
                vec TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    # ---- write ----
    def add_turn(
        self,
        session_id: str,
        turn: int,
        user_text: str,
        agent_text: str,
        summary: str = "",
        importance: float = 1.0,
    ) -> int:
        """写入一轮对话（用户+代理原文 + 摘要）及其 embedding。返回 memory id。"""
        text = f"用户：{user_text}\n代理：{agent_text}"
        emb = self.embedder.embed(text + "\n" + (summary or ""))
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO cold_memory(session_id, turn, kind, text, summary, importance, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (session_id, turn, "turn", text, summary or "", float(importance), time.time()),
        )
        mid = cur.lastrowid
        cur.execute("INSERT INTO memory_index(id, vec) VALUES(?,?)", (mid, json.dumps(emb)))
        self.conn.commit()
        return mid

    def forget(self, mid: int) -> None:
        """软删除一条记忆（doc/07 §5 遗忘，可恢复）。"""
        self.conn.cursor().execute(
            "UPDATE cold_memory SET deleted=1 WHERE id=?", (mid,)
        )
        self.conn.commit()

    def prune_low_importance(self, keep_top_n: int) -> int:
        """保留 importance 最高的 keep_top_n 条，软删除其余（doc/07 §5 压缩的一种极简实现）。"""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT id FROM cold_memory WHERE deleted=0 ORDER BY importance DESC, created_at DESC"
        ).fetchall()
        if len(rows) <= keep_top_n:
            return 0
        to_delete = [r["id"] for r in rows[keep_top_n:]]
        for mid in to_delete:
            cur.execute("UPDATE cold_memory SET deleted=1 WHERE id=?", (mid,))
        self.conn.commit()
        return len(to_delete)

    # ---- read ----
    def search(
        self,
        query_text: str,
        top_k: int = 4,
        sim_threshold: float = 0.0,
        scope: str = "all",
        current_session: Optional[str] = None,
    ) -> List[ColdMemoryItem]:
        """以查询文本检索冷记忆，返回按 similarity×importance 重排后的 top-K。

        scope="current_session" 仅搜当前会话；否则搜全部会话（跨会话召回）。
        """
        qvec = self.embedder.embed(query_text)
        cur = self.conn.cursor()
        if scope == "current_session" and current_session:
            rows = cur.execute(
                "SELECT id, session_id, turn, kind, text, summary, importance, created_at "
                "FROM cold_memory WHERE deleted=0 AND session_id=?",
                (current_session,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT id, session_id, turn, kind, text, summary, importance, created_at "
                "FROM cold_memory WHERE deleted=0"
            ).fetchall()

        items: List[ColdMemoryItem] = []
        for r in rows:
            row = cur.execute("SELECT vec FROM memory_index WHERE id=?", (r["id"],)).fetchone()
            if row is None:
                continue
            vec = json.loads(row["vec"])
            sim = Embedder.cosine(qvec, vec)
            if sim < sim_threshold:
                continue
            items.append(
                ColdMemoryItem(
                    mid=r["id"],
                    session_id=r["session_id"],
                    turn=r["turn"],
                    kind=r["kind"],
                    text=r["text"],
                    summary=r["summary"],
                    importance=float(r["importance"]),
                    similarity=sim,
                    created_at=float(r["created_at"] or 0.0),
                )
            )
        # rerank：相似度 × 重要性（doc/07 §4.3 可选 rerank）
        items.sort(key=lambda it: -(it.similarity * max(0.0, it.importance)))
        return items[: max(0, top_k)]

    def count(self, session_id: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        if session_id:
            return cur.execute(
                "SELECT COUNT(*) FROM cold_memory WHERE deleted=0 AND session_id=?",
                (session_id,),
            ).fetchone()[0]
        return cur.execute("SELECT COUNT(*) FROM cold_memory WHERE deleted=0").fetchone()[0]
