"""仓储实现（doc/03 Repository 接口 / doc/01 选型：sqlite3）。

``SQLiteRepo`` 封装所有持久化操作：消息、权重快照、压缩摘要链、人格演进、上下文压缩日志、
kw_agent_map 映射（doc/03 §2.15）。仓储层与引擎解耦，便于替换后端。
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from ..core.models import TurnSummary, WeightSnapshot


class SQLiteRepo:
    def __init__(self, conn) -> None:
        self.conn = conn

    # ---- messages ----
    def add_message(self, session_id: str, turn: int, role: str, content: str, created_at: Optional[float] = None) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages(session_id, turn, role, content, created_at) VALUES(?,?,?,?,?)",
            (session_id, turn, role, content, created_at or time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    # ---- weight snapshots ----
    def save_snapshot(self, session_id: str, snapshot: WeightSnapshot, created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO weight_snapshots(session_id, turn, l1_json, l2_json, l3_json, created_at) VALUES(?,?,?,?,?,?)",
            (
                session_id,
                snapshot.turn,
                json.dumps(snapshot.l1, ensure_ascii=False),
                json.dumps(snapshot.l2, ensure_ascii=False),
                json.dumps(snapshot.l3, ensure_ascii=False),
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def get_latest_snapshot(self, session_id: str) -> Optional[WeightSnapshot]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, l1_json, l2_json, l3_json FROM weight_snapshots WHERE session_id=? ORDER BY turn DESC LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return WeightSnapshot(
            turn=row["turn"],
            l1=json.loads(row["l1_json"]),
            l2=json.loads(row["l2_json"]),
            l3=json.loads(row["l3_json"]),
        )

    # ---- turn summaries ----
    def save_summary(self, session_id: str, turn: int, text: str, kind: str = "turn", created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO turn_summaries(session_id, turn, text, kind, created_at) VALUES(?,?,?,?,?)",
            (session_id, turn, text, kind, created_at or time.time()),
        )
        self.conn.commit()

    def list_recent_summaries(self, session_id: str, limit: int = 50) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT text FROM turn_summaries WHERE session_id=? ORDER BY turn DESC LIMIT ?",
            (session_id, limit),
        )
        return [r["text"] for r in cur.fetchall()][::-1]

    # ---- persona (doc/10) ----
    def save_persona_spec(self, session_id: str, turn: int, spec_text: str, version: int, is_baseline: bool = False, created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO persona_changes(session_id, turn, spec_text, version, is_baseline, created_at) VALUES(?,?,?,?,?,?)",
            (session_id, turn, spec_text, version, 1 if is_baseline else 0, created_at or time.time()),
        )
        self.conn.commit()

    def get_persona_specs(self, session_id: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, spec_text, version, is_baseline FROM persona_changes WHERE session_id=? ORDER BY version ASC",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ---- context compact log (doc/11) ----
    def log_compact(self, session_id: str, turn: int, ratio_before: float, ratio_after: float, actions: list, created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO context_compact_log(session_id, turn, ratio_before, ratio_after, actions_json, created_at) VALUES(?,?,?,?,?,?)",
            (session_id, turn, ratio_before, ratio_after, json.dumps(actions, ensure_ascii=False), created_at or time.time()),
        )
        self.conn.commit()

    # ---- kw_agent_map (doc/03 §2.15) ----
    def kwmap_upsert(self, src: str, dst: str, direction: str, delta_grad: float, learn_rate: float = 0.05) -> None:
        """按学习率累积更新一条映射（doc/02 §11.9 / doc/03 §2.15）。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT delta, observed_count, confidence, learn_rate FROM kw_agent_map WHERE src_keyword=? AND dst_keyword=?",
            (src, dst),
        )
        row = cur.fetchone()
        if row is None:
            delta = max(0.0, min(1.0, delta_grad))
            cur.execute(
                "INSERT INTO kw_agent_map(src_keyword, dst_keyword, direction, delta, observed_count, confidence, learn_rate) VALUES(?,?,?,?,?,?,?)",
                (src, dst, direction, delta, 1, 0.5, learn_rate),
            )
        else:
            # 指数滑动累积：delta <- clamp(delta + lr*grad)
            new_delta = max(0.0, min(1.0, row["delta"] + learn_rate * delta_grad))
            cur.execute(
                "UPDATE kw_agent_map SET delta=?, observed_count=observed_count+1 WHERE src_keyword=? AND dst_keyword=?",
                (new_delta, src, dst),
            )
        self.conn.commit()

    def kwmap_lookup_by_src(self, src: str) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT dst_keyword, direction, delta FROM kw_agent_map WHERE src_keyword=? ORDER BY delta DESC",
            (src,),
        )
        return [dict(r) for r in cur.fetchall()]

    def kwmap_reset(self) -> None:
        self.conn.cursor().execute("DELETE FROM kw_agent_map")
        self.conn.commit()

    # ---- 会话列表 / 历史摘要（UI 多会话切换）----
    def list_session_ids(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM messages ORDER BY rowid DESC")
        return [r["session_id"] for r in cur.fetchall()]

    def recent_summaries(self, session_id: str, limit: int = 50) -> List[TurnSummary]:
        """返回该会话最近的摘要（turn DESC，最新在前），供会话恢复时重建历史链。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, text, kind, created_at FROM turn_summaries WHERE session_id=? ORDER BY turn DESC LIMIT ?",
            (session_id, limit),
        )
        return [
            TurnSummary(turn=r["turn"], text=r["text"], kind=r["kind"], timestamp=r["created_at"] or 0.0)
            for r in cur.fetchall()
        ]

    # ---- 消息历史（API GET /messages）----
    def list_messages(self, session_id: str, limit: int = 100) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, role, content FROM messages WHERE session_id=? ORDER BY rowid ASC LIMIT ?",
            (session_id, limit),
        )
        return [{"turn": r["turn"], "role": r["role"], "content": r["content"]} for r in cur.fetchall()]

    # ---- 权重快照历史（API GET /weights/history）----
    def list_snapshots(self, session_id: str, limit: int = 200) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, l1_json, l2_json, l3_json FROM weight_snapshots WHERE session_id=? ORDER BY turn ASC LIMIT ?",
            (session_id, limit),
        )
        return [
            {
                "turn": r["turn"],
                "l1": json.loads(r["l1_json"]),
                "l2": json.loads(r["l2_json"]),
                "l3": json.loads(r["l3_json"]),
            }
            for r in cur.fetchall()
        ]

    # ---- 工具调用日志（doc/08 G6）----
    def log_tool_call(self, session_id: str, tool: str, args_json: str = "{}", ok: int = 1, error: str = "", created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO tool_log(session_id, tool, args_json, ok, error, created_at) VALUES(?,?,?,?,?,?)",
            (session_id, tool, args_json, int(ok), error, created_at or time.time()),
        )
        self.conn.commit()
