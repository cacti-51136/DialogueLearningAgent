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
    def log_compact(
        self,
        session_id: str,
        turn: int,
        ratio_before: float,
        ratio_after: float,
        actions: list,
        created_at: Optional[float] = None,
        *,
        trigger_level: str = "COMPACT",
        tokens_before: int = 0,
        tokens_after: int = 0,
    ) -> None:
        """记录一次上下文压缩（doc/11 §8.1 可观测）。

        新增的三个字段（trigger_level / tokens_before / tokens_after）此前虽在 004 里
        声明过，却因 001 已建同名表而从未真正落库（详见 migrations/005 注释）。
        此处使用关键字参数，兼容既有调用点。
        """
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO context_compact_log"
            "(session_id, turn, trigger_level, ratio_before, ratio_after, tokens_before, tokens_after, actions_json, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                turn,
                trigger_level,
                ratio_before,
                ratio_after,
                tokens_before,
                tokens_after,
                json.dumps(actions, ensure_ascii=False),
                created_at or time.time(),
            ),
        )
        self.conn.commit()

    def recent_compacts(self, session_id: str, limit: int = 20) -> List[dict]:
        """读取最近的压缩记录（doc/11 §8.1 可观测 / CLI `dla ctx compact`）。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn, trigger_level, ratio_before, ratio_after, tokens_before, tokens_after,"
            " actions_json, created_at FROM context_compact_log"
            " WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            try:
                d["actions"] = json.loads(d.pop("actions_json") or "[]")
            except (ValueError, TypeError):
                d["actions"] = []
            rows.append(d)
        return rows

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
    # ---- 词库操作日志（doc/03 §2.12 / doc-06 §4.2/§4.4）----
    def log_lexicon_op(
        self,
        session_id: str,
        turn: int,
        op_type: str,
        layer: str,
        target_key: Optional[str],
        payload: Optional[str],
        llm_reason: str = "",
        applied: int = 0,
        created_at: Optional[float] = None,
    ) -> None:
        """记录一次词库操作（scene_ops / agent_ops）。applied=0 表示被护栏拒绝。

        ``payload`` 建议传原始 op 的 JSON，便于审计与回放。
        """
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO lexicon_ops"
            "(session_id, turn, op_type, layer, target_key, payload, llm_reason, applied, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id, turn, op_type, layer, target_key, payload,
                llm_reason, int(applied), created_at or time.time(),
            ),
        )
        self.conn.commit()

    def recent_lexicon_ops(self, session_id: str, limit: int = 50) -> List[dict]:
        """读取最近的词库操作记录（doc-06 可观测 / 审计）。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, turn, op_type, layer, target_key, payload, llm_reason, applied, created_at"
            " FROM lexicon_ops WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    # ---- LVM（本地向量化模型 / 在线学习，doc/06）----
    def save_lvm_heads(self, state: dict, embeddings: dict) -> None:
        """保存 LVM 关系头矩阵 / 偏置 / 步数与冻结嵌入（doc/06 §6.5）。"""
        cur = self.conn.cursor()
        dim = int(state.get("dim", 0))
        for relation, head in state.get("heads", {}).items():
            cur.execute(
                "INSERT INTO lvm_relation_heads(relation, dim, matrix_json, bias_json, step) VALUES(?,?,?,?,?) "
                "ON CONFLICT(relation) DO UPDATE SET dim=excluded.dim, matrix_json=excluded.matrix_json, "
                "bias_json=excluded.bias_json, step=excluded.step",
                (relation, dim, json.dumps(head["matrix"], ensure_ascii=False),
                 json.dumps(head["bias"], ensure_ascii=False), int(state.get("step", 0))),
            )
        if embeddings:
            for keyword, vec in embeddings.items():
                cur.execute(
                    "INSERT INTO lvm_embeddings(keyword, dim, vec_json) VALUES(?,?,?) "
                    "ON CONFLICT(keyword) DO UPDATE SET dim=excluded.dim, vec_json=excluded.vec_json",
                    (keyword, dim, json.dumps(vec, ensure_ascii=False)),
                )
        self.conn.commit()

    def load_lvm_heads(self) -> Optional[dict]:
        """读取已保存的 LVM 状态（matrix/bias/step），无数据返回 None。"""
        cur = self.conn.cursor()
        cur.execute("SELECT relation, dim, matrix_json, bias_json, step FROM lvm_relation_heads")
        rows = cur.fetchall()
        if not rows:
            return None
        heads = {}
        dim = 0
        for r in rows:
            dim = int(r["dim"])
            heads[r["relation"]] = {
                "matrix": json.loads(r["matrix_json"]),
                "bias": json.loads(r["bias_json"]),
            }
            step = int(r["step"])
        return {"dim": dim, "step": step, "heads": heads}

    def log_training_step(self, step: int, loss: float, lr: float, samples: int, head: str) -> None:
        """记录一次 LVM 训练步（doc-06 §6.5 可观测）。"""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO lvm_training_log(step, loss, lr, samples, head, created_at) VALUES(?,?,?,?,?,?)",
            (step, float(loss), float(lr), int(samples), head, time.time()),
        )
        self.conn.commit()

    def recent_training_log(self, limit: int = 20) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT step, loss, lr, samples, head, created_at FROM lvm_training_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def log_feedback_signal(
        self,
        session_id: Optional[str],
        turn: int,
        score: float,
        signal: str = "",
        based_on_turn: Optional[int] = None,
        created_at: Optional[float] = None,
    ) -> None:
        """记录一条满意度反馈信号（doc-06 §4.3）。"""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO feedback_signals(session_id, turn, score, signal, based_on_turn, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (session_id, int(turn), float(score), signal or "", based_on_turn, created_at or time.time()),
        )
        self.conn.commit()

    def recent_feedback_signals(self, session_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        cur = self.conn.cursor()
        if session_id:
            cur.execute(
                "SELECT session_id, turn, score, signal, based_on_turn, created_at "
                "FROM feedback_signals WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            cur.execute(
                "SELECT session_id, turn, score, signal, based_on_turn, created_at "
                "FROM feedback_signals ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]

    def lvm_reset(self) -> None:
        """清空 LVM 全部状态（doc-06 §9：dla lvm reset）。"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM lvm_relation_heads")
        cur.execute("DELETE FROM lvm_embeddings")
        cur.execute("DELETE FROM lvm_training_log")
        cur.execute("DELETE FROM feedback_signals")
        self.conn.commit()

    def log_tool_call(self, session_id: str, tool: str, args_json: str = "{}", ok: int = 1, error: str = "", created_at: Optional[float] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO tool_log(session_id, tool, args_json, ok, error, created_at) VALUES(?,?,?,?,?,?)",
            (session_id, tool, args_json, int(ok), error, created_at or time.time()),
        )
        self.conn.commit()
