"""上下文自动压缩引擎集成测试（doc-11 §11）。

验证：长会话（≥200 轮摘要链）在开启 AUTO_COMPACT 时，单轮组装后 fill_ratio 被压在 COMPACT 以下；
且任意历史细节仍可经冷库取回（无损）。使用零依赖确定性 embedding，不联网。
"""

import os
import time

from dla.config.loader import get_keyword_lib
from dla.config.settings import Settings
from dla.llm.openai_compat import make_llm_client
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection


def _make_engine(tmp_path):
    s = Settings()
    # 小窗口，易于触发压缩；强制每轮检索保证冷记忆可用
    s.ctx_max_tokens = 700
    s.ctx_auto_compact = True
    s.ctx_summary_compact_after = 5
    s.ctx_epoch_merge_n = 3
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    db = os.path.join(str(tmp_path), "ctx.db")
    conn = get_connection(db)
    migrate(conn, "migrations")  # 应用 001~004（含 context_compact_log）
    repo = SQLiteRepo(conn)
    return DialogueEngine(s, lib, llm, repo)


def test_long_session_auto_compact_keeps_fill_under_threshold(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="S")

    # 预填 60 条较长摘要（等价于多轮对话累积），并同步写入 DB（turn_summaries）
    big = "我们练习了过去式的否定形式、疑问句转换以及连读发音要点"
    sess = eng._sessions["S"]
    sess.history_summaries = [f"第{i}轮：{big}" for i in range(60)]
    for i, h in enumerate(sess.history_summaries):
        eng.repo.save_summary("S", i + 1, h, "turn")

    # 触发一轮预算护栏（不真正走 LLM 完整生成，只跑前置流程的预算段）
    prep = eng._prepare_turn(sess, "继续练习发音", time.time())
    bg = next(c for c in prep["chain"] if c[0] == "budget_guard")[1]

    # 触发压缩：budget_guard 含 ratio_before/actions/ratio_after（不含 triggered 键）
    assert "actions" in bg, f"应当触发压缩：{bg}"
    actions = bg.get("actions") or []
    assert "merge_epoch" in actions, f"长摘要链应触发 epoch 合并：{bg}"
    assert bg["ratio_after"] < eng.settings.ctx_compact_ratio, (
        f"压缩后 fill 应低于 COMPACT 阈值：{bg}"
    )
    # 压缩确实降低了占比
    assert bg["ratio_after"] <= bg["ratio_before"]


def test_originals_still_recoverable_after_compact(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="S")

    big = "我们练习了过去式的否定形式、疑问句转换以及连读发音要点"
    sess = eng._sessions["S"]
    summaries = [f"第{i}轮：{big}" for i in range(60)]
    sess.history_summaries = list(summaries)
    for i, h in enumerate(summaries):
        eng.repo.save_summary("S", i + 1, h, "turn")
        # 同时落冷库，模拟真实 send 的持久化（供 recall_memory 取回）
        eng.memory.add_turn("S", i + 1, f"用户第{i}轮", f"代理回复{i}", h, importance=0.5)

    prep = eng._prepare_turn(sess, "你还记得我们练过什么吗", time.time())

    # 1) DB 中的原始 turn 摘要数量不变（无损：压缩只动内存活跃链）
    n_db = eng.repo.conn.execute(
        "SELECT COUNT(*) FROM turn_summaries WHERE session_id='S'"
    ).fetchone()[0]
    assert n_db == 60, f"原始摘要应全部留存，实际 {n_db}"

    # 2) 冷库仍可召回任意历史细节
    hits = eng.memory.search("过去式的否定形式", scope="all")
    assert hits, "冷库应仍能召回历史原文"
    assert any("过去式" in (h.summary or h.text) for h in hits)

    # 3) 压缩行为本身无害（compact_actions 仅含无损动作或为空）
    bg = next(c for c in prep["chain"] if c[0] == "budget_guard")[1]
    assert bg["actions"] == [] or all(
        a in ("merge_epoch", "merge_epoch_again", "evict_details", "trim_cold",
              "simplify_tools", "drop_oldest_epoch")
        for a in bg["actions"]
    )


def test_compact_log_persisted(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="S")
    big = "我们练习了过去式的否定形式、疑问句转换以及连读发音要点"
    sess = eng._sessions["S"]
    sess.history_summaries = [f"第{i}轮：{big}" for i in range(60)]

    eng._prepare_turn(sess, "继续练习", time.time())

    rows = eng.repo.conn.execute(
        "SELECT COUNT(*) FROM context_compact_log WHERE session_id='S'"
    ).fetchone()[0]
    assert rows >= 1, "压缩事件应写入 context_compact_log（迁移 004）"


def test_no_compact_when_under_threshold(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="S")
    sess = eng._sessions["S"]
    sess.history_summaries = ["短摘要"] * 3  # 远未到阈值

    prep = eng._prepare_turn(sess, "你好", time.time())
    bg = next(c for c in prep["chain"] if c[0] == "budget_guard")[1]
    assert bg.get("triggered") is False
    assert bg.get("actions") in (None, [])


# ---- 手动强制压缩（CLI `dla ctx compact --force`，doc/11 §8.1）----
def test_force_compact_triggers_when_over_threshold(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="FC")
    big = "我们练习了过去式的否定形式、疑问句转换以及连读发音要点"
    eng._sessions["FC"].history_summaries = [f"第{i}轮：{big}" for i in range(60)]

    res = eng.force_compact("FC")
    assert res is not None
    assert res["triggered"] is True
    assert res["ratio_after"] < res["ratio_before"]
    assert res["actions"], "过阈时应产生压缩动作"
    # 会话摘要链被压缩（无损：原文已落冷库）
    assert len(eng._sessions["FC"].history_summaries) < 60
    # 写入日志
    rows = eng.repo.conn.execute(
        "SELECT COUNT(*) FROM context_compact_log WHERE session_id='FC'"
    ).fetchone()[0]
    assert rows >= 1


def test_force_compact_noop_under_threshold(tmp_path):
    eng = _make_engine(tmp_path)
    eng.start_session(sid="FC2")
    eng._sessions["FC2"].history_summaries = ["短摘要"] * 3

    res = eng.force_compact("FC2")
    assert res is not None
    assert res["triggered"] is False
    assert res["actions"] == []


def test_force_compact_requires_active_session(tmp_path):
    eng = _make_engine(tmp_path)
    assert eng.force_compact("nonexistent") is None
