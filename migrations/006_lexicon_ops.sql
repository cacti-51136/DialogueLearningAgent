-- 006 · 词库操作日志表（doc/03 §2.12 / doc-06 §4.2/§4.4）
--
-- 记录每轮分析产出的 scene_ops(L1) / agent_ops(L3) 即时增删改操作，可审计、可回放。
-- 此前这些操作被 analyze() 返回后**直接丢弃**（零消费、白花 token），本迁移配合
-- orchestration 层的 _consume_ops 一并补齐（见 #37）。
--
-- 与 keyword_candidates 的区别：候选表管"新概念是否收编进主词表"；本表管"运行期对
-- 当前场景工作集的即时增删改操作"，二者都落库但职责不同。
--
-- layer 字段覆盖 L1（scene_ops）/ L2（user 侧增删改，预留）/ L3（agent_ops）。

CREATE TABLE IF NOT EXISTS lexicon_ops (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    op_type     TEXT NOT NULL,       -- add | update | delete
    layer       TEXT NOT NULL,       -- L1 | L2 | L3
    target_key  TEXT,                -- 操作目标关键词（add 时可为空，由 LLM 建议）
    payload     TEXT,                -- JSON：delta / label / dimension / render_template / conf / 原始 op
    llm_reason  TEXT,
    applied     INTEGER NOT NULL DEFAULT 0,  -- 1=护栏通过已生效 0=被护栏拒绝（记录原因于 payload/llm_reason）
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lexicon_ops_session ON lexicon_ops(session_id, turn);
