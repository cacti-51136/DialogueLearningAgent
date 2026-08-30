-- DLA 初始 schema（doc/03 数据模型）
-- 单列主表 + 权重快照 + 压缩摘要链 + 人格演进 + 上下文压缩日志 + kw_agent_map + LVM 检查点

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);

CREATE TABLE IF NOT EXISTS weight_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    l1_json     TEXT,
    l2_json     TEXT,
    l3_json     TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_session ON weight_snapshots(session_id, turn);

CREATE TABLE IF NOT EXISTS turn_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'turn',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS persona_changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    spec_text   TEXT,
    version     INTEGER NOT NULL DEFAULT 0,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS context_compact_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    ratio_before REAL,
    ratio_after  REAL,
    actions_json TEXT,
    created_at  REAL NOT NULL
);

-- kw_agent_map（doc/03 §2.15）：情绪/脾性词 → Agent 特质 权重映射，纯涌现无 YAML 来源
CREATE TABLE IF NOT EXISTS kw_agent_map (
    src_keyword   TEXT NOT NULL,
    dst_keyword   TEXT NOT NULL,
    direction     TEXT NOT NULL,            -- boost | suppress
    delta         REAL NOT NULL DEFAULT 0.15,
    observed_count INTEGER NOT NULL DEFAULT 0,
    confidence    REAL NOT NULL DEFAULT 0.0,
    learn_rate    REAL NOT NULL DEFAULT 0.05,
    PRIMARY KEY (src_keyword, dst_keyword)
);

CREATE TABLE IF NOT EXISTS lvm_checkpoints (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    step         INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   REAL NOT NULL
);
