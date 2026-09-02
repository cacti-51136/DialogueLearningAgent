-- 004 · 上下文自动压缩日志（doc/11 §8.1 可观测）
CREATE TABLE IF NOT EXISTS context_compact_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    turn          INTEGER NOT NULL,
    trigger_level TEXT NOT NULL DEFAULT 'COMPACT',
    ratio_before  REAL NOT NULL,
    ratio_after   REAL NOT NULL,
    tokens_before INTEGER NOT NULL DEFAULT 0,
    tokens_after  INTEGER NOT NULL DEFAULT 0,
    actions_json  TEXT NOT NULL DEFAULT '[]',
    created_at    REAL NOT NULL
);
