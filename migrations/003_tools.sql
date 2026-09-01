-- 003 · 工具插件调用日志（doc/08 G6 可观测）
CREATE TABLE IF NOT EXISTS tool_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    tool        TEXT NOT NULL,
    args_json   TEXT NOT NULL DEFAULT '{}',
    ok          INTEGER NOT NULL DEFAULT 1,
    error       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
