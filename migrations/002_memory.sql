-- DLA 冷记忆 schema（doc/07 Cold Memory）
-- 与 doc/06 的 keyword_embeddings 物理隔离：本文件只建 cold_memory / memory_index 两张表

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
);
CREATE INDEX IF NOT EXISTS idx_cold_session ON cold_memory(session_id, deleted);

CREATE TABLE IF NOT EXISTS memory_index (
    id  INTEGER PRIMARY KEY,
    vec TEXT NOT NULL
);
