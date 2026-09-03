-- 005 · 补齐 context_compact_log 可观测字段（doc/11 §8.1）
--
-- 缺陷背景：001_init.sql 已用 CREATE TABLE IF NOT EXISTS 建过同名表（仅 7 列，
-- 缺 trigger_level / tokens_before / tokens_after）。原 004 用同样的
-- CREATE TABLE IF NOT EXISTS 试图补齐，因表已存在而**恒为 no-op**，
-- 导致 doc/11 §8.1 要求的可观测字段从未真正落库（PRAGMA 实测确认）。
--
-- 编号为 005 而非复用 004 的原因：迁移按「版本号是否已记入 schema_version」跳过，
-- 既有库（含开发库）已把 4 标记为已应用，留在 004 将永远不会被重跑。
-- 另起 005 可保证**新建库与既有库各恰好执行一次**（重建写法幂等，不丢数据）。
--
-- 采用 SQLite 标准「重建表」写法：建新表 → 拷数据 → 删旧表 → 改名。

CREATE TABLE IF NOT EXISTS context_compact_log_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    turn          INTEGER NOT NULL,
    trigger_level TEXT NOT NULL DEFAULT 'COMPACT',   -- WARN | COMPACT | HARD | MANUAL
    ratio_before  REAL NOT NULL,
    ratio_after   REAL NOT NULL,
    tokens_before INTEGER NOT NULL DEFAULT 0,
    tokens_after  INTEGER NOT NULL DEFAULT 0,
    actions_json  TEXT NOT NULL DEFAULT '[]',
    created_at    REAL NOT NULL
);

-- 迁移既有数据：旧表无 trigger_level/tokens_*，按保守默认值补齐
INSERT INTO context_compact_log_new
    (id, session_id, turn, trigger_level, ratio_before, ratio_after,
     tokens_before, tokens_after, actions_json, created_at)
SELECT id, session_id, turn, 'COMPACT', ratio_before, ratio_after, 0, 0,
       COALESCE(actions_json, '[]'), created_at
FROM context_compact_log;

DROP TABLE context_compact_log;

ALTER TABLE context_compact_log_new RENAME TO context_compact_log;

CREATE INDEX IF NOT EXISTS idx_compact_log_session ON context_compact_log(session_id, turn);
