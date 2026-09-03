-- LVM（本地向量化模型 / 在线学习，doc/06）存储
-- 关系头矩阵 M_r + 偏置、冻结嵌入缓存、训练日志、满意度反馈信号

-- 关系头权重（doc/06 §2.2）：每个 relation 一个 d×d 双线性矩阵 + 对 L3 端的偏置
CREATE TABLE IF NOT EXISTS lvm_relation_heads (
    relation    TEXT PRIMARY KEY,
    dim         INTEGER NOT NULL,
    matrix_json TEXT NOT NULL,
    bias_json   TEXT NOT NULL,
    step        INTEGER NOT NULL DEFAULT 0
);

-- 冻结关键词嵌入 e_k 缓存（doc-06 §2.1）；默认确定性重建，backbone 热启动时持久化
CREATE TABLE IF NOT EXISTS lvm_embeddings (
    keyword    TEXT PRIMARY KEY,
    dim        INTEGER NOT NULL,
    vec_json   TEXT NOT NULL
);

-- 训练日志（doc-06 §6.5）：每步 loss / lr / 样本数，供调试思维链展示
CREATE TABLE IF NOT EXISTS lvm_training_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        INTEGER NOT NULL,
    loss        REAL,
    lr          REAL,
    samples     INTEGER NOT NULL DEFAULT 0,
    head        TEXT,
    created_at  REAL NOT NULL
);

-- 满意度反馈信号（doc-06 §4.3 / §6.1 后向回路训练源）
CREATE TABLE IF NOT EXISTS feedback_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    turn          INTEGER NOT NULL,
    score         REAL NOT NULL,
    signal        TEXT,
    based_on_turn INTEGER,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback_signals(session_id, turn);
