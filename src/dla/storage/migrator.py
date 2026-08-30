"""版本化迁移执行器（doc/01 选型：自建 migration runner）。

依次执行 ``migrations/`` 下 ``NNN_*.sql``，已应用的版本记入 ``schema_version`` 表，幂等可重跑。
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

from ..core.errors import StorageError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate(conn, migrations_dir: str = "migrations") -> list[int]:
    """执行未应用的迁移，返回本次应用的版本号列表。"""
    if not os.path.isdir(migrations_dir):
        raise StorageError(f"迁移目录不存在: {migrations_dir}")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)"
    )
    conn.commit()
    cur.execute("SELECT version FROM schema_version")
    applied = {row[0] for row in cur.fetchall()}

    applied_this: list[int] = []
    for path in sorted(glob.glob(os.path.join(migrations_dir, "*.sql"))):
        base = os.path.basename(path)
        try:
            ver = int(base.split("_", 1)[0])
        except ValueError:
            continue
        if ver in applied:
            continue
        with open(path, encoding="utf-8") as fh:
            sql = fh.read()
        try:
            cur.executescript(sql)
            cur.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES(?, ?)",
                (ver, _now_iso()),
            )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            raise StorageError(f"迁移失败 {base}: {e}") from e
        applied_this.append(ver)
    return applied_this
