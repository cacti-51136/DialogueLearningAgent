"""SQLite 连接管理（doc/03 §4 / doc/01 选型：标准库 sqlite3）。

零依赖。提供连接工厂与上下文管理器。
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterator


def get_connection(path: str) -> sqlite3.Connection:
    """创建/打开 SQLite 连接（自动建父目录，启用外键与行工厂）。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class ConnectionGuard:
    """上下文管理器：自动提交/回滚。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn: sqlite3.Connection = get_connection(path)

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()
