"""SQLite 连接管理（doc/03 §4 / doc/01 选型：标准库 sqlite3）。

零依赖。提供连接工厂与上下文管理器。
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterator


def get_connection(path: str, check_same_thread: bool = True) -> sqlite3.Connection:
    """创建/打开 SQLite 连接（自动建父目录，启用外键与行工厂）。

    ``check_same_thread=False`` 用于多线程场景（如 PyQt UI 的主线程建连、worker 线程
    经仓储写库）；调用方须保证跨线程访问是串行的（本项目 UI 在流式回合期间禁用发送、
    不会并发触碰连接，安全）。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # 多线程/多请求并发时避免瞬间 "database is locked"（本地单用户工具足够）
    conn.execute("PRAGMA busy_timeout = 5000")
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
