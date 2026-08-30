"""结构化 JSON 日志（doc/01 §3 日志：structlog 风格 JSON 输出）。

零依赖实现：标准库 ``logging`` + 自定义 ``JsonFormatter``。若安装了 ``structlog`` 可切换，
但核心流程不依赖它。
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from typing import Optional


class JsonFormatter(logging.Formatter):
    """把日志记录渲染为一行 JSON（便于离线分析）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    *,
    force: bool = False,
) -> logging.Logger:
    """配置根日志器，返回名为 "dla" 的 logger。"""
    root = logging.getLogger()
    target_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(target_level)

    if force or not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        if json_format:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
        # 清掉旧 handler，避免重复
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(handler)

    return logging.getLogger("dla")
