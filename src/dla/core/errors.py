"""DLA 类型化错误体系（doc/01 §9）。

所有自定义异常都继承自 ``DlaError``；客户端永远不应看到堆栈/密钥/内部路径。
"""

from __future__ import annotations


class DlaError(Exception):
    """所有 DLA 错误的基类。"""


class ConfigError(DlaError):
    """配置缺失/非法。"""


class LexiconError(DlaError):
    """词库加载失败（YAML 非法、白名单冲突等）。"""


class LlmError(DlaError):
    """上游 LLM 调用失败。"""


class LlmRateLimitError(LlmError):
    """触发限流。"""


class LlmSchemaError(LlmError):
    """结构化输出不合规。"""


class LlmTimeoutError(LlmError):
    """调用超时。"""


class AnalysisError(DlaError):
    """分析流程异常（抽取/判定/审批）。"""


class StorageError(DlaError):
    """存储读写异常。"""


class ToolError(DlaError):
    """工具插件执行异常。"""


class MemoryError_(DlaError):
    """冷热记忆子系统异常（避免与内建 MemoryError 冲突）。"""


class PersonaError(DlaError):
    """人格演进（doc/10）异常：冲突/偏离越界等。"""
