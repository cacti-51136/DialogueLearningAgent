"""工具插件协议（doc/08 Tool 契约）。

统一 ``Tool`` 数据结构：``name`` / ``description`` / ``parameters``（JSON Schema 片段）/
``run``（处理函数）/ ``dangerous`` 标志（危险工具默认禁用，doc/08）。``ToolResult`` 为执行结果，
``ToolContext`` 为执行期注入的上下文（仓储 / 配置 / 会话等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _default_can_handle(text: str, ctx: ToolContext) -> float:
    """默认路由评分：未声明 ``can_handle`` 的工具默认不参与自动触发（doc-08 §4 两级路由）。"""
    return 0.0


@dataclass
class ToolContext:
    session_id: str
    repo: Any = None
    settings: Any = None
    memory: Any = None  # ColdMemoryStore（recall_memory 等按需取回原文用）
    clock: Any = None  # 可调用 .now() 的 Clock
    llm: Any = None  # LLMClient（工具内部如需二次调用模型）
    user_id: Optional[str] = None
    recent_summaries: List[str] = field(default_factory=list)
    recent_messages: List[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema 片段
    run: Callable[[dict, ToolContext], ToolResult]
    dangerous: bool = False
    is_readonly: bool = True  # 只读工具可自动触发（doc-08 §5）；有副作用工具应显式设为 False
    can_handle: Callable[[str, ToolContext], float] = _default_can_handle  # 路由评分（0~1；>=阈值才自动触发）


# 极简 JSON-Schema 校验（避免引入 jsonschema 依赖；仅校验必填项与基础类型）
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(tool: Tool, args: dict) -> Optional[str]:
    """校验 args 是否符合 ``tool.parameters``（JSON Schema 片段）。返回错误信息（None 表示通过）。"""
    schema = tool.parameters or {}
    if not isinstance(args, dict):
        return "参数必须是对象"
    required = set(schema.get("required", []))
    missing = required - set(args.keys())
    if missing:
        return f"缺少必填参数: {', '.join(sorted(missing))}"
    props = schema.get("properties", {})
    for key, val in args.items():
        spec = props.get(key)
        if not spec:
            continue  # 允许额外参数（宽松）
        expected = spec.get("type")
        if expected and expected in _TYPE_MAP and not isinstance(val, _TYPE_MAP[expected]):
            return f"参数 {key} 类型应为 {expected}"
    return None
