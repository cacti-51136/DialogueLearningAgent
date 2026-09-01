"""端口抽象（doc/01 §4 外部依赖 Ports / D23 SpeechPort）。

定义引擎与外部世界之间的协议（Protocol），使核心引擎不依赖任何具体实现：
- ``LLMClient``：OpenAI 兼容对话完成。
- ``Clock``：时间源（可注入假时钟做测试）。
- ``SpeechPort``：语音输入/输出（本期 no-op，doc/01 D23）。
- ``ChatMessage`` / ``LlmResult``：端口间传输的结构。

核心包 **禁止 import** typer / qt / fastapi（doc/01 §4 关键约束）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, Sequence


@dataclass
class ChatMessage:
    """一条对话消息（端口间传输）。"""

    role: str  # system | user | assistant | tool
    content: str
    name: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class LlmResult:
    """LLM 一次完成的结果。"""

    content: str
    raw: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None
    usage: Optional[dict] = None


@dataclass
class ToolCallRequest:
    """LLM 请求调用本地工具的指令（doc/08）。"""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


class LLMClient(Protocol):
    """OpenAI 兼容的对话完成客户端。"""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> LlmResult:
        """同步完成一次对话。返回 ``LlmResult``。"""
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式生成，逐 token 返回文本片段（不含 system 注入的 turn_summary 标记）。"""
        ...


class Clock(Protocol):
    """时间源。"""

    def now(self) -> float:
        """返回 Unix 时间戳（秒）。"""
        ...


class SpeechPort(Protocol):
    """语音输入/输出端口（doc/01 D23，本期 no-op）。"""

    def transcribe(self, audio_bytes: bytes) -> str:
        """语音识别 → 文本。"""
        ...

    def synthesize(self, text: str) -> bytes:
        """文本 → 语音合成。"""
        ...


class NoOpSpeechPort:
    """``SpeechPort`` 的默认 no-op 实现（doc/01 D23）。"""

    def transcribe(self, audio_bytes: bytes) -> str:  # noqa: D401
        raise NotImplementedError("语音识别本期未实现（SpeechPort no-op）")

    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError("语音合成本期未实现（SpeechPort no-op）")
